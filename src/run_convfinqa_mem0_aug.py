#!/usr/bin/env python3
"""
ConvFinQA Mem0-Augmented Runner + FinQA-aligned Evaluation Schema

This script mirrors the *FinQA baseline v2* CSV schema/column names so you can reuse
the same analysis pipeline.

Key points:
- Dataset: ConvFinQA conversation-level JSON (train/dev/test).
- Method: Mem0-augmented LLM (retrieve memories per turn and inject into prompt).
- Evaluation columns (aligned with FinQA CSV schema):
    idx, method, question, gold_answer, pred_answer,
    exact_match, numeric_close,
    parse_success, format_violation, has_multiple_numbers,
    abs_err, rel_err, judge_correct,
    latency_ms, search_ms,
    prompt_chars, output_chars,
    memory_items, memory_chars
- Extra helpful columns (non-breaking additions): dialog_id, turn_idx, split, model

Usage:
  python src/run_convfinqa_mem0_aug.py --split dev --max-samples 5 --out results/convfinqa_mem0_aug.csv --reset-store
  python src/run_convfinqa_mem0_aug.py --split dev --max-samples 5 --judge

Notes:
- ConvFinQA 'validation' is treated as 'dev' (your folder uses dev.json).
- Gold answers come from annotation.exe_ans_list when present.
"""
# Architecture Matters More Than Scale:
# A Comparative Study of Retrieval and Memory Augmentation for Financial QA
# Under SME Compute Constraints
#
# GitHub: https://github.com/JLiu24-Eng/Retrieval-and-Memory-Augmentation-for-Financial-QA-Study
# Authors: Jianan Liu, Jing Yang, Xianyou Li, Weiran Yan, Yichao Wu, Penghao Liang, Mengwei Yuan

import argparse
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from mem0 import Memory


# =========================
# Paths / Models
# =========================

DEFAULT_DATA_DIR = "data/ConvFinQA/dataset"
STORE_PATH = "./mem0_store_convfinqa"

# If you use Ollama locally, keep these aligned with your environment.
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")  # change if needed
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = OLLAMA_BASE_URL.rstrip("/") + "/api/chat"


FIN_SYSTEM = (
    "You are a financial reasoning assistant. "
    "Answer the question using ONLY the given context (and any remembered facts). "
    "If calculation is required, do it carefully. "
    "Return ONLY the final answer (a number or short phrase)."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator for financial QA.\n"
    "Given a QUESTION, GOLD answer, and PREDICTED answer, label the prediction as CORRECT or WRONG.\n"
    "If the predicted answer matches the gold numerically (allow reasonable rounding) or is an equivalent form "
    "(e.g., percent vs fraction when question asks percent), mark CORRECT.\n"
    "Reply with exactly one token: CORRECT or WRONG."
)


# =========================
# Dataset path resolution
# =========================

def resolve_dataset_path(split: str, base_dir: str) -> Path:
    s = split.lower().strip()
    if s in ("dev", "validation", "val"):
        filename = "dev.json"
    elif s == "train":
        filename = "train.json"
    elif s == "test":
        filename = "test.json"
    else:
        raise ValueError(f"Invalid split={split}. Use one of: train/dev/validation/test")

    p = Path(base_dir) / filename
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    return p


# =========================
# ConvFinQA parsing helpers
# =========================

def table_to_text(table_obj: Any, max_rows: int = 30) -> str:
    if table_obj is None:
        return ""
    if isinstance(table_obj, list):
        rows = table_obj[:max_rows]
        lines = []
        for r in rows:
            if isinstance(r, list):
                lines.append(" | ".join(str(x) for x in r))
            else:
                lines.append(str(r))
        if len(table_obj) > max_rows:
            lines.append(f"... ({len(table_obj)-max_rows} more rows)")
        return "\n".join(lines)
    return str(table_obj)


def get_turns_from_example(ex: Dict[str, Any]) -> List[str]:
    ann = ex.get("annotation") or {}
    if isinstance(ann, dict):
        db = ann.get("dialogue_break")
        if isinstance(db, list) and len(db) > 0:
            if isinstance(db[0], dict) and "question" in db[0]:
                return [d.get("question", "") for d in db]
            return [str(x) for x in db]

    if "dialogue" in ex and isinstance(ex["dialogue"], list):
        return [str(t.get("question") or t.get("q") or t.get("text") or t) for t in ex["dialogue"]]

    raise KeyError("Could not find dialogue turns. Expected annotation.dialogue_break or dialogue[]")


def get_gold_answers_from_example(ex: Dict[str, Any]) -> List[Any]:
    ann = ex.get("annotation") or {}
    if isinstance(ann, dict):
        lst = ann.get("exe_ans_list")
        if isinstance(lst, list):
            return lst
    return []


# =========================
# Ollama chat helper (FinQA-compatible)
# =========================

def ask_ollama(prompt: str, system: str = FIN_SYSTEM, model: str = DEFAULT_LLM_MODEL) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=240,
    )
    r.raise_for_status()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    text = r.json()["message"]["content"].strip()
    return {
        "text": text,
        "latency_ms": dt_ms,
        "prompt_chars": len(prompt),
        "output_chars": len(text),
    }


# =========================
# Mem0 helpers (same pattern as FinQA v2)
# =========================

def reset_store(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[reset] Deleted store at: {path}")


def make_mem0(store_path: str) -> Memory:
    return Memory.from_config({
        "vector_store": {"provider": "chroma", "config": {"path": store_path}},
        "embedder": {
            "provider": "ollama",
            "config": {"model": DEFAULT_EMBED_MODEL, "ollama_base_url": OLLAMA_BASE_URL}
        },
        "llm": {
            "provider": "ollama",
            "config": {"model": DEFAULT_LLM_MODEL, "ollama_base_url": OLLAMA_BASE_URL, "temperature": 0}
        }
    })


def build_base_context(pre_text: str, table_text: str, post_text: str) -> str:
    return f"[Pre-text]\n{pre_text}\n\n[Table]\n{table_text}\n\n[Post-text]\n{post_text}"


def build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nFinal Answer:"


def mem0_augmented_prompt(memory_results: Dict[str, Any], context: str, question: str) -> Tuple[str, Dict[str, Any]]:
    remembered = []
    if isinstance(memory_results, dict) and "results" in memory_results:
        for r in memory_results["results"]:
            if isinstance(r, dict) and r.get("memory"):
                remembered.append(r["memory"])
    remembered_text = "\n".join(f"- {x}" for x in remembered) if remembered else "(none)"

    prompt = (
        f"Remembered facts (from long-term memory):\n{remembered_text}\n\n"
        + build_prompt(context, question)
    )
    stats = {
        "memory_items": len(remembered),
        "memory_chars": len(remembered_text),
    }
    return prompt, stats


# =========================
# Evaluation helpers (copied/compatible with FinQA baseline v2)
# =========================

_NUM_TOKEN_RE = re.compile(
    r"""
    (?P<paren>\(\s*)?                  # optional opening parenthesis for accounting negative
    (?P<sign>[-+])?                      # optional explicit sign
    (?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)   # number
    (?P<paren_end>\s*\))?              # optional closing parenthesis
    (?P<pct>\s*%)?                      # optional percent sign
    """,
    re.VERBOSE,
)


def extract_numbers_with_flags(text: str) -> List[Tuple[float, bool]]:
    if text is None:
        return []
    t = str(text).strip()
    if not t:
        return []
    out: List[Tuple[float, bool]] = []
    for m in _NUM_TOKEN_RE.finditer(t):
        s = m.group("num").replace(",", "")
        try:
            v = float(s)
        except Exception:
            continue
        has_parens = (m.group("paren") is not None) and (m.group("paren_end") is not None)
        sign = m.group("sign")
        if has_parens and sign != "-":
            v = -v
        elif sign == "-":
            v = -abs(v)
        has_pct = (m.group("pct") is not None)
        out.append((v, has_pct))
    return out


def extract_last_number_with_flags(text: str) -> Optional[Tuple[float, bool]]:
    nums = extract_numbers_with_flags(text)
    if not nums:
        return None
    return nums[-1]


def decimals_in_gold(gold: str) -> int:
    if gold is None:
        return 0
    s = str(gold).replace(",", "")
    m = re.search(r"\d+(?:\.(\d+))?", s)
    return len(m.group(1)) if m and m.group(1) else 0


def question_wants_percent(question: str) -> bool:
    q = (question or "").lower()
    return ("percent" in q) or ("percentage" in q) or ("%" in q)


def normalize_pred_to_gold_scale(question: str, gold: str, pred: str) -> Optional[Tuple[float, float, bool]]:
    """Return (pred_value, gold_value, want_pct) after best-candidate selection + percent normalization."""
    g_last = extract_last_number_with_flags(gold)
    if g_last is None:
        return None
    gval, g_has_pct = g_last

    want_pct = g_has_pct or question_wants_percent(question) or ("%" in str(gold or "")) or ("%" in str(pred or ""))

    p_nums = extract_numbers_with_flags(pred)
    if not p_nums:
        return None

    best_p = None
    best_err = None
    for pv, pv_has_pct in p_nums:
        pvv = pv
        if want_pct and (not pv_has_pct) and 0 <= abs(pvv) <= 1.5:
            pvv *= 100.0
        err = abs(pvv - gval)
        if best_err is None or err < best_err:
            best_err = err
            best_p = pvv

    if best_p is None:
        return None
    return (best_p, gval, want_pct)


def parse_pred_diagnostics(pred: str) -> Tuple[bool, bool, bool]:
    """Returns: (parse_success, format_violation, has_multiple_numbers)"""
    if pred is None:
        return (False, True, False)
    t = str(pred).strip()
    nums = extract_numbers_with_flags(t)
    parse_success = len(nums) > 0

    # format_violation: we asked for final answer only; flag if it looks like a long explanation
    # Heuristic: multiple lines or contains typical reasoning markers
    format_violation = ("\n" in t) or (len(t) > 80 and any(k in t.lower() for k in ["because", "therefore", "so ", "step", "calc", "equation", "="]))
    has_multiple_numbers = len(nums) >= 2
    return (parse_success, format_violation, has_multiple_numbers)


def error_metrics(question: str, gold: str, pred: str) -> Optional[Dict[str, float]]:
    norm = normalize_pred_to_gold_scale(question, gold, pred)
    if norm is None:
        return None
    pval, gval, _want_pct = norm
    abs_err = abs(pval - gval)
    rel_err = abs_err / (abs(gval) + 1e-9)
    return {"abs_err": abs_err, "rel_err": rel_err}


def exact_match(pred: str, gold: str, question: str = "") -> bool:
    norm = normalize_pred_to_gold_scale(question, gold, pred)
    if norm is None:
        return False
    pval, gval, _want_pct = norm
    decs = decimals_in_gold(gold)
    if decs == 0:
        return int(round(pval)) == int(round(gval))
    return round(pval, decs) == round(gval, decs)


def numeric_close(pred: str, gold: str, question: str = "", atol: float = 1e-2, rtol: float = 1e-2) -> bool:
    norm = normalize_pred_to_gold_scale(question, gold, pred)
    if norm is None:
        return False
    pval, gval, want_pct = norm
    decs = decimals_in_gold(gold)

    if want_pct:
        if decs == 0:
            abs_tol = 0.5
        elif decs == 1:
            abs_tol = 0.15
        else:
            abs_tol = 0.05
        return math.isclose(pval, gval, abs_tol=abs_tol, rel_tol=0.002)

    if decs == 0:
        abs_tol = 0.5 if abs(gval) < 1000 else max(1.0, abs(gval) * 0.001)
        return abs(pval - gval) <= abs_tol

    if decs == 1:
        return abs(pval - gval) <= 0.15

    return abs(pval - gval) <= 0.05


def judge_answer(question: str, gold: str, pred: str, model: str) -> Optional[bool]:
    prompt = f"QUESTION:\n{question}\n\nGOLD:\n{gold}\n\nPREDICTED:\n{pred}\n\nLabel:"             "\n(Reply with exactly one token: CORRECT or WRONG)"
    resp = ask_ollama(prompt, system=JUDGE_SYSTEM, model=model)
    text = resp["text"].strip().upper()
    if text.startswith("CORRECT"):
        return True
    if text.startswith("WRONG"):
        return False
    return None


# =========================
# Progress helpers
# =========================

def percentile(xs: List[float], p: int) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(ys[f])
    return float(ys[f] + (ys[c] - ys[f]) * (k - f))


def _short(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


# =========================
# Main
# =========================

def run(split: str, max_samples: int, max_turns: Optional[int], history_turns: Optional[int],
        out_csv: str, reset: bool, judge: bool, print_every: int, print_q_chars: int,
        store_path: str, llm_model: str) -> None:

    if reset:
        reset_store(store_path)

    mem = make_mem0(store_path)

    dataset_path = resolve_dataset_path(split, DEFAULT_DATA_DIR)
    print(f"[INFO] Loading split={split} from: {dataset_path}", flush=True)

    with open(dataset_path, "r") as f:
        conversations = json.load(f)
    if not isinstance(conversations, list):
        raise ValueError("Expected dataset JSON root to be a list of conversations")

    if max_samples is not None:
        conversations = conversations[:max_samples]

    # Estimate total turns
    est_total_turns = 0
    for ex in conversations:
        try:
            t = get_turns_from_example(ex)
            if max_turns is not None:
                t = t[:max_turns]
            est_total_turns += len(t)
        except Exception:
            pass

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV schema aligned with FinQA baseline v2
    rows: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    search_latencies_ms: List[float] = []

    global_idx = -1
    start_wall = time.time()

    user_id = "convfinqa_user"  # fixed for reproducibility

    for d_i, ex in enumerate(conversations):
        dialog_id = ex.get("id", f"dialog_{d_i}")

        pre_text = " ".join(ex.get("pre_text", [])) if isinstance(ex.get("pre_text"), list) else str(ex.get("pre_text", ""))
        post_text = " ".join(ex.get("post_text", [])) if isinstance(ex.get("post_text"), list) else str(ex.get("post_text", ""))
        table_text = table_to_text(ex.get("table"))

        base_context = build_base_context(pre_text, table_text, post_text)

        # Add dialog base context ONCE to memory (less noisy than adding per turn)
        mem.add(base_context, user_id=user_id, metadata={"type": "dialog_context", "dialog_id": dialog_id})

        turns = get_turns_from_example(ex)
        golds = get_gold_answers_from_example(ex)
        if max_turns is not None:
            turns = turns[:max_turns]
            if golds:
                golds = golds[:max_turns]

        # keep our own short history for prompt (separate from Mem0 store)
        history: List[Tuple[str, str]] = []

        for turn_idx, question in enumerate(turns):
            global_idx += 1
            if history_turns is None:
                hist_for_prompt = history
            else:
                hist_for_prompt = history[-history_turns:] if history_turns > 0 else []

            # Build context with short history (so baseline behavior is similar across methods)
            hist_text = ""
            if hist_for_prompt:
                hist_text = "\n".join([f"Turn {i+1} Q: {q}\nTurn {i+1} A: {a}" for i, (q, a) in enumerate(hist_for_prompt)])
            context_for_turn = base_context + ("\n\n[Dialogue History]\n" + hist_text if hist_text else "\n\n[Dialogue History]\n(none)")

            gold = golds[turn_idx] if (golds and turn_idx < len(golds)) else None

            # Mem0 search (separate latency)
            t_search0 = time.perf_counter()
            retrieved = mem.search(question, user_id=user_id)
            search_ms = (time.perf_counter() - t_search0) * 1000.0
            search_latencies_ms.append(search_ms)

            prompt, mem_stats = mem0_augmented_prompt(retrieved, context_for_turn, question)

            # LLM call
            resp = ask_ollama(prompt, system=FIN_SYSTEM, model=llm_model)
            pred = resp["text"]

            # Add this turn to memory for future turns
            mem.add(f"Q: {question}\nA: {pred}", user_id=user_id,
                    metadata={"type": "qa_turn", "dialog_id": dialog_id, "turn_idx": turn_idx})

            # Eval
            ps, fv, multi = parse_pred_diagnostics(pred)
            err = error_metrics(question, str(gold) if gold is not None else "", pred) if gold is not None else None
            j_ok = judge_answer(question, str(gold), pred, llm_model) if (judge and gold is not None) else None

            row = {
                "idx": global_idx,
                "method": "mem0_augmented",
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,
                "exact_match": exact_match(pred, str(gold), question) if gold is not None else False,
                "numeric_close": numeric_close(pred, str(gold), question) if gold is not None else False,
                "parse_success": ps,
                "format_violation": fv,
                "has_multiple_numbers": multi,
                "abs_err": (round(err["abs_err"], 6) if err else None),
                "rel_err": (round(err["rel_err"], 6) if err else None),
                "judge_correct": j_ok,
                "latency_ms": round(resp["latency_ms"], 1),
                "search_ms": round(search_ms, 1),
                "prompt_chars": resp["prompt_chars"],
                "output_chars": resp["output_chars"],
                "memory_items": mem_stats.get("memory_items"),
                "memory_chars": mem_stats.get("memory_chars"),

                # extra columns (safe additions)
                "dialog_id": dialog_id,
                "turn_idx": turn_idx,
                "split": split,
                "model": llm_model,
            }

            rows.append(row)
            latencies_ms.append(resp["latency_ms"])

            # per-turn print
            if est_total_turns > 0:
                print(f"[{global_idx}/{est_total_turns}] mem0_aug dialog={dialog_id} turn={turn_idx} exact={row['exact_match']} tol={row['numeric_close']} parse={ps} judge={j_ok} latency_ms={row['latency_ms']} search_ms={row['search_ms']} pred={_short(pred, 60)} gold={gold}", flush=True)
            else:
                print(f"[{global_idx}] mem0_aug dialog={dialog_id} turn={turn_idx} exact={row['exact_match']} tol={row['numeric_close']} parse={ps} judge={j_ok} latency_ms={row['latency_ms']} search_ms={row['search_ms']}", flush=True)

            history.append((question, pred))

            if print_every and ((global_idx + 1) % print_every == 0):
                elapsed = time.time() - start_wall
                p50 = percentile([x for x in latencies_ms], 50)
                p95 = percentile([x for x in latencies_ms], 95)
                sp50 = percentile([x for x in search_latencies_ms], 50)
                sp95 = percentile([x for x in search_latencies_ms], 95)
                print(f"[PROGRESS] turns={global_idx+1} elapsed={elapsed/60:.1f}m latency_p50={p50:.1f}ms latency_p95={p95:.1f}ms search_p50={sp50:.1f}ms search_p95={sp95:.1f}ms", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # Summary
    p50 = percentile(latencies_ms, 50)
    p95 = percentile(latencies_ms, 95)
    sp50 = percentile(search_latencies_ms, 50)
    sp95 = percentile(search_latencies_ms, 95)
    print(f"\nSaved results to: {out_path}")
    if p50 is not None and p95 is not None:
        print(f"Latency p50={p50:.1f}ms p95={p95:.1f}ms")
    if sp50 is not None and sp95 is not None:
        print(f"Mem0 search latency p50={sp50:.1f}ms p95={sp95:.1f}ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["train", "dev", "validation", "test"])
    p.add_argument("--max-samples", type=int, default=20, help="Limit number of dialogues (conversations)")
    p.add_argument("--max-turns", type=int, default=None, help="Limit turns per dialogue")
    p.add_argument("--history-turns", type=int, default=None, help="How many previous turns to include in prompt")
    p.add_argument("--out", default="results/convfinqa_mem0_augmented.csv")
    p.add_argument("--reset-store", action="store_true", help="Reset Mem0 store before run")
    p.add_argument("--judge", action="store_true", help="Enable LLM-as-judge (adds judge_correct)")
    p.add_argument("--print-every", type=int, default=25, help="Print progress summary every N turns")
    p.add_argument("--print-question-chars", type=int, default=80, help="(reserved)" )
    p.add_argument("--store-path", default=STORE_PATH)
    p.add_argument("--model", default=DEFAULT_LLM_MODEL, help="Ollama model name" )
    args = p.parse_args()

    run(
        split=args.split,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        history_turns=args.history_turns,
        out_csv=args.out,
        reset=args.reset_store,
        judge=args.judge,
        print_every=args.print_every,
        print_q_chars=args.print_question_chars,
        store_path=args.store_path,
        llm_model=args.model,
    )


if __name__ == "__main__":
    main()
