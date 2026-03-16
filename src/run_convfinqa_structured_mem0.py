#!/usr/bin/env python3
"""
ConvFinQA Structured Mem0 Runner (dialogue-scoped) + FinQA Schema CSV

Goal
- Provide a ConvFinQA "Structured Mem0" method that mirrors the FinQA structured-mem0 CSV schema
  (same evaluation metric column names/order as your FinQA structured script).

Method
- For each dialogue:
  - Convert the dialogue table into structured facts: "{entity} | {column} = {value}"
  - Add pre/post text lines as contextual facts
  - Store facts in Mem0 (scoped by a per-dialogue user_id to avoid cross-dialogue leakage)
  - For each turn:
      - Retrieve top-k facts from Mem0 using the current question
      - Build a FACTS-only prompt and call Ollama
      - Store the current Q/A back to Mem0 as a structured turn memory

Output CSV (FinQA schema)
fieldnames=[
  "idx","method","question","gold_answer","pred_answer",
  "exact_match","numeric_close",
  "parse_success","format_violation","has_multiple_numbers",
  "abs_err","rel_err","judge_correct",
  "latency_ms","search_ms","prompt_chars","output_chars",
  "n_facts_stored","n_facts_retrieved","memory_items","memory_chars",
]

Notes
- ConvFinQA split: dev.json for dev/validation, train.json, test.json
- Gold answers: annotation.exe_ans_list (turn-aligned)
"""
# Architecture Matters More Than Scale:
# A Comparative Study of Retrieval and Memory Augmentation for Financial QA
# Under SME Compute Constraints
#
# GitHub: https://github.com/JLiu24-Eng/Retrieval-and-Memory-Augmentation-for-Financial-QA-Study
# Authors: Jianan Liu, Jing Yang, Xianyou Li, Weiran Yan, Yichao Wu, Penghao Liang, Mengwei Yuan

import argparse
import csv
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from mem0 import Memory

# -------------------------
# Defaults / env
# -------------------------
DEFAULT_DATA_DIR = "data/ConvFinQA/dataset"
DEFAULT_STORE_PATH = "./mem0_store_convfinqa_structured"
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = OLLAMA_BASE_URL.rstrip("/") + "/api/chat"

FIN_SYSTEM = (
    "You are a financial question answering assistant.\n"
    "Use ONLY the FACTS provided.\n"
    "You MAY derive values using standard arithmetic.\n"
    "Do NOT invent or assume new facts beyond the provided FACTS.\n"
    "Do NOT rescale numbers (e.g., do not turn 637 into 637,000,000).\n"
    "Do NOT convert units unless the question explicitly asks for conversion.\n"
    "If numerator and denominator share the same unit, the unit cancels; use raw numbers.\n"
    "Return ONLY the final numeric answer (optionally with %)."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator for financial QA.\n"
    "Given a QUESTION, GOLD answer, and PREDICTED answer, label the prediction as CORRECT or WRONG.\n"
    "If the predicted answer matches the gold numerically (allow reasonable rounding) or is an equivalent form "
    "(e.g., percent vs fraction when question asks percent), mark CORRECT.\n"
    "Reply with exactly one token: CORRECT or WRONG."
)

# -------------------------
# Dataset helpers
# -------------------------
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

def normalize_text_lines(x: Any, max_lines: int = 80) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for t in x:
            s = str(t).strip()
            if s:
                out.append(s)
        return out[:max_lines]
    s = str(x).strip()
    return [s] if s else []

# -------------------------
# Structured facts
# -------------------------
def _norm_cell(s: Any) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def table_to_facts(table: Any, max_rows: int = 40) -> List[str]:
    """
    Convert table (list of rows) into row-level facts.
    Heuristic:
      - first row is header
      - first column is entity name
      - create facts: "{entity} | {col} = {value}"
    """
    if not table or not isinstance(table, list) or len(table) < 2:
        return []
    if not isinstance(table[0], list):
        # best-effort: table is list of strings or mixed
        rows = []
        for r in table[:max_rows]:
            if isinstance(r, str):
                parts = [p.strip() for p in r.split("|")]
                rows.append(parts)
            else:
                rows.append([str(r)])
        if len(rows) < 2 or not isinstance(rows[0], list):
            return []
        table = rows

    header = [_norm_cell(x).lower() for x in table[0]]
    facts: List[str] = []
    for row in table[1:1 + max_rows]:
        if not isinstance(row, list) or len(row) < 2:
            continue
        entity = _norm_cell(row[0]).lower()
        if not entity:
            continue
        for j in range(1, min(len(row), len(header))):
            col = header[j] if header[j] else f"col_{j}"
            val = _norm_cell(row[j])
            if not val or val == ".":
                continue
            facts.append(f"{entity} | {col} = {val}")
    return facts

def build_context_facts(ex: Dict[str, Any]) -> List[str]:
    pre_lines = normalize_text_lines(ex.get("pre_text"))
    post_lines = normalize_text_lines(ex.get("post_text"))
    tbl_facts = table_to_facts(ex.get("table"))
    facts: List[str] = []
    # keep table facts first (most structured)
    facts.extend(tbl_facts)
    # add text lines as "context | sentence = ..."
    for s in pre_lines:
        facts.append(f"context | pre_text = {s}")
    for s in post_lines:
        facts.append(f"context | post_text = {s}")
    # dedupe while preserving order
    seen = set()
    out = []
    for f in facts:
        k = f.strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out

def build_prompt_from_facts(facts: List[str], question: str) -> str:
    facts_text = "\n".join(f"- {x}" for x in facts) if facts else "(none)"
    normalization_rules = (
        "\nNORMALIZATION RULES:\n"
        "- Do NOT rescale numbers (e.g., do not turn 637 into 637,000,000).\n"
        "- Do NOT convert units unless the question explicitly asks for conversion.\n"
        "- If numerator and denominator share the same unit, the unit cancels: use raw numbers.\n"
        "- If the question asks for a percentage, return a percentage (with %).\n"
    )
    return (
        f"FACTS:\n{facts_text}\n"
        f"{normalization_rules}\n"
        f"QUESTION:\n{question}\n\n"
        f"FINAL ANSWER:"
    )

# -------------------------
# Ollama helper
# -------------------------
def ask_ollama(prompt: str, system: str, model: str, timeout_s: int = 240, retries: int = 1) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 256},
                },
                timeout=timeout_s,
            )
            r.raise_for_status()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            text = r.json()["message"]["content"].strip()
            return {"text": text, "latency_ms": dt_ms, "prompt_chars": len(prompt), "output_chars": len(text)}
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            timeout_s = int(timeout_s * 1.5)
            time.sleep(0.2 + 0.3 * attempt)
            continue
        except Exception as e:
            last_err = e
            break
    return {"text": "UNKNOWN", "latency_ms": None, "prompt_chars": len(prompt), "output_chars": 0, "error": str(last_err) if last_err else "unknown_error"}

# -------------------------
# Mem0 helpers
# -------------------------
def reset_store(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[reset] Deleted store at: {path}", flush=True)

def make_mem0(store_path: str, embed_model: str, llm_model: str) -> Memory:
    return Memory.from_config({
        "vector_store": {"provider": "chroma", "config": {"path": store_path}},
        "embedder": {"provider": "ollama", "config": {"model": embed_model, "ollama_base_url": OLLAMA_BASE_URL}},
        "llm": {"provider": "ollama", "config": {"model": llm_model, "ollama_base_url": OLLAMA_BASE_URL, "temperature": 0}},
    })

# -------------------------
# Evaluation (FinQA v2-ish)
# -------------------------
_NUM_TOKEN_RE = re.compile(
    r"""
    (?P<paren>\(\s*)?                  # optional opening parenthesis for accounting negative
    (?P<sign>[-+])?                    # optional explicit sign
    (?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)   # number
    (?P<paren_end>\s*\))?              # optional closing parenthesis
    (?P<pct>\s*%)?                     # optional percent sign
    """,
    re.VERBOSE,
)

def extract_numbers_with_flags(text: Any) -> List[Tuple[float, bool]]:
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

def extract_last_number_with_flags(text: Any) -> Optional[Tuple[float, bool]]:
    nums = extract_numbers_with_flags(text)
    if not nums:
        return None
    return nums[-1]

def decimals_in_gold(gold: Any) -> int:
    if gold is None:
        return 0
    s = str(gold).replace(",", "")
    m = re.search(r"\d+(?:\.(\d+))?", s)
    return len(m.group(1)) if m and m.group(1) else 0

def question_wants_percent(question: str) -> bool:
    q = (question or "").lower()
    return ("percent" in q) or ("percentage" in q) or ("%" in q)

def normalize_pred_to_gold_scale(question: str, gold: Any, pred: Any) -> Optional[Tuple[float, float, bool]]:
    """Return (pred_value, gold_value, want_pct) after best-candidate selection + percent normalization."""
    g_last = extract_last_number_with_flags(gold)
    if g_last is None:
        return None
    gval, g_has_pct = g_last

    want_pct = bool(g_has_pct) or question_wants_percent(question) or ("%" in str(gold or "")) or ("%" in str(pred or ""))

    p_nums = extract_numbers_with_flags(pred)
    if not p_nums:
        return None

    best_p = None
    best_err = None
    for pv, pv_has_pct in p_nums:
        pvv = pv
        # If expecting percent but pred is likely a fraction, normalize it (0.12 -> 12)
        if want_pct and (not pv_has_pct) and 0 <= abs(pvv) <= 1.5:
            pvv *= 100.0
        err = abs(pvv - gval)
        if best_err is None or err < best_err:
            best_err = err
            best_p = pvv

    if best_p is None:
        return None
    return (best_p, gval, want_pct)

def exact_match(pred: Any, gold: Any, question: str) -> bool:
    norm = normalize_pred_to_gold_scale(question, gold, pred)
    if norm is None:
        return False
    pval, gval, _want_pct = norm
    decs = decimals_in_gold(gold)
    if decs == 0:
        return int(round(pval)) == int(round(gval))
    return round(pval, decs) == round(gval, decs)

def numeric_close(pred: Any, gold: Any, question: str) -> bool:
    norm = normalize_pred_to_gold_scale(question, gold, pred)
    if norm is None:
        return False
    pval, gval, want_pct = norm
    decs = decimals_in_gold(gold)

    if want_pct:
        abs_tol = 0.5 if decs == 0 else (0.15 if decs == 1 else 0.05)
        return math.isclose(pval, gval, abs_tol=abs_tol, rel_tol=0.002)

    if decs == 0:
        tol = 0.5 if abs(gval) < 1000 else max(1.0, abs(gval) * 0.001)
        return abs(pval - gval) <= tol
    if decs == 1:
        return abs(pval - gval) <= 0.15
    return abs(pval - gval) <= 0.05

def parse_success(pred: Any) -> bool:
    return extract_last_number_with_flags(pred) is not None

def has_multiple_numbers(pred: Any) -> bool:
    return len(extract_numbers_with_flags(pred)) >= 2

def format_violation(pred: Any) -> bool:
    t = (str(pred or "")).strip()
    if not t:
        return True
    if "\n" in t:
        return True
    bad_markers = ["because", "therefore", "so ", "we ", "let's", "calculation", "="]
    if any(m in t.lower() for m in bad_markers) and has_multiple_numbers(t):
        return True
    return False

def numeric_error(pred: Any, gold: Any, question: str) -> Optional[Dict[str, float]]:
    norm = normalize_pred_to_gold_scale(question, gold, pred)
    if norm is None:
        return None
    pval, gval, _want_pct = norm
    abs_err = abs(pval - gval)
    rel_err = abs_err / (abs(gval) + 1e-9)
    return {"abs_err": abs_err, "rel_err": rel_err}

def judge_correct(question: str, gold: Any, pred: Any, model: str) -> Optional[bool]:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"GOLD:\n{gold}\n\n"
        f"PREDICTED:\n{pred}\n\n"
        "Label:"
        "\n(Reply with exactly one token: CORRECT or WRONG)"
    )
    resp = ask_ollama(prompt, system=JUDGE_SYSTEM, model=model)
    txt = (resp.get("text") or "").strip().upper()
    if txt.startswith("CORRECT"):
        return True
    if txt.startswith("WRONG"):
        return False
    return None

def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)

def _short(s: str, n: int = 60) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"

# -------------------------
# Runner
# -------------------------
FINQA_FIELDNAMES = [
    "idx","method","question","gold_answer","pred_answer",
    "exact_match","numeric_close",
    "parse_success","format_violation","has_multiple_numbers",
    "abs_err","rel_err","judge_correct",
    "latency_ms","search_ms","prompt_chars","output_chars",
    "n_facts_stored","n_facts_retrieved","memory_items","memory_chars",
]

def run(
    split: str,
    data_dir: str,
    max_samples: Optional[int],
    max_turns: Optional[int],
    history_turns: Optional[int],
    out_csv: str,
    reset: bool,
    k_retrieve: int,
    judge: bool,
    print_every: int,
    store_path: str,
    llm_model: str,
    embed_model: str,
) -> None:
    dataset_path = resolve_dataset_path(split, data_dir)
    print(f"[INFO] Loading split={split} from: {dataset_path}", flush=True)
    with open(dataset_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    if not isinstance(conversations, list):
        raise ValueError("Expected dataset JSON root to be a list of conversations")

    if max_samples is not None:
        conversations = conversations[:max_samples]

    if reset:
        reset_store(store_path)

    mem = make_mem0(store_path, embed_model=embed_model, llm_model=llm_model)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    latencies_ms: List[float] = []
    search_latencies_ms: List[float] = []
    start_wall = time.time()
    global_idx = -1

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FINQA_FIELDNAMES)
        w.writeheader()

        for d_i, ex in enumerate(conversations):
            dialog_id = str(ex.get("id", f"dialog_{d_i}"))
            user_id = f"convfinqa_struct::{dialog_id}"  # dialogue-scoped user id

            turns = get_turns_from_example(ex)
            golds = get_gold_answers_from_example(ex)

            if max_turns is not None:
                turns = turns[:max_turns]

            # build & store context facts once per dialogue
            context_facts = build_context_facts(ex)
            # Store context facts in a single add-only call (avoids Mem0 UPDATE/DELETE noise)
            if context_facts:
                messages = [{"role": "user", "content": cf} for cf in context_facts]
                mem.add(
                    messages,
                    user_id=user_id,
                    metadata={"type": "context_facts", "dialog_id": dialog_id},
                    infer=False,
                )
            history: List[Tuple[str, str]] = []

            for t_i, question in enumerate(turns):
                global_idx += 1

                if history_turns is not None and history_turns >= 0:
                    history_for_prompt = history[-history_turns:] if history_turns else []
                else:
                    history_for_prompt = history

                gold = golds[t_i] if (golds and t_i < len(golds)) else None

                # retrieve facts
                t_search0 = time.perf_counter()
                retrieved = mem.search(question, user_id=user_id)
                search_ms = (time.perf_counter() - t_search0) * 1000.0
                search_latencies_ms.append(search_ms)

                remembered: List[str] = []
                if isinstance(retrieved, dict) and "results" in retrieved:
                    for r in retrieved["results"]:
                        if isinstance(r, dict) and r.get("memory"):
                            remembered.append(str(r["memory"]).strip())
                # de-dupe and top-k
                seen = set()
                dedup = []
                for x in remembered:
                    if not x or x in seen:
                        continue
                    seen.add(x)
                    dedup.append(x)
                remembered = dedup[:max(0, k_retrieve)]

                memory_items = len(remembered)
                memory_chars = len("\n".join(remembered)) if remembered else 0

                prompt = build_prompt_from_facts(remembered, question)

                # call LLM
                resp = ask_ollama(prompt, system=FIN_SYSTEM, model=llm_model)
                pred = resp.get("text", "")

                # store QA turn as a memory (so later turns can retrieve)
                                # Store QA turn as add-only memory (no consolidation)
                qa_text = f"qa_turn | q = {question} | a = {pred}"
                mem.add(
                    [{"role": "user", "content": qa_text}],
                    user_id=user_id,
                    metadata={"type": "qa_turn", "dialog_id": dialog_id, "turn_idx": t_i},
                    infer=False,
                )
# evaluation
                err = numeric_error(pred, gold, question) if gold is not None else None

                row = {
                    "idx": global_idx,
                    "method": "structured_mem0",
                    "question": question,
                    "gold_answer": gold,
                    "pred_answer": pred,
                    "exact_match": exact_match(pred, gold, question) if gold is not None else False,
                    "numeric_close": numeric_close(pred, gold, question) if gold is not None else False,
                    "parse_success": parse_success(pred),
                    "format_violation": format_violation(pred),
                    "has_multiple_numbers": has_multiple_numbers(pred),
                    "abs_err": (round(err["abs_err"], 6) if err else None),
                    "rel_err": (round(err["rel_err"], 6) if err else None),
                    "judge_correct": (judge_correct(question, gold, pred, llm_model) if (judge and gold is not None) else None),
                    "latency_ms": (round(resp["latency_ms"], 1) if resp.get("latency_ms") is not None else None),
                    "search_ms": round(search_ms, 1),
                    "prompt_chars": resp.get("prompt_chars"),
                    "output_chars": resp.get("output_chars"),
                    "n_facts_stored": (len(context_facts) + 1) if t_i == 0 else 1,  # context stored once + QA
                    "n_facts_retrieved": len(remembered),
                    "memory_items": memory_items,
                    "memory_chars": memory_chars,
                }

                w.writerow(row)

                if resp.get("latency_ms") is not None:
                    latencies_ms.append(resp["latency_ms"])

                # progress print (includes dialog+turn, but does NOT add extra CSV cols)
                print(
                    f"[{global_idx}] struct_mem0 dialog={dialog_id} turn={t_i} "
                    f"exact={row['exact_match']} tol={row['numeric_close']} "
                    f"lat={row['latency_ms']}ms search={row['search_ms']}ms "
                    f"ret={row['n_facts_retrieved']} pred={_short(pred)} gold={gold}",
                    flush=True,
                )

                if print_every and ((global_idx + 1) % print_every == 0):
                    elapsed = time.time() - start_wall
                    p50 = percentile([x for x in latencies_ms if x is not None], 50)
                    p95 = percentile([x for x in latencies_ms if x is not None], 95)
                    sp50 = percentile([x for x in search_latencies_ms if x is not None], 50)
                    sp95 = percentile([x for x in search_latencies_ms if x is not None], 95)
                    print(
                        f"[PROGRESS] turns={global_idx+1} elapsed={elapsed/60:.1f}m "
                        f"latency_p50={p50:.1f}ms latency_p95={p95:.1f}ms "
                        f"search_p50={sp50:.1f}ms search_p95={sp95:.1f}ms",
                        flush=True,
                    )

                history.append((question, pred))

    print(f"\nSaved results to: {out_csv}", flush=True)
    p50 = percentile([x for x in latencies_ms if x is not None], 50)
    p95 = percentile([x for x in latencies_ms if x is not None], 95)
    if p50 is not None and p95 is not None:
        print(f"Latency p50={p50:.1f}ms p95={p95:.1f}ms", flush=True)
    if search_latencies_ms:
        sp50 = percentile([x for x in search_latencies_ms if x is not None], 50)
        sp95 = percentile([x for x in search_latencies_ms if x is not None], 95)
        if sp50 is not None and sp95 is not None:
            print(f"Search latency p50={sp50:.1f}ms p95={sp95:.1f}ms", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["train", "dev", "validation", "test"])
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--max-samples", type=int, default=20, help="Limit number of dialogues")
    p.add_argument("--max-turns", type=int, default=None, help="Limit turns per dialogue")
    p.add_argument("--history-turns", type=int, default=None, help="Unused for structured prompt; kept for parity")
    p.add_argument("--out", default="results/convfinqa_structured_mem0.csv")
    p.add_argument("--reset-store", action="store_true", help="Reset Mem0 store path before run")
    p.add_argument("--k", type=int, default=12, help="Top-K retrieved facts to include in prompt")
    p.add_argument("--judge", action="store_true", help="Enable LLM-as-judge evaluation")
    p.add_argument("--print-every", type=int, default=25, help="Print summary every N turns")
    p.add_argument("--store-path", default=DEFAULT_STORE_PATH)
    p.add_argument("--model", default=DEFAULT_LLM_MODEL, help="Ollama model name")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model name")
    args = p.parse_args()

    run(
        split=args.split,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        history_turns=args.history_turns,
        out_csv=args.out,
        reset=args.reset_store,
        k_retrieve=args.k,
        judge=args.judge,
        print_every=args.print_every,
        store_path=args.store_path,
        llm_model=args.model,
        embed_model=args.embed_model,
    )

if __name__ == "__main__":
    main()
