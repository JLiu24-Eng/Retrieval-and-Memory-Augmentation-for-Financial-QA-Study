#!/usr/bin/env python3
"""
FinQA Baseline and Mem0-Augmented Runner

Run with --mem0 to switch from Baseline LLM to Mem0-Augmented mode.

Evaluates two architectures on the FinQA benchmark:
  - Baseline LLM (--mem0 off): plain LLM with full document context.
  - Mem0-Augmented (--mem0): context and Q/A history stored in Mem0;
    relevant memories retrieved and injected per question.

Output CSV columns:
  idx, method, question, gold_answer, pred_answer,
  exact_match, numeric_close,
  parse_success, format_violation, has_multiple_numbers,
  abs_err, rel_err, judge_correct,
  latency_ms, search_ms, prompt_chars, output_chars,
  memory_items, memory_chars

CLI:
  python run_finqa_baseline_mem0.py --split validation --limit 492 --out results/finqa_baseline.csv
  python run_finqa_baseline_mem0.py --split validation --limit 492 --mem0 --out results/finqa_mem0_aug.csv --judge
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
from typing import Any, Dict, Optional, Tuple

import requests
from mem0 import Memory

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.1:latest"
EMBED_MODEL = "nomic-embed-text"

STORE_PATH = "./mem0_store"

FIN_SYSTEM = (
    "You are a financial reasoning assistant. "
    "Answer the question using ONLY the given context. "
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


def reset_store(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[reset] Deleted store at: {path}")


def ask_ollama(prompt: str, system: str = FIN_SYSTEM) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": LLM_MODEL,
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


def build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nFinal Answer:"


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


def extract_last_number_with_flags(text: str) -> Optional[Tuple[float, bool]]:
    """
    Returns (value, has_percent_sign) from the last number-like token.
    Handles accounting negatives: (35) -> -35
    """
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None

    matches = list(_NUM_TOKEN_RE.finditer(t))
    if not matches:
        return None

    m = matches[-1]
    num_str = m.group("num").replace(",", "")
    try:
        val = float(num_str)
    except ValueError:
        return None

    # Accounting parentheses mean negative if no explicit sign
    has_parens = (m.group("paren") is not None) and (m.group("paren_end") is not None)
    explicit_sign = m.group("sign")
    if has_parens and explicit_sign != "-":
        val = -val
    elif explicit_sign == "-":
        val = -abs(val)

    has_pct = m.group("pct") is not None
    return val, has_pct


def percent_expected(question: str, gold: str, pred: str) -> bool:
    q = (question or "").lower()
    g = str(gold or "")
    p = str(pred or "")
    return (
        "%" in g
        or "%" in p
        or "percent" in q
        or "percentage" in q
    )


def decimals_in_gold(gold: str) -> int:
    s = str(gold or "").replace(",", "")
    m = re.search(r"\d+(?:\.(\d+))?", s)
    return len(m.group(1)) if m and m.group(1) else 0


def normalize_pred_to_gold_scale(question: str, gold: str, pred: str) -> Optional[Tuple[float, float, bool]]:
    """
    Returns (pred_val, gold_val, want_percent) after applying
    fraction->percent conversion when appropriate.
    """
    g = extract_last_number_with_flags(gold)
    p = extract_last_number_with_flags(pred)
    if g is None or p is None:
        return None

    gval, g_pct = g
    pval, p_pct = p

    want_pct = percent_expected(question, gold, pred) or g_pct

    # If percent expected but prediction is a fraction (0.xx), convert to percent
    if want_pct and (not p_pct) and 0 <= abs(pval) <= 1.5:
        pval *= 100.0

    return pval, gval, want_pct


def exact_match(pred: str, gold: str, question: str = "") -> bool:
    """
    "Exact" in a numeric sense:
    - same value after rounding to gold precision
    - supports percent implied by question
    """
    norm = normalize_pred_to_gold_scale(question, gold, pred)
    if norm is None:
        return False

    pval, gval, _want_pct = norm
    decs = decimals_in_gold(gold)

    if decs == 0:
        return int(round(pval)) == int(round(gval))
    return round(pval, decs) == round(gval, decs)


def numeric_close(pred: str, gold: str, question: str = "", atol: float = 1e-2, rtol: float = 1e-2) -> bool:
    """
    "Close" match with human-reasonable tolerances.
    Handles:
    - percent vs fraction (0.xx) normalization
    - accounting negatives (35) -> -35
    - tolerances based on gold precision
    """
    norm = normalize_pred_to_gold_scale(question, gold, pred)
    if norm is None:
        return False

    pval, gval, want_pct = norm
    decs = decimals_in_gold(gold)

    # More forgiving tolerances for % and for integer answers
    if want_pct:
        # percent points
        if decs == 0:
            abs_tol = 0.5
        elif decs == 1:
            abs_tol = 0.15
        else:
            abs_tol = 0.05
        return math.isclose(pval, gval, abs_tol=abs_tol, rel_tol=0.002)

    # non-percent
    if decs == 0:
        abs_tol = 0.5 if abs(gval) < 1000 else max(1.0, abs(gval) * 0.001)
        return abs(pval - gval) <= abs_tol
    elif decs == 1:
        return abs(pval - gval) <= 0.15
    else:
        return abs(pval - gval) <= 0.05


def parse_success(text: str) -> bool:
    return extract_last_number_with_flags(text) is not None


def has_multiple_numbers(text: str) -> bool:
    if not text:
        return False
    return len(list(_NUM_TOKEN_RE.finditer(str(text)))) >= 2


def format_violation(text: str) -> bool:
    """
    Heuristic: output is expected to be "ONLY the final answer".
    Flags multi-line or obvious reasoning-y outputs.
    """
    t = (text or "").strip()
    if not t:
        return True
    if "\n" in t:
        return True
    bad_markers = ["because", "therefore", "so ", "we ", "let's", "calculation", "="]
    if any(m in t.lower() for m in bad_markers) and has_multiple_numbers(t):
        return True
    return False


def numeric_error(pred: str, gold: str, question: str = "") -> Optional[Dict[str, float]]:
    """
    Continuous error metrics:
      - abs_err
      - rel_err (|pred-gold| / (|gold|+eps))
    """
    norm = normalize_pred_to_gold_scale(question, gold, pred)
    if norm is None:
        return None
    pval, gval, _want_pct = norm
    abs_err = abs(pval - gval)
    rel_err = abs_err / (abs(gval) + 1e-9)
    return {"abs_err": abs_err, "rel_err": rel_err}


def judge_correct(question: str, gold: str, pred: str) -> Optional[bool]:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"GOLD:\n{gold}\n\n"
        f"PREDICTED:\n{pred}\n\n"
        "LABEL:"
    )
    j = ask_ollama(prompt, system=JUDGE_SYSTEM)["text"].strip().upper()
    if "CORRECT" in j:
        return True
    if "WRONG" in j:
        return False
    return None


def percentile(values, p: float) -> Optional[float]:
    """
    Simple percentile (0-100). Linear interpolation between closest ranks.
    """
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


def make_mem0() -> Memory:
    return Memory.from_config({
        "vector_store": {"provider": "chroma", "config": {"path": STORE_PATH}},
        "embedder": {
            "provider": "ollama",
            "config": {"model": EMBED_MODEL, "ollama_base_url": "http://localhost:11434"}
        },
        "llm": {
            "provider": "ollama",
            "config": {"model": LLM_MODEL, "ollama_base_url": "http://localhost:11434", "temperature": 0}
        }
    })


def mem0_augmented_prompt(memory_results: Dict[str, Any], context: str, question: str) -> Tuple[str, Dict[str, Any]]:
    """
    Convert Mem0 search results to text and prepend them as “Remembered facts”.
    Returns: (prompt, stats)
    """
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


def load_finqa_local(split: str):
    split_map = {
        "train": "train.json",
        "validation": "dev.json",
        "test": "test.json",
    }
    path = os.path.join("data", "FinQA", "dataset", split_map[split])
    if not os.path.exists(path):
        raise FileNotFoundError(f"FinQA file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # FinQA JSON is usually a dict with key "data" but sometimes it is directly a list
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def finqa_normalize(ex: dict) -> Tuple[str, str, str]:
    """
    FinQA JSON schema:
      - pre_text: list[str]
      - post_text: list[str]
      - table: list[list[str]]
      - qa: {question: str, answer: str, ...}
    Returns: (context, question, answer)
    """
    qa = ex.get("qa", {}) or {}
    question = (qa.get("question") or "").strip()
    answer = str(qa.get("answer") or "").strip()

    pre = ex.get("pre_text") or []
    post = ex.get("post_text") or []
    table = ex.get("table") or []

    pre_text = "\n".join([s.strip() for s in pre if isinstance(s, str) and s.strip()])
    post_text = "\n".join([s.strip() for s in post if isinstance(s, str) and s.strip()])

    table_str = ""
    if isinstance(table, list) and table:
        if isinstance(table[0], list):
            table_str = "\n".join(["\t".join([str(c) for c in row]) for row in table])
        else:
            table_str = str(table)

    context_parts = []
    if pre_text:
        context_parts.append("Pre-text:\n" + pre_text)
    if table_str:
        context_parts.append("Table:\n" + table_str)
    if post_text:
        context_parts.append("Post-text:\n" + post_text)

    context = "\n\n".join(context_parts).strip()
    return context, question, answer


def run(split: str, limit: int, out_csv: str, use_mem0: bool, reset: bool, judge: bool) -> None:
    ds = load_finqa_local(split)

    if reset and use_mem0:
        reset_store(STORE_PATH)

    mem = make_mem0() if use_mem0 else None
    user_id = "finqa_user_1"

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = [
        "idx", "method", "question", "gold_answer", "pred_answer",
        "exact_match", "numeric_close",
        "parse_success", "format_violation", "has_multiple_numbers",
        "abs_err", "rel_err",
        "judge_correct",
        "latency_ms", "search_ms",
        "prompt_chars", "output_chars",
        "memory_items", "memory_chars",
    ]

    latencies_ms = []
    search_latencies_ms = []

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i in range(min(limit, len(ds))):
            ex = ds[i]
            context, question, gold = finqa_normalize(ex)

            if not question or not context or not gold:
                print(f"[{i}] Skipping: missing fields question/context/answer")
                continue

            search_ms: Optional[float] = None
            mem_stats: Dict[str, Any] = {"memory_items": None, "memory_chars": None}

            if not use_mem0:
                method = "llm_baseline"
                prompt = build_prompt(context, question)
                resp = ask_ollama(prompt)
                pred = resp["text"]
            else:
                method = "mem0_augmented"

                # Add context once as “memory” for this user (simple baseline).
                mem.add(context, user_id=user_id, metadata={"type": "context", "idx": i})

                # Retrieve relevant memories for this question (measure separately)
                t_search0 = time.perf_counter()
                retrieved = mem.search(question, user_id=user_id)
                search_ms = (time.perf_counter() - t_search0) * 1000.0
                search_latencies_ms.append(search_ms)

                prompt, mem_stats = mem0_augmented_prompt(retrieved, context, question)
                resp = ask_ollama(prompt)
                pred = resp["text"]

                # Optionally add QA turn as memory
                mem.add(f"Q: {question}\nA: {pred}", user_id=user_id, metadata={"type": "qa", "idx": i})

            # Metrics
            ps = parse_success(pred)
            fv = format_violation(pred)
            multi = has_multiple_numbers(pred)
            err = numeric_error(pred, gold, question)
            j_ok = judge_correct(question, gold, pred) if judge else None

            row = {
                "idx": i,
                "method": method,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,
                "exact_match": exact_match(pred, gold, question),
                "numeric_close": numeric_close(pred, gold, question),
                "parse_success": ps,
                "format_violation": fv,
                "has_multiple_numbers": multi,
                "abs_err": (round(err["abs_err"], 6) if err else None),
                "rel_err": (round(err["rel_err"], 6) if err else None),
                "judge_correct": j_ok,
                "latency_ms": round(resp["latency_ms"], 1),
                "search_ms": (round(search_ms, 1) if search_ms is not None else None),
                "prompt_chars": resp["prompt_chars"],
                "output_chars": resp["output_chars"],
                "memory_items": mem_stats.get("memory_items"),
                "memory_chars": mem_stats.get("memory_chars"),
            }
            w.writerow(row)

            latencies_ms.append(resp["latency_ms"])

            print(
                f"[{i}] {method} exact={row['exact_match']} tol={row['numeric_close']} "
                f"parse={ps} judge={j_ok} latency_ms={row['latency_ms']} "
                f"search_ms={row['search_ms']} pred={pred} gold={gold}"
            )

    # Summary
    p50 = percentile(latencies_ms, 50)
    p95 = percentile(latencies_ms, 95)
    print(f"\nSaved results to: {out_csv}")
    if p50 is not None and p95 is not None:
        print(f"Latency p50={p50:.1f}ms p95={p95:.1f}ms")

    if use_mem0 and search_latencies_ms:
        sp50 = percentile(search_latencies_ms, 50)
        sp95 = percentile(search_latencies_ms, 95)
        if sp50 is not None and sp95 is not None:
            print(f"Mem0 search latency p50={sp50:.1f}ms p95={sp95:.1f}ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", default="results/finqa_baseline.csv")
    p.add_argument("--mem0", action="store_true", help="Run Mem0-augmented variant")
    p.add_argument("--reset-store", action="store_true", help="Reset Mem0 store (only relevant with --mem0)")
    p.add_argument("--judge", action="store_true", help="Run LLM-as-a-judge evaluation (extra calls)")

    args = p.parse_args()

    run(
        split=args.split,
        limit=args.limit,
        out_csv=args.out,
        use_mem0=args.mem0,
        reset=args.reset_store,
        judge=args.judge,
    )
