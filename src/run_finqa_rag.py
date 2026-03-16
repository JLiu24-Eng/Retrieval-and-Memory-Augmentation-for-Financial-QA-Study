#!/usr/bin/env python3
"""
FinQA RAG Runner

Retrieval-Augmented Generation for the FinQA benchmark.
No persistent memory — per-question in-context retrieval only.
  1. Document evidence (pre_text / table / post_text) decomposed into granular facts.
  2. Facts and query embedded via Ollama (nomic-embed-text); cosine top-k selected.
  3. Retrieved facts passed to the LLM as the sole context.

Design
- NO Mem0. No persistent memory lifecycle.
- Per-example in-context retrieval:
    1) Convert FinQA evidence (pre_text/table/post_text) into "facts" (short strings).
    2) Embed facts and the question via Ollama /api/embeddings.
    3) Retrieve top-k facts by cosine similarity.
    4) Ask the chat model using ONLY retrieved facts + question.

CSV schema
Matches run_finqa_baseline.py exactly:
  idx, method, question, gold_answer, pred_answer,
  exact_match, numeric_close,
  parse_success, format_violation, has_multiple_numbers,
  abs_err, rel_err,
  judge_correct,
  latency_ms, search_ms,
  prompt_chars, output_chars,
  memory_items, memory_chars
"""
# Architecture Matters More Than Scale:
# A Comparative Study of Retrieval and Memory Augmentation for Financial QA
# Under SME Compute Constraints
#
# GitHub: https://github.com/JLiu24-Eng/Retrieval-and-Memory-Augmentation-for-Financial-QA-Study
# Authors: Jianan Liu, Jing Yang, Xianyou Li, Weiran Yan, Yichao Wu, Penghao Liang, Mengwei Yuan

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# -------------------------
# Defaults / endpoints
# -------------------------
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")

DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:latest")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

DEFAULT_DATA_DIR = os.environ.get("FINQA_DATA_DIR", "data/FinQA/dataset")
DEFAULT_STORE_ROOT = os.environ.get("RAG_STORE_ROOT", "./rag_embed_cache_finqa")

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

FIELDNAMES = [
    "idx", "method", "question", "gold_answer", "pred_answer",
    "exact_match", "numeric_close",
    "parse_success", "format_violation", "has_multiple_numbers",
    "abs_err", "rel_err",
    "judge_correct",
    "latency_ms", "search_ms",
    "prompt_chars", "output_chars",
    "memory_items", "memory_chars",
]

# -------------------------
# I/O helpers
# -------------------------
def reset_store_root(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[reset] Deleted embedding cache at: {path}")

def percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    d0 = ys[f] * (c - k)
    d1 = ys[c] * (k - f)
    return float(d0 + d1)

# -------------------------
# Ollama calls
# -------------------------
def _post_json_with_retry(
    url: str,
    payload: Dict[str, Any],
    timeout_s: int = 240,
    max_retries: int = 3,
    backoff_base_s: float = 2.0,
) -> Tuple[requests.Response, float]:
    """
    POST json to `url` with simple retry/backoff.
    Retries on:
      - timeouts / connection errors
      - transient HTTP errors (429/5xx)
    Returns (response, latency_ms) for the *successful* attempt.
    """
    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            # Raise for HTTP errors (4xx/5xx)
            r.raise_for_status()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return r, dt_ms
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError) as e:
            last_exc = e
        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = getattr(e.response, "status_code", None)
            # Retry only for transient statuses
            if status is None or status not in (429, 500, 502, 503, 504):
                raise

        # Retry path
        if attempt < max_retries:
            sleep_s = min(backoff_base_s * (2 ** (attempt - 1)), 30.0)
            # small deterministic jitter
            sleep_s += 0.1 * attempt
            time.sleep(sleep_s)
        else:
            assert last_exc is not None
            raise last_exc

def ask_ollama(prompt: str, system: str = FIN_SYSTEM, model: str = DEFAULT_LLM_MODEL) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    r, dt_ms = _post_json_with_retry(OLLAMA_CHAT_URL, payload, timeout_s=240, max_retries=3)
    text = r.json()["message"]["content"].strip()
    return {
        "text": text,
        "latency_ms": dt_ms,
        # Char-based proxies (portable across local models)
        "prompt_chars": len(prompt),
        "output_chars": len(text),
    }

def embed_text(text: str, model: str = DEFAULT_EMBED_MODEL) -> List[float]:
    payload = {"model": model, "prompt": text}
    r, _ = _post_json_with_retry(OLLAMA_EMBED_URL, payload, timeout_s=240, max_retries=3)
    out = r.json()
    emb = out.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise ValueError("No embedding returned from Ollama.")
    return [float(x) for x in emb]

def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def embed_cached(text: str, store_root: str, model: str) -> List[float]:
    os.makedirs(store_root, exist_ok=True)
    key = _hash_text(model + "||" + text)
    fp = Path(store_root) / f"{key}.json"
    if fp.exists():
        try:
            obj = json.loads(fp.read_text())
            if isinstance(obj, dict) and isinstance(obj.get("embedding"), list):
                return [float(x) for x in obj["embedding"]]
        except Exception:
            pass
    emb = embed_text(text, model=model)
    fp.write_text(json.dumps({"embedding": emb}))
    return emb

# -------------------------
# FinQA dataset loading
# -------------------------
def load_finqa_local(split: str, data_dir: str) -> List[Dict[str, Any]]:
    s = split.lower().strip()
    if s in ("validation", "val", "dev"):
        fn = "dev.json"
    elif s == "train":
        fn = "train.json"
    elif s == "test":
        fn = "test.json"
    else:
        raise ValueError("split must be one of: train, validation, test")
    p = Path(data_dir) / fn
    if not p.exists():
        raise FileNotFoundError(f"FinQA file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def finqa_normalize(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    question = ex.get("qa", {}).get("question") or ex.get("question") or ""
    answer = ex.get("qa", {}).get("answer") or ex.get("answer") or ""

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
    return context, question, str(answer)

# -------------------------
# Facts: text + table rows
# -------------------------
def normalize_text_lines(x: Any, max_lines: int = 120) -> List[str]:
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

def table_to_facts(table: Any, max_rows: int = 60) -> List[str]:
    if not table or not isinstance(table, list) or len(table) < 2:
        return []
    if not isinstance(table[0], list):
        # Already string-ish rows
        facts = []
        for r in table[:max_rows]:
            s = str(r).strip()
            if s:
                facts.append(s)
        return facts

    header = [str(c).strip() for c in table[0]]
    facts: List[str] = []
    for row in table[1:1 + max_rows]:
        if not isinstance(row, list):
            continue
        row_cells = [str(c).strip() for c in row]
        # Compact "col: val" format
        pairs = []
        for h, v in zip(header, row_cells):
            if h and v:
                pairs.append(f"{h}: {v}")
        if pairs:
            facts.append(" | ".join(pairs))
        else:
            joined = "\t".join(row_cells).strip()
            if joined:
                facts.append(joined)
    return facts

def build_facts(ex: Dict[str, Any], max_pre_lines: int = 60, max_post_lines: int = 60) -> List[str]:
    pre = normalize_text_lines(ex.get("pre_text"), max_lines=max_pre_lines)
    post = normalize_text_lines(ex.get("post_text"), max_lines=max_post_lines)
    table = ex.get("table")
    tfacts = table_to_facts(table, max_rows=60)

    facts: List[str] = []
    for s in pre:
        facts.append(f"PRE: {s}")
    for s in tfacts:
        facts.append(f"TABLE: {s}")
    for s in post:
        facts.append(f"POST: {s}")

    # Dedup while preserving order
    seen = set()
    out = []
    for f in facts:
        k = f.strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out

def cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def retrieve_topk_facts(facts: List[str], question: str, k: int, store_root: str, embed_model: str) -> Tuple[List[str], float]:
    t0 = time.perf_counter()
    if not facts:
        return [], 0.0
    qemb = embed_cached(question, store_root=store_root, model=embed_model)
    # Embed facts (cached)
    scored: List[Tuple[float, str]] = []
    for f in facts:
        femb = embed_cached(f, store_root=store_root, model=embed_model)
        scored.append((cosine(qemb, femb), f))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [f for _, f in scored[: max(1, k)]]
    ms = (time.perf_counter() - t0) * 1000.0
    return top, ms

def build_rag_prompt(retrieved_facts: List[str], question: str) -> str:
    facts_block = "\n".join([f"- {f}" for f in retrieved_facts])
    return (
        "FACTS:\n"
        f"{facts_block}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "Final Answer:"
    )

# -------------------------
# Evaluation helpers (copied from run_finqa_baseline.py)
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

def extract_last_number_with_flags(text: str) -> Optional[Tuple[float, bool]]:
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
    return ("%" in g) or ("%" in p) or ("percent" in q) or ("percentage" in q)

def decimals_in_gold(gold: str) -> int:
    s = str(gold or "").replace(",", "")
    m = re.search(r"\d+(?:\.(\d+))?", s)
    return len(m.group(1)) if m and m.group(1) else 0

def normalize_pred_to_gold_scale(question: str, gold: str, pred: str) -> Optional[Tuple[float, float, bool]]:
    g = extract_last_number_with_flags(gold)
    p = extract_last_number_with_flags(pred)
    if g is None or p is None:
        return None
    gval, g_pct = g
    pval, p_pct = p
    want_pct = percent_expected(question, gold, pred)
    if want_pct and (not p_pct) and (abs(pval) <= 1.5):
        pval = pval * 100.0
    return pval, gval, want_pct

def exact_match(pred: str, gold: str, question: str) -> bool:
    n = normalize_pred_to_gold_scale(question, gold, pred)
    if n is None:
        return False
    pval, gval, _ = n
    return abs(pval - gval) < 1e-9

def numeric_close(pred: str, gold: str, question: str) -> bool:
    n = normalize_pred_to_gold_scale(question, gold, pred)
    if n is None:
        return False
    pval, gval, _ = n
    d = decimals_in_gold(gold)
    if d >= 3:
        tol = 10 ** (-d)
    else:
        tol = 0.5 if abs(gval) < 1000 else max(1.0, abs(gval) * 0.001)
    return abs(pval - gval) <= tol

def parse_success(pred: str) -> bool:
    return extract_last_number_with_flags(pred) is not None

def has_multiple_numbers(pred: str) -> bool:
    if pred is None:
        return False
    return len(list(_NUM_TOKEN_RE.finditer(str(pred)))) >= 2

def format_violation(pred: str) -> bool:
    if pred is None:
        return True
    t = str(pred).strip()
    if not t:
        return True
    # Accept a short phrase or a number; but reject long multi-sentence explanations.
    return ("\n" in t) and (len(t.splitlines()) > 2)

def numeric_error(pred: str, gold: str, question: str) -> Optional[Dict[str, float]]:
    n = normalize_pred_to_gold_scale(question, gold, pred)
    if n is None:
        return None
    pval, gval, _ = n
    abs_err = abs(pval - gval)
    rel_err = abs_err / (abs(gval) + 1e-9)
    return {"abs_err": abs_err, "rel_err": rel_err}

def judge_correct(question: str, gold: str, pred: str, model: str = DEFAULT_LLM_MODEL) -> Optional[bool]:
    prompt = f"QUESTION:\n{question}\n\nGOLD:\n{gold}\n\nPREDICTED:\n{pred}\n\nLabel:"
    resp = ask_ollama(prompt, system=JUDGE_SYSTEM, model=model)
    ans = resp["text"].strip().upper()
    if ans.startswith("CORRECT"):
        return True
    if ans.startswith("WRONG"):
        return False
    return None

# -------------------------
# Runner
# -------------------------
def run(split: str, limit: int, out_csv: str, k: int, store_root: str, reset: bool, judge: bool,
        llm_model: str, embed_model: str, data_dir: str) -> None:
    ds = load_finqa_local(split, data_dir=data_dir)

    if reset:
        reset_store_root(store_root)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    latencies_ms: List[float] = []
    search_ms_list: List[float] = []

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()

        for i in range(min(limit, len(ds))):
            ex = ds[i]
            context, question, gold = finqa_normalize(ex)

            if not question or not gold:
                print(f"[{i}] Skipping: missing question/answer")
                continue

            facts = build_facts(ex)
            retrieved, search_ms = retrieve_topk_facts(facts, question, k=k, store_root=store_root, embed_model=embed_model)
            search_ms_list.append(search_ms)

            prompt = build_rag_prompt(retrieved, question)
            try:
                resp = ask_ollama(prompt, model=llm_model)
                pred = resp["text"]
                latency_ms = resp["latency_ms"]
                prompt_chars = resp["prompt_chars"]
                output_chars = resp["output_chars"]
            except Exception as e:
                # Don't fail the whole run on a single example.
                print(f"[{i}] ERROR during LLM call: {type(e).__name__}: {e}")
                pred = ""
                latency_ms = None
                prompt_chars = len(prompt)
                output_chars = 0

            ps = parse_success(pred)
            fv = format_violation(pred)
            multi = has_multiple_numbers(pred)
            err = numeric_error(pred, gold, question)
            if judge:
                try:
                    j_ok = judge_correct(question, gold, pred, model=llm_model)
                except Exception as e:
                    print(f"[{i}] ERROR during judge call: {type(e).__name__}: {e}")
                    j_ok = None
            else:
                j_ok = None

            row = {
                "idx": i,
                "method": "rag",
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
                "latency_ms": (round(latency_ms, 1) if latency_ms is not None else None),
                "search_ms": round(search_ms, 1),
                "prompt_chars": prompt_chars,
                "output_chars": output_chars,
                # Keep schema parity with baseline v2
                "memory_items": None,
                "memory_chars": None,
            }
            w.writerow(row)
            if latency_ms is not None:
                latencies_ms.append(latency_ms)

            print(
                f"[{i}] rag exact={row['exact_match']} tol={row['numeric_close']} "
                f"parse={ps} judge={j_ok} latency_ms={row['latency_ms']} search_ms={row['search_ms']} "
                f"ret={len(retrieved)} pred={pred} gold={gold}"
            )

    p50 = percentile(latencies_ms, 50)
    p95 = percentile(latencies_ms, 95)
    print(f"\nSaved results to: {out_csv}")
    if p50 is not None and p95 is not None:
        print(f"Latency p50={p50:.1f}ms p95={p95:.1f}ms")

    sp50 = percentile(search_ms_list, 50)
    sp95 = percentile(search_ms_list, 95)
    if sp50 is not None and sp95 is not None:
        print(f"Search p50={sp50:.1f}ms p95={sp95:.1f}ms")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", choices=["train", "validation", "test", "dev", "val"])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", default="results/finqa_rag.csv")
    p.add_argument("--k", type=int, default=12, help="Top-k facts to retrieve per question")
    p.add_argument("--store-root", default=DEFAULT_STORE_ROOT, help="Embedding cache directory")
    p.add_argument("--reset-store-root", action="store_true", help="Reset embedding cache before run")
    p.add_argument("--judge", action="store_true", help="Run LLM-as-a-judge evaluation (extra calls)")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    args = p.parse_args()

    run(
        split=args.split,
        limit=args.limit,
        out_csv=args.out,
        k=args.k,
        store_root=args.store_root,
        reset=args.reset_store_root,
        judge=args.judge,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        data_dir=args.data_dir,
    )
