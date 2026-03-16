#!/usr/bin/env python3
"""
ConvFinQA RAG Runner

Retrieval-Augmented Generation for the ConvFinQA benchmark.
No persistent memory — per-turn in-context retrieval only.

- No Mem0, no memory lifecycle.
- In-dialogue retrieval only: context facts -> embeddings -> cosine top-k.
- Embeddings via Ollama /api/embeddings (embed-model).
- Chat via Ollama /api/chat.

CSV schema (identical to FinQA structured mem0 script):
  idx, method, question, gold_answer, pred_answer,
  exact_match, numeric_close,
  parse_success, format_violation, has_multiple_numbers,
  abs_err, rel_err, judge_correct,
  latency_ms, search_ms, prompt_chars, output_chars,
  n_facts_stored, n_facts_retrieved, memory_items, memory_chars
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
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# -------------------------
# Defaults / env (parity)
# -------------------------
DEFAULT_DATA_DIR = "data/ConvFinQA/dataset"
DEFAULT_STORE_ROOT = "./rag_embed_cache_convfinqa"   # embedding cache only
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

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

FINQA_FIELDNAMES = [
    "idx","method","question","gold_answer","pred_answer",
    "exact_match","numeric_close",
    "parse_success","format_violation","has_multiple_numbers",
    "abs_err","rel_err","judge_correct",
    "latency_ms","search_ms","prompt_chars","output_chars",
    "n_facts_stored","n_facts_retrieved","memory_items","memory_chars",
]

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
# Facts from table/text
# -------------------------
def _norm_cell(s: Any) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def table_to_facts(table: Any, max_rows: int = 40) -> List[str]:
    if not table or not isinstance(table, list) or len(table) < 2:
        return []
    if not isinstance(table[0], list):
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
    facts.extend(tbl_facts)
    for s in pre_lines:
        facts.append(f"context | pre_text = {s}")
    for s in post_lines:
        facts.append(f"context | post_text = {s}")
    seen = set()
    out = []
    for f in facts:
        k = f.strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out

# -------------------------
# Embedding cache + retrieval
# -------------------------
def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

class EmbedCache:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Optional[List[float]]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and isinstance(obj.get("embedding"), list):
                return [float(x) for x in obj["embedding"]]
        except Exception:
            return None
        return None

    def set(self, key: str, emb: List[float]) -> None:
        p = self._path(key)
        try:
            p.write_text(json.dumps({"embedding": emb}), encoding="utf-8")
        except Exception:
            pass

def ollama_embed(text: str, embed_model: str, ollama_url: str, timeout_s: int, cache: Optional[EmbedCache]) -> List[float]:
    key = sha1_text(f"{embed_model}::{text}")
    if cache:
        hit = cache.get(key)
        if hit is not None:
            return hit
    url = ollama_url.rstrip("/") + "/api/embeddings"
    r = requests.post(url, json={"model": embed_model, "prompt": text}, timeout=int(timeout_s))
    r.raise_for_status()
    emb = r.json().get("embedding")
    if not isinstance(emb, list):
        raise ValueError("Ollama embeddings response missing 'embedding' list")
    emb_f = [float(x) for x in emb]
    if cache:
        cache.set(key, emb_f)
    return emb_f

def build_index(facts: List[str], embed_model: str, ollama_url: str, timeout_s: int, cache: Optional[EmbedCache]) -> Tuple[List[str], List[List[float]]]:
    texts: List[str] = []
    embs: List[List[float]] = []
    for f in facts:
        texts.append(f)
        embs.append(ollama_embed(f, embed_model=embed_model, ollama_url=ollama_url, timeout_s=timeout_s, cache=cache))
    return texts, embs

def retrieve_topk(query: str, texts: List[str], embs: List[List[float]], embed_model: str, ollama_url: str, timeout_s: int, cache: Optional[EmbedCache], k: int) -> List[str]:
    qemb = ollama_embed(query, embed_model=embed_model, ollama_url=ollama_url, timeout_s=timeout_s, cache=cache)
    scored: List[Tuple[float, int]] = []
    for i, e in enumerate(embs):
        scored.append((cosine_sim(qemb, e), i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [texts[i] for _, i in scored[:max(0, k)] if texts[i]]

# -------------------------
# Ollama chat helper
# -------------------------
def ask_ollama(prompt: str, system: str, model: str, ollama_url: str, timeout_s: int, temperature: float, retries: int = 1) -> Dict[str, Any]:
    chat_url = ollama_url.rstrip("/") + "/api/chat"
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                chat_url,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": float(temperature), "num_predict": 256},
                },
                timeout=int(timeout_s),
            )
            r.raise_for_status()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            text = r.json()["message"]["content"].strip()
            return {"text": text, "latency_ms": dt_ms, "prompt_chars": len(prompt), "output_chars": len(text)}
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            timeout_s = int(timeout_s * 1.25)
            time.sleep(0.2 + 0.3 * attempt)
            continue
        except Exception as e:
            last_err = e
            break
    return {"text": "UNKNOWN", "latency_ms": None, "prompt_chars": len(prompt), "output_chars": 0, "error": str(last_err) if last_err else "unknown_error"}

# -------------------------
# Prompt
# -------------------------
def build_prompt(facts: List[str], question: str, history: List[Tuple[str, str]]) -> str:
    facts_text = "\n".join(f"- {x}" for x in facts) if facts else "(none)"
    hist_block = ""
    if history:
        lines = []
        for i, (q, a) in enumerate(history, start=1):
            lines.append(f"Turn {i} Q: {q}")
            lines.append(f"Turn {i} A: {a}")
        hist_block = "\n".join(lines)

    normalization_rules = (
        "\nNORMALIZATION RULES:\n"
        "- Do NOT rescale numbers (e.g., do not turn 637 into 637,000,000).\n"
        "- Do NOT convert units unless the question explicitly asks for conversion.\n"
        "- If numerator and denominator share the same unit, the unit cancels: use raw numbers.\n"
        "- If the question asks for a percentage, return a percentage (with %).\n"
    )
    return (
        f"FACTS:\n{facts_text}\n\n"
        f"DIALOGUE HISTORY:\n{hist_block if hist_block else '(none)'}\n"
        f"{normalization_rules}\n"
        f"QUESTION:\n{question}\n\n"
        f"FINAL ANSWER:"
    )

# -------------------------
# Evaluation
# -------------------------
_NUM_TOKEN_RE = re.compile(
    r"""
    (?P<paren>\(\s*)?
    (?P<sign>[-+])?
    (?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)
    (?P<paren_end>\s*\))?
    (?P<pct>\s*%)?
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
    return nums[-1] if nums else None

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
    pval, gval, _ = norm
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
    pval, gval, _ = norm
    abs_err = abs(pval - gval)
    rel_err = abs_err / (abs(gval) + 1e-9)
    return {"abs_err": abs_err, "rel_err": rel_err}

def judge_correct(question: str, gold: Any, pred: Any, model: str, ollama_url: str, timeout_s: int) -> Optional[bool]:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"GOLD:\n{gold}\n\n"
        f"PREDICTED:\n{pred}\n\n"
        "Label:"
        "\n(Reply with exactly one token: CORRECT or WRONG)"
    )
    resp = ask_ollama(prompt, system=JUDGE_SYSTEM, model=model, ollama_url=ollama_url, timeout_s=timeout_s, temperature=0.0)
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
def run(
    split: str,
    data_dir: str,
    max_samples: Optional[int],
    max_turns: Optional[int],
    history_turns: Optional[int],
    out_csv: str,
    reset_store_root: bool,
    k_retrieve: int,
    judge: bool,
    print_every: int,
    store_root: str,
    llm_model: str,
    embed_model: str,
    ollama_url: str,
    timeout_s: int,
    temperature: float,
) -> None:
    dataset_path = resolve_dataset_path(split, data_dir)
    print(f"[INFO] Loading split={split} from: {dataset_path}", flush=True)

    if reset_store_root and os.path.exists(store_root):
        shutil.rmtree(store_root)
    os.makedirs(store_root, exist_ok=True)
    cache = EmbedCache(store_root)

    with open(dataset_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    if not isinstance(conversations, list):
        raise ValueError("Expected dataset JSON root to be a list of conversations")

    if max_samples is not None:
        conversations = conversations[:max_samples]

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
            turns = get_turns_from_example(ex)
            golds = get_gold_answers_from_example(ex)
            if max_turns is not None:
                turns = turns[:max_turns]

            context_facts = build_context_facts(ex)
            texts, embs = build_index(context_facts, embed_model=embed_model, ollama_url=ollama_url, timeout_s=timeout_s, cache=cache)

            history: List[Tuple[str, str]] = []

            for t_i, question in enumerate(turns):
                global_idx += 1
                gold = golds[t_i] if (golds and t_i < len(golds)) else None

                if history_turns is None:
                    hist_for_prompt = history
                elif history_turns <= 0:
                    hist_for_prompt = []
                else:
                    hist_for_prompt = history[-history_turns:]

                t_search0 = time.perf_counter()
                remembered = retrieve_topk(question, texts=texts, embs=embs, embed_model=embed_model, ollama_url=ollama_url, timeout_s=timeout_s, cache=cache, k=k_retrieve)
                search_ms = (time.perf_counter() - t_search0) * 1000.0
                search_latencies_ms.append(search_ms)

                memory_items = len(remembered)
                memory_chars = len("\n".join(remembered)) if remembered else 0

                prompt = build_prompt(remembered, question, hist_for_prompt)
                resp = ask_ollama(prompt, system=FIN_SYSTEM, model=llm_model, ollama_url=ollama_url, timeout_s=timeout_s, temperature=temperature)
                pred = resp.get("text", "")

                history.append((question, pred))

                err = numeric_error(pred, gold, question) if gold is not None else None

                row = {
                    "idx": global_idx,
                    "method": "rag",
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
                    "judge_correct": (judge_correct(question, gold, pred, llm_model, ollama_url=ollama_url, timeout_s=timeout_s) if (judge and gold is not None) else None),
                    "latency_ms": (round(resp["latency_ms"], 1) if resp.get("latency_ms") is not None else None),
                    "search_ms": round(search_ms, 1),
                    "prompt_chars": resp.get("prompt_chars"),
                    "output_chars": resp.get("output_chars"),
                    "n_facts_stored": len(context_facts) if t_i == 0 else 0,
                    "n_facts_retrieved": len(remembered),
                    "memory_items": memory_items,
                    "memory_chars": memory_chars,
                }

                w.writerow(row)

                if resp.get("latency_ms") is not None:
                    latencies_ms.append(resp["latency_ms"])

                print(
                    f"[{global_idx}] rag dialog={dialog_id} turn={t_i} "
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

    print(f"\nSaved results to: {out_csv}", flush=True)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["train", "dev", "validation", "test"])
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--max-samples", type=int, default=20, help="Limit number of dialogues (parity with mem0_augmented)")
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--history-turns", type=int, default=None, help="default=None (parity with baseline/mem0_augmented)")
    p.add_argument("--out", default="results/convfinqa_rag.csv")
    p.add_argument("--reset-store-root", action="store_true", help="Delete embedding cache directory before run")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--print-question-chars", type=int, default=80, help="(reserved, kept for parity)")
    p.add_argument("--store-root", default=DEFAULT_STORE_ROOT, help="Embedding cache directory")
    p.add_argument("--model", default=DEFAULT_LLM_MODEL, help="Ollama chat model name")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model name")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--timeout-s", type=int, default=180)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    run(
        split=args.split,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        max_turns=args.max_turns,
        history_turns=args.history_turns,
        out_csv=args.out,
        reset_store_root=args.reset_store_root,
        k_retrieve=args.k,
        judge=args.judge,
        print_every=args.print_every,
        store_root=args.store_root,
        llm_model=args.model,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
        timeout_s=args.timeout_s,
        temperature=args.temperature,
    )

if __name__ == "__main__":
    main()
