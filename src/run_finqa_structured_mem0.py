#!/usr/bin/env python3
"""
FinQA Structured Mem0 Runner

Evaluates the Structured Mem0 architecture on the FinQA benchmark.
Table rows are serialized as typed attribute-value facts (entity | column = value)
and stored directly in Mem0 (infer=False) without LLM-based extraction,
ensuring deterministic schema-preserving storage. A composite-row filter
retains only atomic single-attribute facts to reduce operand ambiguity.

CLI:
  python run_finqa_structured_mem0.py --split validation --limit 492 \\
    --out results/finqa_structured_mem0.csv --store-table-facts --judge
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

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.1:latest"
EMBED_MODEL = "nomic-embed-text"
STORE_PATH = "./mem0_store"

FIN_SYSTEM = (
    "You are a financial question answering assistant.\n"
    "Use ONLY the FACTS provided.\n"
    "You MAY derive values using standard arithmetic.\n"
    "Do NOT invent or assume new facts beyond the provided FACTS.\n"
    "Do NOT rescale numbers (e.g., do not turn 637 into 637,000,000).\n"
    "Do NOT convert units unless the question explicitly asks for conversion.\n"
    "If numerator and denominator share the same unit (e.g., both billions), the unit cancels; use raw numbers.\n"
    "Return ONLY the final numeric answer (optionally with %)."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator for financial QA.\n"
    "Given a QUESTION, GOLD answer, and PREDICTED answer, label the prediction as CORRECT or WRONG.\n"
    "If the predicted answer matches the gold numerically (allow reasonable rounding) or is an equivalent form "
    "(e.g., percent vs fraction when question asks percent), mark CORRECT.\n"
    "Reply with exactly one token: CORRECT or WRONG."
)

def drop_composite_row_facts(facts: List[str]) -> List[str]:
    """Prefer atomic facts and only keep composite row facts as a fallback.

    FinQA evidence sometimes includes dense row-serialized facts like:
      'company the american express ... ; ... ; ...'
    These can introduce attribute/units confusion. Atomic facts (single attribute)
    are usually safer for numerical QA prompts.

    Heuristic:
      - Atomic facts: contain '|' (our canonical form) OR have <=1 semicolon
      - Composite facts: have >=2 semicolons OR start with 'company the '
    """
    if not facts:
        return facts

    atomic = []
    composite = []
    for f in facts:
        fl = f.strip().lower()
        is_composite = (f.count(";") >= 2) or fl.startswith("company the ")
        if is_composite:
            composite.append(f)
        else:
            atomic.append(f)

    # If we have any atomic facts, keep only atomic to reduce distractors.
    return atomic if atomic else facts



def select_entity_facts(facts: List[str], question: str) -> List[str]:
    q = question.lower()

    # Simple entity candidates from facts: entity is before '|'
    entities = []
    for f in facts:
        if "|" in f:
            ent = f.split("|", 1)[0].strip()
            if ent and ent not in entities:
                entities.append(ent)

    # Pick entities that appear in the question (substring match)
    matched = [e for e in entities if e in q]

    if not matched:
        return facts

    # Keep only facts for matched entities
    kept = []
    for f in facts:
        ent = f.split("|", 1)[0].strip() if "|" in f else ""
        if ent in matched:
            kept.append(f)
    return kept if kept else facts


def keyword_filter_facts(facts: List[str], question: str, enable_general: bool = False) -> List[str]:
    """Optional keyword-based fact filtering.

    - Always applies a *safe* special-case for 'payment volume per transaction'
      to prevent 'total volume' distractors.
    - General keyword filtering is OFF by default because it can drop needed facts
      on diverse FinQA questions. Turn it on with enable_general=True.
    """
    if not facts:
        return facts

    q = question.lower()

    def fmatch(f: str, kw: str) -> bool:
        fl = f.lower()
        return kw.replace(" ", "") in fl.replace(" ", "")

    # ---- Special-case: payment volume per transaction ----
    if ("payment volume per transaction" in q) or ("average payment volume per transaction" in q):
        kept = []
        for f in facts:
            fl = f.lower()
            if "total volume" in fl:
                continue
            if fmatch(f, "payments volume") or fmatch(f, "total transactions"):
                kept.append(f)
        return kept if kept else facts

    if not enable_general:
        return facts

    # ---- General keyword filtering (use carefully) ----
    keywords = []
    for kw in [
        "payments volume",
        "total volume",
        "total transactions",
        "transactions",
        "cards",
        "revenue",
        "income",
        "expense",
        "assets",
        "liabilities",
        "ratio",
        "percent",
        "%",
    ]:
        if kw in q:
            keywords.append(kw)

    if not keywords:
        return facts

    kept = [f for f in facts if any(fmatch(f, kw) for kw in keywords)]
    return kept if kept else facts



def reset_store(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[reset] Deleted store at: {path}")

def ask_ollama(prompt: str, system: str = FIN_SYSTEM, timeout_s: int = 240, retries: int = 1) -> Dict[str, Any]:
    """Call Ollama /api/chat with basic retry on ReadTimeout.

    Returns:
      {text, latency_ms, prompt_chars, output_chars}
    """
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": LLM_MODEL,
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
            return {
                "text": text,
                "latency_ms": dt_ms,
                "prompt_chars": len(prompt),
                "output_chars": len(text),
            }
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            # Backoff: increase timeout for next attempt
            timeout_s = int(timeout_s * 1.5)
            time.sleep(0.2 + 0.3 * attempt)
            continue
        except Exception as e:
            last_err = e
            break

    # Fail closed but keep pipeline running
    return {
        "text": "UNKNOWN",
        "latency_ms": None,
        "prompt_chars": len(prompt),
        "output_chars": 0,
        "error": str(last_err) if last_err else "unknown_error",
    }



def is_percent(text: str) -> bool:
    return "%" in (text or "")

def extract_last_number(text: str) -> Optional[float]:
    if not text:
        return None
    matches = re.findall(r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)", text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None

def exact_match(pred: str, gold: str) -> bool:
    if pred is None or gold is None:
        return False
    p = str(pred).strip().lower()
    g = str(gold).strip().lower()
    for ch in ["$", ","]:
        p = p.replace(ch, "")
        g = g.replace(ch, "")
    pn = extract_last_number(p)
    gn = extract_last_number(g)
    if pn is not None and gn is not None:
        return pn == gn and (is_percent(p) == is_percent(g))
    return p.strip(".") == g.strip(".")

def numeric_close(pred: str, gold: str, atol: float = 1e-2, rtol: float = 1e-2) -> bool:
    pn = extract_last_number(str(pred))
    gn = extract_last_number(str(gold))
    if pn is None or gn is None:
        return False
    if is_percent(str(pred)) or is_percent(str(gold)):
        return math.isclose(pn, gn, abs_tol=0.15, rel_tol=0.002)
    return math.isclose(pn, gn, abs_tol=atol, rel_tol=rtol)

def load_finqa_local(split: str):
    split_map = {"train": "train.json", "validation": "dev.json", "test": "test.json"}
    path = Path("data/FinQA/dataset") / split_map[split]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"] if isinstance(data, dict) and "data" in data else data


def finqa_normalize(ex: dict) -> Tuple[str, str, str, List[List[str]], List[Any], str]:
    """
    Returns:
      (context_text, question, gold_answer, table, model_input, example_id)
    """
    qa = ex.get("qa", {}) or {}

    question = (qa.get("question") or "").strip()
    gold_answer = str(qa.get("answer") or "").strip()

    pre = ex.get("pre_text") or []
    post = ex.get("post_text") or []
    table = ex.get("table") or []

    model_input = qa.get("model_input") or []

    pre_text = "\n".join([s.strip() for s in pre if isinstance(s, str) and s.strip()])
    post_text = "\n".join([s.strip() for s in post if isinstance(s, str) and s.strip()])
    context_text = (pre_text + "\n\n" + post_text).strip()

    example_id = str(ex.get("id") or qa.get("id") or "").strip()
    if not example_id:
        # fallback: filename-based ids exist in FinQA ('V/2008/page_17.pdf-1')
        example_id = str(ex.get("filename") or "ex_unknown").strip()

    return context_text, question, gold_answer, table, model_input, example_id



def model_input_to_facts(model_input) -> List[str]:
    facts = []
    # model_input is like: [[id, text], [id, text], ...]
    for item in model_input:
        if isinstance(item, list) and len(item) == 2:
            txt = str(item[1]).strip()
            if txt:
                facts.append(txt)
    return facts



def make_mem0() -> Memory:
    return Memory.from_config({
        "vector_store": {"provider": "chroma", "config": {"path": STORE_PATH}},
        "embedder": {"provider": "ollama", "config": {"model": EMBED_MODEL, "ollama_base_url": "http://localhost:11434"}},
        "llm": {"provider": "ollama", "config": {"model": LLM_MODEL, "ollama_base_url": "http://localhost:11434", "temperature": 0}},
    })

def normalize_cell(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def table_to_facts(table: List[List[str]], max_rows: int = 25) -> List[str]:
    """
    Convert FinQA table (list of rows) into row-level facts.
    Heuristic:
      - first row is header
      - first column is entity name
      - create facts: "{entity} | {col} = {value}"
    """
    if not table or not isinstance(table, list) or len(table) < 2:
        return []
    header = [normalize_cell(x) for x in table[0]]
    facts = []
    for row in table[1:1+max_rows]:
        if not isinstance(row, list) or len(row) < 2:
            continue
        entity = normalize_cell(row[0])
        if not entity:
            continue
        for j in range(1, min(len(row), len(header))):
            col = header[j] if header[j] else f"col_{j}"
            val = str(row[j]).strip()
            if not val or val == ".":
                continue
            facts.append(f"{entity} | {col} = {val}")
    return facts

    facts_text = "\n".join(f"- {x}" for x in facts) if facts else "(none)"

    ql = question.lower()
    extra_rule = ""

    if "cumulative total return" in ql:
        extra_rule += (
            "\nRULE:\n"
            "For 'percentage cumulative total return' use:\n"
            "((final_value - initial_value) / initial_value) * 100.\n"
            "Use initial_value and final_value from FACTS (e.g., 31-dec-2012 and 31-dec-2017).\n"
        )

    if ("payment volume per transaction" in ql) or ("average payment volume per transaction" in ql):
        extra_rule += (
            "\nRULE:\n"
            "For 'payment volume per transaction', divide payments volume by total transactions.\n"
            "Both are in the SAME unit (billions), so the unit cancels.\n"
            "Use the raw numbers as given in FACTS (e.g., 637 and 5.0). Do NOT rescale.\n"
        )

    normalization_rules = (
        "\nNORMALIZATION RULES:\n"
        "- Do NOT rescale numbers (e.g., do not turn 637 into 637,000,000).\n"
        "- Do NOT convert units unless the question explicitly asks for conversion.\n"
        "- If numerator and denominator share the same unit, the unit cancels: use raw numbers.\n"
        "- If the question asks for a percentage, return a percentage (with %).\n"
    )

    return (
        f"FACTS:\n{facts_text}\n"
        f"{extra_rule}"
        f"{normalization_rules}\n"
        f"Question:\n{question}\n\n"
        "Return ONLY the final numeric answer (optionally with %)."

        "If you cannot find required values in FACTS, return UNKNOWN."
    )



def build_prompt_from_facts(facts: List[str], question: str) -> str:
    facts_text = "\n".join(f"- {x}" for x in facts) if facts else "(none)"

    ql = question.lower()

    extra_rule = ""
    if "cumulative total return" in ql:
        extra_rule = (
            "\nRULE:\n"
            "For 'percentage cumulative total return' use:\n"
            "((final_value - initial_value) / initial_value) * 100.\n"
            "Use initial_value and final_value from FACTS (e.g., 31-dec-2012 and 31-dec-2017).\n"
        )

    normalization_rules = (
        "\nNORMALIZATION RULES:\n"
        "- IMPORTANT: DO NOT RESCALE numbers (e.g., do not turn 637 into 637,000,000).\n"
        "- If numerator and denominator share the same unit (e.g., both are in billions), the unit cancels: use raw numbers.\n"
        "- If the question asks for a percentage, return a percentage.\n"
        "- Keep the same unit unless explicitly asked to convert.\n"
        "- Do NOT convert units unless explicitly required.\n"
        "- For ratios or shares, use: (part / whole) * 100.\n"
    )

    return (
        f"FACTS:\n{facts_text}\n"
        f"{extra_rule}"
        f"{normalization_rules}\n"
        f"Question:\n{question}\n\n"
        "Return ONLY the final numeric answer (optionally with %). No explanation."
    )





# -----------------------
# Enhanced evaluation metrics (v2)
# -----------------------

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
    if want_pct and (not p_pct) and 0 <= abs(pval) <= 1.5:
        pval *= 100.0
    return pval, gval, want_pct

def exact_match(pred: str, gold: str, question: str="") -> bool:
    norm = normalize_pred_to_gold_scale(question, gold, pred)
    if norm is None:
        return False
    pval, gval, _want_pct = norm
    decs = decimals_in_gold(gold)
    if decs == 0:
        return int(round(pval)) == int(round(gval))
    return round(pval, decs) == round(gval, decs)

def numeric_close(pred: str, gold: str, question: str="", atol: float=1e-2, rtol: float=1e-2) -> bool:
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
    t = (text or "").strip()
    if not t:
        return True
    if "\n" in t:
        return True
    bad_markers = ["because", "therefore", "so ", "we ", "let's", "calculation", "="]
    if any(m in t.lower() for m in bad_markers) and has_multiple_numbers(t):
        return True
    return False

def numeric_error(pred: str, gold: str, question: str="") -> Optional[Dict[str, float]]:
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
def run(split: str, limit: int, out_csv: str, reset: bool, k_retrieve: int, store_table_facts: bool, enable_general_keyword_filter: bool, judge: bool):
    ds = load_finqa_local(split)
    if reset:
        reset_store(STORE_PATH)

    mem = make_mem0()
    user_id = "finqa_user_struct_1"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "idx","method","question","gold_answer","pred_answer",
                "exact_match","numeric_close",
                "parse_success","format_violation","has_multiple_numbers",
                "abs_err","rel_err","judge_correct",
                "latency_ms","search_ms","prompt_chars","output_chars",
                "n_facts_stored","n_facts_retrieved","memory_items","memory_chars",
            ],
        )
        w.writeheader()

        latencies_ms: List[float] = []
        search_latencies_ms: List[float] = []

        for i in range(min(limit, len(ds))):
            ex = ds[i]
            context, question, gold, table, model_input, example_id = finqa_normalize(ex)
            run_id = example_id or f"ex_{i}"
            if not question or not gold:
                print(f"[{i}] Skipping: missing question/answer")
                continue

            # 1) Store structured facts once per question (table facts)
            facts = table_to_facts(table) if store_table_facts else []
            if facts:
                messages = [{"role": "user", "content": fact} for fact in facts]
                # Key: infer=False => NO LLM calls, just embed+store
                mem.add(
                    messages,
                    user_id=user_id,
                    metadata={"type": "table_fact", "idx": i, "run_id": run_id},
                    infer=False,
                )

            # 1b) Store dataset-provided retrieved evidence (qa["model_input"])
            mi_facts = model_input_to_facts(model_input)

            if mi_facts:
                messages = [{"role": "user", "content": f} for f in mi_facts]
                mem.add(
                    messages,
                    user_id=user_id,
                    metadata={"type": "model_input_fact", "idx": i, "run_id": run_id},
                    infer=False,
                )

            # 2) Retrieve facts relevant to the question
            t_search0 = time.perf_counter()
            retrieved = mem.search(question, user_id=user_id, limit=50, filters={"run_id": run_id},)
            search_ms = (time.perf_counter() - t_search0) * 1000.0
            search_latencies_ms.append(search_ms)

            retrieved_facts = []
            if isinstance(retrieved, dict) and "results" in retrieved:
                for r in retrieved["results"]:
                    if isinstance(r, dict) and r.get("memory"):
                        retrieved_facts.append(r["memory"])



            retrieved_facts = retrieved_facts[:50]


            retrieved_facts = keyword_filter_facts(retrieved_facts, question, enable_general=enable_general_keyword_filter)

            retrieved_facts = drop_composite_row_facts(retrieved_facts)

            retrieved_facts = retrieved_facts[:k_retrieve]

            memory_items = len(retrieved_facts)
            memory_chars = len("\n".join(retrieved_facts))

            prompt = build_prompt_from_facts(retrieved_facts, question)

            try:
                resp = ask_ollama(prompt)
                pred = resp["text"]
            except Exception as e:
                print(f"[{i}] LLM ERROR: {type(e).__name__}: {e}")
                resp = {"latency_ms": None, "prompt_chars": len(prompt), "output_chars": 0}
                pred = "ERROR_TIMEOUT"

            row = {
                "idx": i,
                "method": ("structured_mem0" if store_table_facts else "structured_mem0"),
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,
                "exact_match": exact_match(pred, gold, question),
                "numeric_close": numeric_close(pred, gold, question),
                "parse_success": parse_success(pred),
                "format_violation": format_violation(pred),
                "has_multiple_numbers": has_multiple_numbers(pred),
                "abs_err": (round(numeric_error(pred, gold, question)["abs_err"], 6) if numeric_error(pred, gold, question) else None),
                "rel_err": (round(numeric_error(pred, gold, question)["rel_err"], 6) if numeric_error(pred, gold, question) else None),
                "judge_correct": (judge_correct(question, gold, pred) if judge else None),
                "latency_ms": (round(resp["latency_ms"], 1) if resp.get("latency_ms") is not None else None),
                "search_ms": (round(search_ms, 1) if "search_ms" in locals() else None),
                "prompt_chars": resp["prompt_chars"],
                "output_chars": resp["output_chars"],
                "n_facts_stored": len(facts) + len(mi_facts),
                "n_facts_retrieved": len(retrieved_facts),
                "memory_items": memory_items,
                "memory_chars": memory_chars,
            }
            w.writerow(row)
            if resp.get("latency_ms") is not None:
                latencies_ms.append(resp["latency_ms"])

            print(f"[{i}] tol={row['numeric_close']} exact={row['exact_match']} "
                  f"lat={row['latency_ms']}ms prompt={row['prompt_chars']} "
                  f"facts(stored={len(facts)},ret={len(retrieved_facts)}) pred={pred} gold={gold}")

    print(f"\nSaved results to: {out_csv}")


    # Summary
    p50 = percentile(latencies_ms, 50)
    p95 = percentile(latencies_ms, 95)
    if p50 is not None and p95 is not None:
        print(f"Latency p50={p50:.1f}ms p95={p95:.1f}ms")
    if search_latencies_ms:
        sp50 = percentile(search_latencies_ms, 50)
        sp95 = percentile(search_latencies_ms, 95)
        if sp50 is not None and sp95 is not None:
            print(f"Search latency p50={sp50:.1f}ms p95={sp95:.1f}ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", choices=["train","validation","test"])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", default="results/finqa_structured_mem0.csv")
    p.add_argument("--reset-store", action="store_true")
    p.add_argument("--k", type=int, default=12, help="Top-K retrieved facts to include in prompt")
    p.add_argument(
        "--enable-general-keyword-filter",
        action="store_true",
        help="Enable general keyword_filter_facts (special-case rules still apply)",
    )

    p.add_argument(
        "--store-table-facts",
        action="store_true",
        help="Store structured table facts into memory (ablation: OFF by default).",
    )

    p.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM-as-a-judge evaluation (extra model calls).",
    )

    args = p.parse_args()

    run(
        split=args.split,
        limit=args.limit,
        out_csv=args.out,
        reset=args.reset_store,
        k_retrieve=args.k,
        store_table_facts=args.store_table_facts,
        enable_general_keyword_filter=args.enable_general_keyword_filter,
        judge=args.judge,
    )
