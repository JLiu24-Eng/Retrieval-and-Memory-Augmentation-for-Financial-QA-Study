#!/usr/bin/env python3
"""
ConvFinQA Baseline Runner (LLM-only) + FinQA-style Evaluation Metrics

Loads ConvFinQA conversation-level JSON and runs a baseline LLM over each turn,
using dialogue history as context. Saves per-turn CSV with metrics:

Strict metrics (simple parsing):
  - exact_match: compare last parsed number (rounded to gold decimals)
  - numeric_close: last-number within tolerance

Corrected metrics (more forgiving & closer to your FinQA corrected eval):
  - corr_exact: best-candidate number match (handles % fraction normalization)
  - corr_close: best-candidate numeric close with percent-aware tolerance
  - abs_err / rel_err (based on corrected best candidate)

Optional judge metrics:
  - --judge: LLM-as-judge correctness (true/false), judge latency

CLI:
  --split train|dev|validation|test   (validation == dev)
  --data-dir data/ConvFinQA/dataset
  --max-samples N   (dialogues)
  --max-turns N     (turns per dialogue)
  --history-turns N (how many prior turns to include)
  --output results/xxx.csv
"""
# Architecture Matters More Than Scale:
# A Comparative Study of Retrieval and Memory Augmentation for Financial QA
# Under SME Compute Constraints
#
# GitHub: https://github.com/JLiu24-Eng/Retrieval-and-Memory-Augmentation-for-Financial-QA-Study
# Authors: Jianan Liu, Jing Yang, Xianyou Li, Weiran Yan, Yichao Wu, Penghao Liang, Mengwei Yuan

import argparse
import json
import time
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


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

    # fallback format
    if "dialogue" in ex and isinstance(ex["dialogue"], list):
        return [str(t.get("question") or t.get("q") or t.get("text") or t) for t in ex["dialogue"]]

    raise KeyError("Could not find dialogue turns. Expected annotation.dialogue_break or dialogue[]")


def get_gold_answers_from_example(ex: Dict[str, Any]) -> List[Any]:
    # ConvFinQA: executed answers per turn
    ann = ex.get("annotation") or {}
    if isinstance(ann, dict):
        lst = ann.get("exe_ans_list")
        if isinstance(lst, list):
            return lst
    return []


# =========================
# LLM client (Ollama)
# =========================

@dataclass
class OllamaClient:
    base_url: str
    model: str
    temperature: float = 0.0
    timeout_s: int = 180

    def generate(self, prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json()
        return (j.get("response") or "").strip()


# =========================
# Prompting
# =========================

def build_prompt(pre_text: str, table_text: str, post_text: str,
                 history: List[Tuple[str, str]],
                 question: str) -> str:
    hist_block = ""
    if history:
        lines = []
        for i, (q, a) in enumerate(history, start=1):
            lines.append(f"Turn {i} Q: {q}")
            lines.append(f"Turn {i} A: {a}")
        hist_block = "\n".join(lines)

    prompt = f"""You are a financial reasoning assistant. Answer with ONLY the final numeric answer (no explanation).

[Pre-text]
{pre_text}

[Table]
{table_text}

[Post-text]
{post_text}

[Dialogue History]
{hist_block if hist_block else "(none)"}

[Current Question]
{question}

Return ONLY the final answer (e.g., 12.3, -45, 7%, 0.12).
"""
    return prompt


def build_judge_prompt(question: str, gold: str, pred: str) -> str:
    return f"""You are grading a numeric QA answer. Reply ONLY 'true' or 'false'.
Question: {question}
Gold answer: {gold}
Model answer: {pred}
Is the model answer correct (allow reasonable rounding and percent/fraction equivalence)?
"""


# =========================
# Evaluation (strict + corrected)
# =========================

# number capture: optional parentheses, sign, commas, decimals, optional %
NUM_RE = re.compile(
    r"(?P<paren>\(\s*)?"
    r"(?P<sign>[-+])?"
    r"(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<paren_end>\s*\))?"
    r"(?P<pct>\s*%)?"
)

def extract_numbers(text: Any) -> List[Tuple[float, bool]]:
    """Return list of (value, has_percent_sign)."""
    if text is None:
        return []
    t = str(text).strip()
    if not t:
        return []
    out: List[Tuple[float, bool]] = []
    for m in NUM_RE.finditer(t):
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


def extract_last_number(text: Any) -> Optional[float]:
    nums = extract_numbers(text)
    if not nums:
        return None
    return nums[-1][0]


def decimals_in_gold(gold: Any) -> int:
    s = str(gold or "").replace(",", "")
    m = re.search(r"\d+(?:\.(\d+))?", s)
    return len(m.group(1)) if m and m.group(1) else 0


def percent_expected(question: str, gold: Any, pred: Any) -> bool:
    q = (question or "").lower()
    g = str(gold or "")
    p = str(pred or "")
    return ("%" in g) or ("%" in p) or ("percent" in q) or ("percentage" in q)


def best_candidate(question: str, gold: Any, pred: Any) -> Optional[Tuple[float, float, float, bool]]:
    """
    Choose the prediction number that best matches gold, with percent normalization.
    Returns: (abs_err, pred_norm, gold_val, want_pct)
    """
    gnums = extract_numbers(gold)
    if not gnums:
        return None
    gv, g_pct = gnums[-1]
    want_pct = percent_expected(question, gold, pred) or g_pct

    best = None
    for pv, pv_pct in extract_numbers(pred):
        pvv = pv
        # If expecting percent but pred is likely a fraction, normalize it (0.12 -> 12)
        if want_pct and (not pv_pct) and 0 <= abs(pvv) <= 1.5:
            pvv *= 100.0
        ae = abs(pvv - gv)
        if best is None or ae < best[0]:
            best = (ae, pvv, gv, want_pct)
    return best


def exact_match(gold: Any, pred: Any) -> bool:
    """Strict exact: compare last parsed numbers, rounding to gold decimals."""
    gv = extract_last_number(gold)
    pv = extract_last_number(pred)
    if gv is None or pv is None:
        return False
    decs = decimals_in_gold(gold)
    if decs == 0:
        return int(round(pv)) == int(round(gv))
    return round(pv, decs) == round(gv, decs)


def numeric_close(gold: Any, pred: Any) -> bool:
    """Strict numeric close: last parsed numbers within tolerance."""
    gv = extract_last_number(gold)
    pv = extract_last_number(pred)
    if gv is None or pv is None:
        return False
    decs = decimals_in_gold(gold)
    if decs == 0:
        tol = 0.5 if abs(gv) < 1000 else max(1.0, abs(gv) * 0.001)
        return abs(pv - gv) <= tol
    if decs == 1:
        return abs(pv - gv) <= 0.15
    return abs(pv - gv) <= 0.05


def corr_exact(question: str, gold: Any, pred: Any) -> bool:
    """Corrected exact: best candidate number, percent normalization, rounded to gold decimals."""
    bc = best_candidate(question, gold, pred)
    if bc is None:
        return False
    _, pv, gv, _ = bc
    decs = decimals_in_gold(gold)
    if decs == 0:
        return int(round(pv)) == int(round(gv))
    return round(pv, decs) == round(gv, decs)


def corr_close(question: str, gold: Any, pred: Any) -> bool:
    """Corrected numeric close: best candidate number, percent-aware tolerance."""
    bc = best_candidate(question, gold, pred)
    if bc is None:
        return False
    _, pv, gv, want_pct = bc
    decs = decimals_in_gold(gold)

    if want_pct:
        # more forgiving tolerance for percent answers
        abs_tol = 0.5 if decs == 0 else (0.15 if decs == 1 else 0.05)
        return math.isclose(pv, gv, abs_tol=abs_tol, rel_tol=0.002)

    if decs == 0:
        tol = 0.5 if abs(gv) < 1000 else max(1.0, abs(gv) * 0.001)
        return abs(pv - gv) <= tol
    if decs == 1:
        return abs(pv - gv) <= 0.15
    return abs(pv - gv) <= 0.05


# =========================
# Progress helpers
# =========================

def _short(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


# =========================
# Main runner
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", choices=["train", "dev", "validation", "test"], default="dev")
    parser.add_argument("--data-dir", default="data/ConvFinQA/dataset")

    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of dialogues (conversations) processed")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Limit number of turns processed per dialogue (depth cap)")
    parser.add_argument("--history-turns", type=int, default=None,
                        help="How many previous turns to include as context (None = all previous turns)")

    parser.add_argument("--print-every", type=int, default=25,
                        help="Print a progress summary every N turns")
    parser.add_argument("--print-question-chars", type=int, default=80)

    parser.add_argument("--output", "--out", dest="output", default="results/convfinqa_baseline.csv")

    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=180)

    parser.add_argument("--judge", action="store_true",
                        help="Enable LLM-as-judge correctness (adds judge_correct/judge_latency_ms)")

    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.split, args.data_dir)
    print(f"[INFO] Loading split={args.split} from: {dataset_path}", flush=True)

    with open(dataset_path, "r") as f:
        conversations = json.load(f)

    if not isinstance(conversations, list):
        raise ValueError("Expected dataset JSON root to be a list of conversations")

    if args.max_samples is not None:
        conversations = conversations[: args.max_samples]

    # estimate total turns
    est_total_turns = 0
    for ex in conversations:
        try:
            turns = get_turns_from_example(ex)
            if args.max_turns is not None:
                turns = turns[: args.max_turns]
            est_total_turns += len(turns)
        except Exception:
            pass

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OllamaClient(
        base_url=args.ollama_url,
        model=args.model,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )

    rows: List[Dict[str, Any]] = []
    global_turn = 0
    start_wall = time.time()
    rolling_lat: List[float] = []

    print(f"[INFO] Dialogs={len(conversations)}  Estimated turns≈{est_total_turns}", flush=True)

    for dialog_idx, ex in enumerate(conversations):
        dialog_id = ex.get("id", f"dialog_{dialog_idx}")

        pre_text = " ".join(ex.get("pre_text", [])) if isinstance(ex.get("pre_text"), list) else str(ex.get("pre_text", ""))
        post_text = " ".join(ex.get("post_text", [])) if isinstance(ex.get("post_text"), list) else str(ex.get("post_text", ""))
        table_text = table_to_text(ex.get("table"))

        turns = get_turns_from_example(ex)
        golds = get_gold_answers_from_example(ex)

        if args.max_turns is not None:
            turns = turns[: args.max_turns]
            if golds:
                golds = golds[: args.max_turns]

        history: List[Tuple[str, str]] = []

        for turn_idx, question in enumerate(turns):
            global_turn += 1

            if args.history_turns is None:
                hist_for_prompt = history
            else:
                hist_for_prompt = history[-args.history_turns:] if args.history_turns > 0 else []

            q_preview = _short(question, args.print_question_chars)
            if est_total_turns > 0:
                print(f'[TURN {global_turn}/{est_total_turns}] dialog={dialog_id} turn={turn_idx} Q="{q_preview}"',
                      flush=True)
            else:
                print(f'[TURN {global_turn}] dialog={dialog_id} turn={turn_idx} Q="{q_preview}"', flush=True)

            prompt = build_prompt(pre_text, table_text, post_text, hist_for_prompt, question)

            t0 = time.time()
            pred = client.generate(prompt)
            latency_ms = (time.time() - t0) * 1000.0

            rolling_lat.append(latency_ms)
            if len(rolling_lat) > 200:
                rolling_lat.pop(0)

            gold = golds[turn_idx] if (golds and turn_idx < len(golds)) else None

            # strict metrics
            em = exact_match(gold, pred)
            nc = numeric_close(gold, pred)

            # corrected metrics
            cem = corr_exact(question, gold, pred)
            cnc = corr_close(question, gold, pred)

            # errors based on corrected best candidate
            bc = best_candidate(question, gold, pred)
            abs_err = bc[0] if bc else None
            rel_err = (bc[0] / (abs(bc[2]) + 1e-9)) if bc else None

            # optional judge
            judge_correct = None
            judge_latency_ms = None
            if args.judge and gold is not None:
                jp = build_judge_prompt(question=question, gold=str(gold), pred=str(pred))
                jt0 = time.time()
                jresp = client.generate(jp).strip().lower()
                judge_latency_ms = (time.time() - jt0) * 1000.0
                judge_correct = True if "true" in jresp else False if "false" in jresp else None

            rows.append({
                "dialog_id": dialog_id,
                "turn_idx": turn_idx,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,

                "latency_ms": latency_ms,
                "prompt_chars": len(prompt),
                "output_chars": len(pred or ""),

                "exact_match": em,
                "numeric_close": nc,
                "corr_exact": cem,
                "corr_close": cnc,
                "abs_err": abs_err,
                "rel_err": rel_err,

                "judge_correct": judge_correct,
                "judge_latency_ms": judge_latency_ms,

                "split": args.split,
                "model": args.model,
            })

            history.append((question, pred))

            if args.print_every and (global_turn % args.print_every == 0):
                elapsed = time.time() - start_wall
                avg_lat = sum(rolling_lat) / max(1, len(rolling_lat))
                rate = global_turn / max(1e-9, elapsed)
                remaining = ""
                if est_total_turns > 0:
                    eta_s = (est_total_turns - global_turn) / max(1e-9, rate)
                    remaining = f"  ETA≈{eta_s/60:.1f}m"
                print(f"[PROGRESS] turns={global_turn} elapsed={elapsed/60:.1f}m "
                      f"avg_latency≈{avg_lat:.0f}ms rate≈{rate:.2f} turns/s{remaining}",
                      flush=True)

        if (dialog_idx + 1) % 10 == 0:
            print(f"[INFO] Completed {dialog_idx+1}/{len(conversations)} dialogues", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    total_elapsed = time.time() - start_wall
    print(f"[OK] Saved results to: {out_path}", flush=True)
    print(f"[DONE] Dialogs={len(conversations)} Turns={len(df)} Elapsed={total_elapsed/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
