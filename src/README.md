# Architecture Matters More Than Scale
### A Comparative Study of Retrieval and Memory Augmentation for Financial QA Under SME Compute Constraints

**Authors:** Jianan Liu, Jing Yang, Xianyou Li, Weiran Yan, Yichao Wu, Penghao Liang, Mengwei Yuan

---

## Overview

This repository contains the evaluation code and per-sample result logs for our paper comparing four LLM-based financial QA architectures across the [FinQA](https://github.com/czyssrs/FinQA) and [ConvFinQA](https://github.com/czyssrs/ConvFinQA) benchmarks.

All experiments use a locally hosted **Llama 3.1 8B** (instruction-tuned) served via [Ollama](https://ollama.ai), with `nomic-embed-text` for embeddings — no cloud API required.

---

## Architectures

| Architecture | Description |
|---|---|
| **Baseline LLM** | Full document context injected directly into the prompt. No retrieval or memory. |
| **RAG** | Per-query cosine similarity retrieval over decomposed document facts (PRE/TABLE/POST). No persistent memory. |
| **Mem0-Augmented** | Document context and Q/A history stored as free-form text in Mem0; retrieved memories injected per turn. |
| **Structured Mem0** | Table rows serialized as typed `entity \| column = value` facts; stored with `infer=False` for deterministic embedding. Composite-row filter removes multi-attribute distractors. |

---

## Key Results

### FinQA (492 validation samples)

| Method | Corr. Close | Corr. Exact | Judge | p50 (ms) |
|---|---|---|---|---|
| Structured Mem0 | **0.427** | **0.354** | 0.565 | 1,019 |
| Baseline LLM | 0.386 | 0.319 | **0.583** | 1,466 |
| Mem0-Augmented | 0.309 | 0.238 | 0.514 | 2,100 |
| RAG | 0.311 | 0.256 | 0.537 | **913** |

### ConvFinQA (500 dialogs, 1,490 turns)

| Method | Auto Close | Exact Match | Judge | p50 (ms) |
|---|---|---|---|---|
| **RAG** | **52.75%** | **43.49%** | 57.52% | 2,517 |
| Baseline LLM | 48.46% | 41.95% | 52.08% | 1,317 |
| Structured Mem0 | 46.64% | 38.19% | 48.86% | 2,625 |
| Mem0-Augmented | 43.22% | 36.17% | **63.76%** | 2,951 |

**Key finding:** Structured Mem0 wins on FinQA (deterministic, operand-explicit); RAG wins on ConvFinQA (conversational, reference-implicit). An oracle router combining both achieves +2.9pp combined accuracy over the best single architecture at no additional inference cost.

---

## Repository Structure

```
├── run_finqa_baseline_mem0.py           # FinQA: Baseline LLM + Mem0-Augmented (--mem0 flag)
├── run_finqa_rag.py                # FinQA: RAG
├── run_finqa_structured_mem0.py    # FinQA: Structured Mem0
├── run_convfinqa_baseline.py       # ConvFinQA: Baseline LLM
├── run_convfinqa_mem0_aug.py       # ConvFinQA: Mem0-Augmented
├── run_convfinqa_rag.py            # ConvFinQA: RAG
├── run_convfinqa_structured_mem0.py# ConvFinQA: Structured Mem0
├── results/
│   ├── finQA_baseline_corrected.csv
│   ├── finQA_rag_v2_corrected.csv
│   ├── finQA_mem0_aug_corrected.csv
│   ├── finQA_structured_corrected.csv
│   ├── convfinqa_baseline_corrected.csv
│   ├── convfinqa_rag_corrected.csv
│   ├── convfinqa_mem0_aug_corrected.csv
│   └── convfinqa_structured_corrected.csv
```

---

## Setup

### Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.1:latest
ollama pull nomic-embed-text

# Install Python dependencies
pip install requests mem0ai chromadb
```

### Data

Download the datasets and place them as follows:

```
data/
├── FinQA/dataset/
│   ├── train.json
│   ├── dev.json
│   └── test.json
└── ConvFinQA/dataset/
    ├── train.json
    ├── dev.json
    └── test.json
```

FinQA: https://github.com/czyssrs/FinQA  
ConvFinQA: https://github.com/czyssrs/ConvFinQA

---

## Usage

### FinQA

```bash
# Baseline LLM
python run_finqa_baseline_mem0.py --split validation --limit 492 \
  --out results/finqa_baseline.csv --judge

# Mem0-Augmented
python run_finqa_baseline_mem0.py --split validation --limit 492 \
  --mem0 --reset-store --out results/finqa_mem0_aug.csv --judge

# RAG (k=12 facts per query)
python run_finqa_rag.py --split validation --limit 492 \
  --k 12 --out results/finqa_rag.csv --judge

# Structured Mem0
python run_finqa_structured_mem0.py --split validation --limit 492 \
  --store-table-facts --k 12 --reset-store \
  --out results/finqa_structured_mem0.csv --judge
```

### ConvFinQA

```bash
# Baseline LLM
python run_convfinqa_baseline.py --split dev --max-samples 500 \
  --out results/convfinqa_baseline.csv --judge

# Mem0-Augmented
python run_convfinqa_mem0_aug.py --split dev --max-samples 500 \
  --reset-store --out results/convfinqa_mem0_aug.csv --judge

# RAG
python run_convfinqa_rag.py --split dev --max-samples 500 \
  --k 12 --out results/convfinqa_rag.csv --judge

# Structured Mem0
python run_convfinqa_structured_mem0.py --split dev --max-samples 500 \
  --reset-store --out results/convfinqa_structured_mem0.csv --judge
```

---

## Output CSV Schema

All scripts produce CSVs with a consistent schema:

| Column | Description |
|---|---|
| `idx` | Sample index |
| `method` | Architecture name |
| `question` | Input question |
| `gold_answer` | Ground-truth answer |
| `pred_answer` | Model prediction |
| `corrected_exact` | Exact match after symbolic normalization |
| `corrected_close` | Tolerance-based match after normalization (primary metric) |
| `judge_correct` | LLM-as-judge correctness (secondary metric) |
| `latency_ms` | End-to-end inference time |
| `prompt_chars` | Input prompt length (proxy for token cost) |

---

## Citation

```bibtex
@inproceedings{liu2025architecture,
  title={Architecture Matters More Than Scale: A Comparative Study of Retrieval
         and Memory Augmentation for Financial QA Under SME Compute Constraints},
  author={Liu, Jianan and Yang, Jing and Li, Xianyou and Yan, Weiran and
          Wu, Yichao and Liang, Penghao and Yuan, Mengwei},
  booktitle={IEEE Conference on Financial AI \& Decision Systems},
  year={2025}
}
```
