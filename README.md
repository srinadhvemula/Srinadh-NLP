# Srinadh-NLP

## BERT-LS: Lexical Simplification with Pretrained Encoders

A context-aware lexical simplification system that replaces complex words with simpler alternatives while preserving meaning and grammaticality.

---

## Overview

Lexical simplification (LS) replaces complex words in a sentence with simpler alternatives of equivalent meaning. Traditional unsupervised approaches only consider the complex word in isolation, producing many spurious candidates that do not fit the sentence context.

BERT-LS solves this by feeding the **entire sentence** into BERT's masked language model. The target word is masked in the input, but the original unmasked sentence is appended as context — giving BERT a "memory" of the word while generating simpler alternatives. Candidates generated this way are contextually appropriate: they fit the grammar, meaning, and topic of the sentence.

The system ranks candidates using five signals (BERT prediction order, semantic similarity, word frequency, language model fluency, and paraphrase database quality) and applies a semantic guard to prevent meaning reversal before committing to a replacement.

**Key results** (this implementation, lite mode — no FastText):

| Test Suite | Score |
|------------|-------|
| Smoke test (21 items) | 20/21 — 95.2% |
| Accuracy test (45 items) | 39/45 — 86.7% |
| BenchLS sample (10 items) | 6/10 — 60.0% |
| Engine startup time | ~17 seconds |

For detailed analysis see [ANALYSIS.md](ANALYSIS.md). For the readability-aware innovation see [INNOVATION.md](INNOVATION.md).

---


## Architecture

### Sentence-Pair Input Design

```
Input:  "The physician [MASK] the examination."
        "The physician administered the examination."
         ↓ concatenated as a sentence pair ↓
[CLS] The physician [MASK] the examination. [SEP]
      The physician administered the examination. [SEP]
```

The second copy gives BERT bidirectional context about both the sentence **and** the specific word being replaced. This is a zero-shot use of the pretrained model — no fine-tuning required.

### Five-Stage Pipeline

```
sentence + target_word
       |
   [1] Tokenisation & Position Tracking
       BERT WordPiece tokenisation, NLTK word-to-token index alignment
       |
   [2] Complex Word Identification
       Primary: gpu_attention.model (TF sequence labeler)
       Fallback: 50% Zipf frequency + 25% word length + 25% syllable count
       Threshold: 0.45 composite score
       |
   [3] BERT Candidate Generation
       Build sentence-pair features with target masked
       Extract top-80 BERT vocabulary predictions at mask position
       Filter: subword tokens (##), same-stem words (Porter Stemmer)
       Return: top-10 candidates
       |
   [4] PPDB Augmentation
       SQLite query against 299 MB paraphrase database
       Returns candidates with AGigaSim quality scores
       |
   [5] Weighted Rank Fusion + Semantic Guard
       5-factor scoring → select best candidate
       Guard: Wu-Palmer similarity ≥ 0.55, antonym rejection
       Morphological post-processing (tense, number, comparative)
       |
output: replacement word (or original if no valid substitution)
```

### Five-Factor Ranking

| Factor | Weight | Computation |
|--------|--------|-------------|
| BERT Prediction Order | 1.0 | Ordinal rank in BERT top-80 |
| Semantic Similarity | **1.5** | FastText cosine / Wu-Palmer fallback |
| Word Frequency | 0.8 | SUBTLEX Zipf frequency rank |
| LM Fluency | 1.2 | Pseudo-perplexity at mask position |
| PPDB Quality | 0.5 | AGigaSim score from paraphrase DB |

Weights are tuned asymmetrically (post-paper refinement): semantic similarity and LM fluency are weighted higher than raw BERT prediction order. PPDB is a bonus signal, not the primary driver.

### Two Operating Modes

| Mode | RAM | Startup | Semantic similarity | BenchLS accuracy |
|------|-----|---------|---------------------|-----------------|
| **Lite** (no FastText) | ~2 GB | ~17 s | Wu-Palmer (WordNet) | ~60% |
| **Full** (with FastText) | ~8 GB | ~90 s | FastText cosine | ~70–75% |

<center><img width="80%" src="https://github.com/qiang2100/BERT-LS/blob/master/BERT_LS.png" /></center>

---

## Example

Given "John composed these verses." with target words **composed** and **verses**:

| System | Candidates: composed | Candidates: verses |
|--------|---------------------|-------------------|
| Glavas (2015) | created, produced, arranged | lines, passages, chapters |
| Paetzold-NE (2017) | wrote, created, made | songs, poems, works |
| **BERT-LS** | **wrote, created, written** | **poems, songs, writings** |

BERT-LS ranks `wrote` and `poems` first because it sees the full sentence context. Result: **"John wrote these poems."**

<center><img width="50%" src="https://github.com/qiang2100/BERT-LS/blob/master/Example1.png" /></center>

---

## Quick Start

### Prerequisites

```bash
pip install torch
pip install flask nltk openpyxl scipy scikit-learn textstat
```

Python 3.8+ recommended. BERT model is downloaded automatically on first run via HuggingFace (~440 MB).

### Required Data Files

| File | Size | Notes |
|------|------|-------|
| `datasets/ppdb.db` | 299 MB | inlcuded (SQLite-indexed PPDB 2.0) |
| `SUBTLEX_frequency.xlsx` | 10.8 MB | Included (74k words, frequency data) |
| `gpu_attention.model` | 62.9 MB | Included (CWI sequence labeler) |
| FastText embeddings | 4.5 GB | Optional — download separately for full mode |

FastText download (optional, full mode only) & PPDB Link for Download:
```
https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
https://www.kaggle.com/datasets/oleksandrrechynskyi/ppdbtldr-paraphrase-dataset
```
Optimized Dataset Link.
```
https://drive.google.com/drive/folders/1Zege1hM3lvpTIENprRr6XkgURfJe0ZIv?usp=sharing
```
### Run the Web App

```bash
# Lite mode (recommended — ~2 GB RAM, ~17s startup)
python app.py --skip-fasttext

# Full mode (requires FastText file at datasets/Fasttext/crawl-300d-2M-subword.vec)
python app.py
```

Open `http://localhost:5000` in a browser.

### Python API

```python
from simplify_engine import SimplificationEngine

engine = SimplificationEngine(
    bert_model='bert-base-uncased',
    word_frequency_path='SUBTLEX_frequency.xlsx',
    ppdb_path='datasets/ppdb.db',
    # word_embeddings_path='datasets/Fasttext/crawl-300d-2M-subword.vec',  # full mode
)

# Simplify a specific word in context
result = engine.simplify_word(
    "The physician administered the preliminary examination.",
    "administered"
)
print(result['replacement'])   # -> 'gave'
print(result['candidates'])    # -> ['gave', 'conducted', 'performed', ...]

# Auto-identify and simplify complex words
result = engine.simplify_sentence(
    "She endeavored to comprehend the intricate methodology."
)
print(result['simplified'])    # -> "She tried to understand the complex process."
```

### Readability-Aware Simplification (Innovation)

```python
from simplify_engine import SimplificationEngine
from readability_analyzer import ReadabilityAnalyzer

engine = SimplificationEngine(
    word_frequency_path='SUBTLEX_frequency.xlsx',
    ppdb_path='datasets/ppdb.db',
)

analyzer = ReadabilityAnalyzer(engine, target_grade=8.0)

# Score text without simplifying
scores = analyzer.score_text("The physician administered the preliminary examination.")
print(f"FK Grade: {scores.flesch_kincaid_grade}")  # -> ~14.2
print(f"Level:    {scores.interpretation}")         # -> Graduate / Professional

# Simplify to Grade 8 with readability analysis
result = analyzer.simplify_to_target(
    "The physician administered the preliminary examination and "
    "deliberated on the contentious legislative proposal."
)
print(result.summary())
```

```bash
# CLI usage
python readability_analyzer.py "The physician administered the examination." --target-grade 8
python readability_analyzer.py "Text." --analyze-only
python readability_analyzer.py "Text." --json
```

### Command-Line Evaluation

```bash
# Smoke test (core functionality)
py -3.12 test_engine.py

# Full accuracy test suite (45 items)
py -3.12 test_accuracy.py

# Evaluate on BenchLS dataset
py -3.12 LSBert2.py \
    --eval_dir datasets/BenchLS.txt \
    --bert_model bert-base-uncased \
    --ppdb datasets/ppdb.db \
    --word_frequency SUBTLEX_frequency.xlsx \
    --output_SR_file results/benchls_output.txt \
    --do_eval --no_cuda --do_lower_case

# Windows batch runner
run_LSBert2.bat
```

---

## Evaluation Results

### Test Suite Results (Lite Mode, 2026-03-28)

| Category | Passed | Failed | Rate |
|----------|--------|--------|------|
| Semantic Preservation | 7 | 0 | 100% |
| WordNet Semantic Guard | 4 | 0 | 100% |
| Morphological Preservation | 3 | 2 | 60% |
| Edge Cases (short/long/symbols) | 6 | 0 | 100% |
| Auto-Simplification | 3 | 0 | 100% |
| BenchLS Sample (10 items) | 6 | 4 | 60% |
| **Total** | **39** | **6** | **86.7%** |

### Evaluation Metrics Defined

**Substitution Selection** (candidate list vs. gold labels):
- **Potential**: fraction of instances where ≥1 candidate matches gold (ceiling measure)
- **Precision**: |candidates ∩ gold| / |candidates|
- **Recall**: |candidates ∩ gold| / |gold|
- **F-score**: harmonic mean

**Full Pipeline** (selected replacement vs. gold):
- **Pipeline Accuracy**: (changed AND in gold) / total — primary metric
- **Pipeline Precision**: (unchanged OR in gold) / total
- **Changed Proportion**: fraction of target words actually replaced

### Datasets

| Dataset | Items | Source |
|---------|-------|--------|
| BenchLS | 929 | Wikipedia + simplification annotations |
| NNSeval | 239 | Wikipedia, non-native speaker study |
| lex.mturk | 501 | Amazon Mechanical Turk crowd annotations |

---

## Known Limitations

| Issue | Impact | Notes |
|-------|--------|-------|
| Irregular past-tense morphology | 2 of 45 failures | "administered" → "took" instead of -ed form |
| CWI model requires TF1.x | Complex word detection uses fallback | Frequency heuristic used instead of learned model |
| Speed: 4–10s per word on CPU | Not suitable for large-scale documents | GPU or distilled BERT needed for real-time use |
| Sentence length: max 128 tokens | Very long sentences are truncated | BERT hard limit |
| Semantic drift without FastText | ~60% vs ~70–75% BenchLS accuracy | Enable full mode for better semantic similarity |

See [ANALYSIS.md](ANALYSIS.md) for root cause analysis and fix directions for each limitation.

---

## REST API Reference

Start the web app with `python app.py --skip-fasttext`, then:

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| GET | `/api/status` | — | Engine health + loaded components |
| POST | `/api/simplify` | `{"sentence": "...", "target_word": "..."}` | Replacement + candidates + scores |
| POST | `/api/identify` | `{"sentence": "..."}` | List of complex words with complexity scores |
| POST | `/api/simplify_auto` | `{"sentence": "..."}` | Auto-identified + simplified sentence |
| POST | `/api/readability_simplify` | `{"text": "...", "target_grade": 8.0}` | Iteratively simplified text to target grade level |

Example:
```bash
curl -X POST http://localhost:5000/api/simplify \
     -H "Content-Type: application/json" \
     -d '{"sentence": "The physician administered the examination.", "target_word": "administered"}'
```

---

## Project Structure

```
Lexical Simplification/
├── app.py                       # Flask web application
├── requirements.txt             # Python dependencies
├── start_webapp.bat             # Windows launcher (batch)
├── start_webapp.ps1             # Windows launcher (PowerShell)
├── templates/
│   └── index.html               # Web UI (single-page app)
├── src/
│   ├── simplify_engine.py       # Main SimplificationEngine class
│   ├── readability_analyzer.py  # [NEW] Readability-aware simplification module
│   ├── LSBert2.py               # Original evaluation script
│   ├── evaluate.py              # Automated benchmark pipeline
│   ├── test_engine.py           # Smoke tests
│   ├── test_accuracy.py         # Accuracy test suite
│   ├── complex_word.py          # Complex word identification (CWI)
│   ├── labeler.py               # TF sequence labeling model (CWI)
│   ├── PPDB/ppdb.py             # PPDB wrapper (SQLite + memory modes)
│   ├── pytorch_pretrained_bert/ # BERT model implementation
│   ├── datasets/
│   │   ├── BenchLS.txt          # 929-item benchmark
│   │   ├── NNSeval.txt          # 239-item benchmark
│   │   ├── lex.mturk.txt        # 501-item MTurk dataset
│   │   └── ppdb.db              # 299 MB SQLite PPDB
│   └── results/                 # Test output files
├── ANALYSIS.md                  # Performance analysis
├── INNOVATION.md                # Innovation documentation
└── README.md
```

---

## Analysis and Innovation

This repository includes two additional documents beyond the original paper:

- **[ANALYSIS.md](ANALYSIS.md)** — Deep architectural analysis, exact ranking weights, measured performance on all test suites, failure mode root causes, and comparison with baselines.

- **[INNOVATION.md](INNOVATION.md)** — Documents the readability-aware extension: motivation, design, concrete examples, and limitations. Implemented in `readability_analyzer.py`.

The core innovation is a **feedback-driven simplification loop** that measures text readability (Flesch-Kincaid Grade, SMOG, ARI, Coleman-Liau, Dale-Chall, Flesch Reading Ease) before and after each pass, targeting a user-specified grade level. This addresses a gap in the original evaluation framework: gold-label accuracy measures whether a substitution is plausible, but not whether the text is actually measurably easier to read.

---

## Implementation Guide

### System Requirements

| Requirement | Minimum (Lite Mode) | Recommended (Full Mode) |
|------------|---------------------|------------------------|
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 10 GB |
| Disk | 1.5 GB | 7 GB |
| OS | Windows 10 / Ubuntu 20.04 / macOS 11 | Same |
| GPU | Not required | Optional (CUDA 11+ for speedup) |

---

### Step-by-Step Installation

**Step 1 — Clone or download the project**
```bash
git clone <repo-url> "Lexical Simplification"
cd "Lexical Simplification"
```

**Step 2 — Create a virtual environment (recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**Step 3 — Install Python dependencies**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install flask nltk openpyxl scipy scikit-learn textstat numpy
```

> For GPU support: replace the torch install line with the CUDA-specific index URL from [pytorch.org](https://pytorch.org/get-started/locally/).

**Step 4 — Download required NLTK data**

Run this once in a Python shell:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**Step 5 — Verify data files are present**
```
src/
├── SUBTLEX_frequency.xlsx      ← must exist (10.8 MB)
├── gpu_attention.model         ← must exist (62.9 MB, CWI model)
└── datasets/
    ├── ppdb.db                 ← must exist (299 MB SQLite)
    ├── BenchLS.txt             ← must exist (benchmark)
    └── NNSeval.txt             ← must exist (benchmark)
```

**Step 6 — Run the smoke test to verify installation**
```bash
cd src
py -3 test_engine.py
```
Expected: `20/21 tests passed` (one known morphology edge-case failure is expected).

**Step 7 — Start the web application**
```bash
# From the project root directory
python app.py --skip-fasttext
```
Open `http://localhost:5000`. Engine loads on first request (~17 seconds).

---

### SimplificationEngine — Parameter Reference

```python
SimplificationEngine(
    bert_model='bert-base-uncased',      # HuggingFace model name or local path
    word_embeddings_path=None,           # Path to FastText .vec file (full mode only)
    word_frequency_path=None,            # Path to SUBTLEX_frequency.xlsx
    ppdb_path=None,                      # Path to ppdb.db (SQLite)
    max_seq_length=128,                  # Max BERT tokens per sentence pair
    num_selections=10,                   # Top-N candidates to keep after BERT
    use_cuda=False,                      # Set True if CUDA GPU is available
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `bert_model` | `'bert-base-uncased'` | Downloaded automatically ~440 MB on first run |
| `word_embeddings_path` | `None` | Omit for lite mode; provide path for FastText full mode |
| `word_frequency_path` | `None` | Required; point to `src/SUBTLEX_frequency.xlsx` |
| `ppdb_path` | `None` | Required; point to `src/datasets/ppdb.db` |
| `max_seq_length` | `128` | Hard BERT limit; long sentences are truncated |
| `num_selections` | `10` | Top-N BERT candidates after subword/same-stem filtering |
| `use_cuda` | `False` | Set `True` to use GPU; reduces per-word time from 4–10s to <1s |

---

### ReadabilityAnalyzer — Parameter Reference

```python
ReadabilityAnalyzer(
    engine,                          # A loaded SimplificationEngine instance
    target_grade=8.0,                # US grade level to target (default: general public)
    max_passes=3,                    # Max simplification iterations per sentence
    convergence_threshold=0.2,       # Minimum composite grade change to continue
    min_improvement_per_pass=0.1,    # Stop iterating if improvement falls below this
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `target_grade` | `8.0` | 6 = elementary, 8 = general public, 12 = high school, 16 = college |
| `max_passes` | `3` | Prevents infinite loops; 3 passes is usually sufficient |
| `convergence_threshold` | `0.2` | If grade doesn't improve by 0.2, stop the loop |
| `min_improvement_per_pass` | `0.1` | Per-pass minimum; plateau detection |

---

### Adjusting Ranking Weights

To experiment with different weighting strategies, edit the weights directly in [src/simplify_engine.py](src/simplify_engine.py):

```python
# Around line 50–60 in SimplificationEngine.__init__
self.W_BERT = 1.0    # BERT prediction rank
self.W_SIS  = 1.5    # Semantic similarity (HIGHEST — meaning preservation)
self.W_CNT  = 0.8    # Word frequency
self.W_LM   = 1.2    # LM fluency (pseudo-perplexity)
self.W_PPDB = 0.5    # PPDB quality score (LOWEST — bonus signal)
```

Increasing `W_SIS` further biases toward synonym precision; increasing `W_CNT` biases toward more common (but potentially less precise) substitutions.

---

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: ppdb.db` | Incorrect working directory | Run from the project root, not from `src/` |
| `FileNotFoundError: SUBTLEX_frequency.xlsx` | Missing file | Confirm file is at `src/SUBTLEX_frequency.xlsx` |
| BERT download on first run | Expected — 440 MB auto-download | Wait; cached in `~/.cache/huggingface/` after first run |
| `WARNING: TF1 CWI model failed to load` | TensorFlow not installed or TF2 only | Normal — frequency heuristic fallback is used automatically |
| Engine startup takes >30s | Large BERT model loading on CPU | Normal for first load; subsequent loads use OS file cache |
| `CUDA out of memory` | GPU VRAM < 2 GB | Set `use_cuda=False` or use a smaller BERT variant |
| `nltk.exceptions.LookupError` | NLTK data not downloaded | Run the `nltk.download()` commands in Step 4 above |
| Per-word time is 4–10 seconds | CPU-only inference | Use `use_cuda=True` with a CUDA GPU for <1s per word |
| BenchLS accuracy is ~60% | Lite mode (no FastText) | Download FastText and run without `--skip-fasttext` for 70–75% |


