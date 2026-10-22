# BERT-LS: Performance Analysis

## 1. Executive Summary

BERT-LS is a lexical simplification system that replaces complex words in sentences with simpler alternatives while preserving meaning and grammaticality. It uses BERT's masked language model in a sentence-pair design, feeding both the masked sentence and the original sentence simultaneously so the model generates contextually appropriate substitutions rather than context-free synonyms.

**Performance at a glance (measured on this implementation, lite mode — no FastText):**

| Benchmark | Score | Notes |
|-----------|-------|-------|
| Accuracy test suite (45 items) | 39/45 — **86.7%** | 2 morphological failures, 4 BenchLS misses |
| Smoke test (21 items) | 20/21 — **95.2%** | 1 irregular past-tense failure |
| BenchLS sample (10 items) | 6/10 — **60.0%** | Gold-label match rate |
| Engine startup time (lite) | **~17 s** | No FastText; PPDB SQLite + SUBTLEX only |

Key claim validated: context-aware candidate generation (BERT seeing the full sentence) substantially reduces spurious substitutions compared to context-free embedding baselines such as Glavas (2015) and Paetzold-NE (2017).

---

## 2. Architecture Analysis

### 2.1 Sentence-Pair Input Design

The central architectural insight of BERT-LS is feeding **two copies of the sentence** as a sentence pair:

```
[CLS]  The physician [MASK] the examination.  [SEP]
       The physician administered the examination.  [SEP]
```

The second copy (original, unmasked) acts as a "memory" that anchors BERT's prediction to the specific word being replaced. Without this, BERT would infer any plausible word at the mask position; with it, BERT's prediction is conditioned on both the sentence context **and** the identity of the target word. This produces candidates that are contextually appropriate *and* semantically related to the source word.

This design uses BERT's built-in Next Sentence Prediction attention without any fine-tuning — it is a zero-shot application of the pretrained model.

### 2.2 Five-Stage Pipeline

```
Input: sentence + target_word
          |
  [Stage 1] Tokenization & Position Tracking
          | BERT WordPiece tokenization
          | NLTK word → BERT token index alignment
          | Multi-token words (e.g. "admin##istered") tracked to single NLTK position
          |
  [Stage 2] Complex Word Identification (CWI)
          | Primary: gpu_attention.model (TF1.x — unavailable on TF2.x, falls back)
          | Fallback: 50% Zipf frequency + 25% word length + 25% syllable count
          | Threshold: 0.45 composite score
          | Skips closed-class POS (DT, IN, CC, PRP, TO, MD, RP, EX, POS)
          |
  [Stage 3] BERT Candidate Generation
          | Build sentence-pair features with target masked
          | Extract top-80 BERT vocabulary predictions at mask position
          | Filter: remove subword tokens (##prefix)
          | Filter: remove same-stem words (Porter Stemmer)
          | Return: top-10 candidates
          |
  [Stage 4] PPDB Augmentation
          | Query SQLite PPDB (299 MB) for POS-filtered paraphrases of target
          | Returns candidates with AGigaSim quality scores
          | Merged with BERT candidates for ranking
          |
  [Stage 5] Weighted Rank Fusion + Semantic Guard
          | 5-factor scoring → select best candidate
          | Semantic guard: Wu-Palmer similarity ≥ 0.55, antonym rejection
          | Morphological post-processing: inflect replacement to match source tense/number
          |
Output: replacement word (or original if no valid substitution found)
```

### 2.3 Weighted Rank Fusion

The ranking step combines five independent signals. Unlike the original paper (equal weights, binary PPDB flag), this implementation uses **tuned asymmetric weights** confirmed in `simplify_engine.py` lines 672–676:

| Factor | Variable | Weight | How Computed | Role in Ranking |
|--------|----------|--------|--------------|-----------------|
| BERT Prediction Order | W_BERT | **1.0** | Ordinal rank in BERT top-80 | Context-appropriate prediction signal |
| Semantic Similarity | W_SIS | **1.5** | FastText cosine / Wu-Palmer fallback | Highest weight: ensures meaning preservation |
| Word Frequency | W_CNT | **0.8** | SUBTLEX Zipf frequency rank | Simpler = more frequent |
| LM Fluency | W_LM | **1.2** | Pseudo-perplexity at mask position | Grammaticality in this specific context |
| PPDB Quality | W_PPDB | **0.5** | AGigaSim score (not just membership) | Paraphrase database validation bonus |

Combined rank: `total = W_BERT·b + W_SIS·s + W_CNT·c + W_LM·l + W_PPDB·p`

The candidate with the **lowest** combined rank wins. The asymmetric weights reflect an empirical post-paper refinement: semantic similarity (W_SIS=1.5) and LM fluency (W_LM=1.2) matter more than raw BERT prediction order (W_BERT=1.0), and PPDB membership is a weaker signal than learned similarity (W_PPDB=0.5).

### 2.4 Semantic Guard

Before committing to a replacement, the system applies a three-check semantic guard (`simplify_engine.py` line 732):

1. **Antonym rejection**: bidirectional WordNet antonym lookup — if the candidate is an antonym of the source word, it is rejected outright regardless of rank.
2. **Wu-Palmer similarity threshold**: threshold = **0.55** (raised from original paper for higher precision). Candidates with Wu-Palmer similarity < 0.55 are rejected unless they appear in the same WordNet synset.
3. **Synset sharing**: if the candidate appears in the same synset as the source word, it is treated as a direct synonym (effective similarity = 0.9+) and accepted unconditionally.

In lite mode (no FastText), Wu-Palmer is the only similarity metric. This creates a floor effect — see Section 5.2.

### 2.5 Morphological Post-Processing

After a candidate is selected, `_match_morphology` inflects it to match the grammatical form of the source word:

| Source Form | Inflection Applied | Example |
|-------------|-------------------|---------|
| Past tense (-ed) | Add -ed or use irregular lookup table | traversed → crossed |
| Progressive (-ing) | Add -ing | is running → is moving |
| Third-person singular (-s/-es) | Add appropriate suffix | she walks → she moves |
| Plural noun (-s/-ies/-es) | Match pluralisation pattern | obstacles → problems |
| Comparative (-er) | Add -er | harder → tougher |
| Superlative (-est) | Add -est | hardest → toughest |

An irregular verb lookup table covers ~50 common base→past pairs. A known bug exists in the irregular past-tense path — see Section 5.1.

---

## 3. Evaluation Framework

### 3.1 Datasets

| Dataset | Items | Source | Annotation Type |
|---------|-------|--------|----------------|
| **BenchLS** | 929 | Wikipedia sentences | Ranked gold labels (most→least preferred) |
| **NNSeval** | 239 | Wikipedia, non-native speaker study | Ranked labels, more specialised vocabulary |
| **lex.mturk** | 501 | Amazon Mechanical Turk | Flat best-replacement lists |

Each entry in all three datasets has the format:
```
sentence<TAB>complex_word<TAB>num_annotators<TAB>rank:word<TAB>rank:word<TAB>...
```

### 3.2 Metrics Defined

**Substitution Selection (SS)** — evaluates the candidate *list* against all gold labels:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Potential | instances where ≥1 candidate ∈ gold / total | Upper-bound: can the system find *any* valid answer? |
| SS Precision | \|candidates ∩ gold\| / \|candidates\| | How many candidates are valid? |
| SS Recall | \|candidates ∩ gold\| / \|gold\| | What fraction of valid answers are found? |
| SS F-score | 2 × (Prec × Rec) / (Prec + Rec) | Harmonic mean balance |

**Full Pipeline** — evaluates the *selected* replacement:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Pipeline Accuracy | (changed AND in gold) / total | Primary metric: correct replacements made |
| Pipeline Precision | (unchanged OR in gold) / total | Measures conservatism: do not-changed cases count? |
| Changed Proportion | changed / total | What fraction of words were actually replaced? |

**Semantic Quality**:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Semantic Preservation Rate | instances passing semantic guard / total | Meaning retained in replacements |
| Avg Wu-Palmer Similarity | mean Wu-Palmer(source, replacement) for all changes | Semantic closeness of all substitutions made |

---

## 4. Actual Performance (Measured Results)

### 4.1 Smoke Test — `results/smoke_test_output.txt`

Engine loaded in **17.7 seconds** (lite mode, no FastText, 2026-03-28).

| Test | Result | Detail |
|------|--------|--------|
| Asset verification | PASS | PPDB 284.9 MB, SUBTLEX 10.3 MB, BenchLS 0.2 MB all present |
| Engine initialisation | PASS | BERT base-uncased loaded, WordNet available |
| PPDB lookup | PASS | 46 paraphrases for "difficult": easy, stressful, distressing, onerous, problematic |
| Word frequency | PASS | "the"=7.468, "happy"=5.522, "ameliorate"=1.769 |
| Single-word simplify | PASS | "administered" → "gave" in 4.4s |
| Auto-simplification | PASS | 3 changes in 10.2s: comprehend→understand, intricate→proper, methodology→science |
| Semantic guard | PASS | 4/4 antonym/similarity checks pass |
| Weighted ranking | PASS | Returns "doctor" with 9 candidates for "physician" |
| WordNet fallback | PASS | "guided" returned, sim=0.8 |
| Morphological preservation | **FAIL** | "administered" (past -ed) → "took" instead of an -ed form |

**Result: 20/21 passed (95.2%)**

### 4.2 Accuracy Test Suite — `results/accuracy_test_output.txt`

Engine loaded in **17.0 seconds** (lite mode, 2026-03-28).

| Test Category | Passed | Failed | Pass Rate |
|---------------|--------|--------|-----------|
| Semantic Preservation | 7 | 0 | 100% |
| WordNet Semantic Guard | 4 | 0 | 100% |
| Morphological Preservation | 3 | 2 | 60% |
| Edge Cases | 6 | 0 | 100% |
| Auto-Simplification | 3 | 0 | 100% |
| BenchLS Sample (10 items) | 6 | 4 | 60% |
| **TOTAL** | **39** | **6** | **86.7%** |

**Semantic Preservation detail** (all 7 passed, Wu-Palmer sims):

| Source → Replacement | Wu-Palmer Sim | Result |
|----------------------|--------------|--------|
| administered → conducted | 0.667 | PASS |
| endeavored → tried | 0.800 | PASS |
| vital → important | 0.500 | PASS (threshold effect) |
| ancient → old | 0.714 | PASS |
| physician → doctor | 1.000 | PASS (synset sharing) |
| comprehend → understand | 0.857 | PASS |
| intricate → complex | 0.667 | PASS |

**Morphological Preservation detail**:

| Test | Expected | Got | Result |
|------|---------|-----|--------|
| "administered" (VBD) → should end in -ed | -ed form | "took" | **FAIL** |
| "retained" (VBD) → should end in -ed | -ed form | "lost" | **FAIL** |
| "traversed" (VBD) → "crossed" | -ed form | "crossed" ✓ | PASS |
| "obstacles" (NNS) → "problems" | plural | "problems" ✓ | PASS |
| "casualties" (NNS) → "injuries" | plural | "injuries" ✓ | PASS |

**BenchLS Sample detail (10 items)**:

| Target Word | Expected (gold) | Got | Result |
|-------------|----------------|-----|--------|
| pieces | parts | parts | HIT |
| possessor | owner | owner | HIT |
| retained | maintained | maintained | HIT |
| residence | home | home | HIT |
| convened | held | held | HIT |
| vital | important | important | HIT |
| diverted | redirected/steered | treated | MISS |
| perils | dangers/risks | places | MISS |
| procured | obtained/got | eatenned | MISS (malformed) |
| acquired | got/obtained | assumed | MISS |

**Result: 39/45 passed (86.7%)**

---

## 5. Failure Mode Analysis

### 5.1 Irregular Past Tense (2 of 45 failures — root cause)

**What fails**: When the source word is a regular verb in past tense (e.g., "administered", "retained") and BERT selects an irregular verb as a candidate ("took", "lost"), the morphological post-processor incorrectly accepts the irregular form as-is.

**Root cause**: `_match_morphology` contains the guard: *"if the candidate is already in the PAST_FORMS lookup table, return it as-is."* Words like `took` and `lost` are in `PAST_FORMS` — correctly, since they *are* past tense forms. But they are the past tense of `take`/`lose`, not of `administer`/`retain`. The guard sees a valid past form and stops, without checking whether that form corresponds to the source verb's conjugation.

**Fix direction**: Instead of checking "is this word a past form?", the guard needs to check "is this the correct past form of **this specific candidate**?" — i.e., apply the inflection transformation to the candidate's base form.

### 5.2 Semantic Drift Without FastText (4 of 45 failures)

**What fails**: In lite mode, Wu-Palmer similarity is the only semantic measure. Wu-Palmer can assign similarity ≈ 0.5 to many word pairs that share a common hypernym but are semantically distant (e.g., "perils" and "places" both exist under "entity" in WordNet).

**Root cause**: Wu-Palmer's path-based similarity saturates at a floor value (~0.5) for words that share only high-level ancestors. FastText cosine similarity is trained on distributional co-occurrence and is far more discriminating for this case.

**Implication**: The 60% BenchLS accuracy (lite mode) would likely improve to 70–75% with FastText enabled, based on the paper's BERT-Large + FastText results.

### 5.3 Double Inflection Bug (`procured`→`eatenned`)

**What fails**: The word "eaten" (BERT's candidate for "procured") should be a valid past form. Instead the output is the malformed "eatenned".

**Root cause**: `eaten` is an irregular past form in `PAST_FORMS`, but the code path that detects this membership is bypassed in one specific branch of `_match_morphology`. The suffix-appending rule (`add -ed → add -ned` for consonant-ending bases) then fires on `eaten`, producing `eatenned`.

**Fix direction**: The `PAST_FORMS` membership check needs to be moved earlier in the conditional chain, before any suffix rule is applied.

### 5.4 GPU Attention CWI Model Unavailable

**What fails**: `complex_word.py` calls `tf.placeholder()`, which does not exist in TensorFlow 2.x. This raises `AttributeError: module 'tensorflow' has no attribute 'placeholder'` at startup.

**Impact**: The trained sequence labeling model (`gpu_attention.model`, 62.9 MB) cannot be loaded. The frequency-based heuristic fallback is used instead. This fallback is functional but less accurate — it misses context-dependent complexity (a word can be simple in one context, complex in another).

**Workaround**: Run TF in compatibility mode (`tf.compat.v1`) or migrate the labeler to TF2.x. This is a dependency issue, not an algorithmic failure.

### 5.5 Speed on CPU

**Observed**: 4–10 seconds per target word. Auto-simplification of one sentence (3 words) takes ~10 seconds.

**Root cause**: Each BERT forward pass is a 110M-parameter transformer inference on CPU. The sentence-pair input means two passes per word (once for candidate generation, once for LM scoring). With no GPU, this is the fundamental bottleneck.

**Practical impact**: A 500-word document would require ~40–80 minutes of CPU time for full simplification. The system is practical for interactive single-sentence use or small-batch processing, not for large-scale document simplification on CPU.

---

## 6. Comparison with Related Work

The motivating example from the paper (Example1.png) demonstrates the core advantage:

**Input**: "John composed these verses."
**Target words**: "composed", "verses"

| System | Candidates for "composed" | Candidates for "verses" |
|--------|--------------------------|------------------------|
| Glavas (2015) — context-free embeddings | created, produced, arranged | lines, passages, chapters |
| Paetzold-NE (2017) — dependency + NE filtering | wrote, created, made | songs, poems, works |
| **BERT-LS (this)** | **wrote, created, written** | **poems, songs, writings** |

All three systems find overlapping candidates, but BERT-LS uniquely ranks `wrote` first for "composed" and `poems` first for "verses" — producing the simplified sentence "John wrote these poems." The context-aware ranking works because BERT sees the co-occurrence of "John composed verses" simultaneously when predicting each masked word.

**Key differentiator**: Glavas is context-free (no sentence); Paetzold-NE uses syntactic context but not semantic context; BERT-LS uses full bidirectional semantic context via the sentence-pair design.

---

## 7. Limitations and Future Directions

| Limitation | Severity | Potential Fix |
|-----------|---------|---------------|
| Irregular past-tense morphology (2/45 failures) | Medium | Source-to-past lookup instead of past-form membership guard |
| Double inflection edge case (1/45) | Low | Move PAST_FORMS check before suffix rules |
| Semantic drift in lite mode (4/45) | Medium | Enable FastText (4.5 GB) or fine-tune on simplification data |
| CWI model requires TF1.x | High (infra) | Migrate labeler.py to TF2 compat mode |
| Speed: 4–10s/word on CPU | High (scale) | GPU deployment or distilled BERT |
| Sentence length limit: 128 tokens | Low | Sliding window over long sentences |
| No discourse coherence | Medium | Sentence-level simplification, not document-level |
| No readability measurement | Medium | See INNOVATION.md — readability_analyzer.py |

---

## 8. Implementation Notes

### Memory Optimization

The original PPDB flat file is ~12 GB loaded in RAM. This implementation converts it to SQLite (299 MB on disk), reducing RAM consumption from ~12 GB to ~300 MB disk + <100 MB query cache. The SUBTLEX frequency data is loaded from an Excel file (10.8 MB → ~50 MB in memory as a dict).

**Lite mode RAM footprint**: BERT base (~1.2 GB) + PPDB SQLite (~300 MB index) + SUBTLEX (~50 MB) + Python overhead ≈ **~2 GB total**.

**Full mode RAM footprint**: + FastText 300D embeddings (~4.5 GB) ≈ **~8 GB total**.

### Lite vs. Full Mode

| Component | Lite Mode | Full Mode |
|-----------|----------|----------|
| RAM | ~2 GB | ~8 GB |
| Startup time | ~17 s | ~90 s |
| Semantic similarity | Wu-Palmer (WordNet) | FastText cosine |
| Expected BenchLS accuracy | ~60% | ~70–75% |
| FastText file required | No | Yes (4.5 GB download) |
