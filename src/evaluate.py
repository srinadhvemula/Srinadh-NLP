"""
BERT-LS Automated Evaluation Pipeline.

Evaluates the simplification engine against standard lexical simplification
benchmarks: BenchLS, NNSeval, and lex.mturk.

Metrics computed:
  - Substitution Selection: Potential, Precision, Recall, F-score
  - Full Pipeline: Accuracy, Changed Proportion, Precision
  - Semantic: Preservation Rate, Average WordNet Similarity

Usage:
    py -3.12 evaluate.py                          # Evaluate on BenchLS (default)
    py -3.12 evaluate.py --dataset NNSeval.txt    # Evaluate on NNSeval
    py -3.12 evaluate.py --dataset all            # Evaluate on all datasets
    py -3.12 evaluate.py --max-samples 50         # Quick test with 50 samples
"""

import os
import sys
import json
import time
import argparse
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Dataset readers
# ---------------------------------------------------------------------------

def read_benchls(path):
    """
    Read BenchLS-format dataset.
    Format: sentence<TAB>complex_word<TAB>num_annotators<TAB>rank:word<TAB>...
    Returns list of (sentence, complex_word, gold_labels).
    """
    items = []
    with open(path, 'r', encoding='ISO-8859-1') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue

            sentence = parts[0].strip()
            complex_word = parts[1].strip()

            # Parse gold labels: "rank:word" pairs
            gold_labels = []
            for entry in parts[3:]:
                entry = entry.strip()
                if not entry:
                    continue
                if ':' in entry:
                    _, word = entry.split(':', 1)
                    word = word.strip()
                    if word and word not in gold_labels:
                        gold_labels.append(word)
                else:
                    word = entry.strip()
                    if word and word not in gold_labels:
                        gold_labels.append(word)

            if sentence and complex_word and gold_labels:
                items.append((sentence, complex_word, gold_labels))

    return items


def read_lex_mturk(path):
    """
    Read lex.mturk dataset (same format as BenchLS).
    """
    return read_benchls(path)


def read_nnseval(path):
    """
    Read NNSeval dataset (same tab-separated format).
    """
    return read_benchls(path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ss_metrics(all_candidates, all_gold_labels):
    """
    Substitution Selection metrics: Potential, Precision, Recall, F-score.

    all_candidates: list of lists of candidate words
    all_gold_labels: list of lists of gold label words
    """
    assert len(all_candidates) == len(all_gold_labels)

    instances = len(all_candidates)
    if instances == 0:
        return {'potential': 0, 'precision': 0, 'recall': 0, 'f_score': 0}

    potential_count = 0
    precision_num = 0
    precision_den = 0
    recall_num = 0
    recall_den = 0

    for candidates, golds in zip(all_candidates, all_gold_labels):
        candidates_lower = [c.lower() for c in candidates]
        golds_lower = [g.lower() for g in golds]

        common = set(candidates_lower).intersection(set(golds_lower))

        if len(common) >= 1:
            potential_count += 1

        precision_num += len(common)
        precision_den += len(candidates_lower)
        recall_num += len(common)
        recall_den += len(golds_lower)

    potential = potential_count / instances
    precision = precision_num / precision_den if precision_den > 0 else 0
    recall = recall_num / recall_den if recall_den > 0 else 0
    f_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        'potential': round(potential, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f_score': round(f_score, 4),
    }


def compute_pipeline_metrics(substitutions, source_words, all_gold_labels):
    """
    Full pipeline metrics: Accuracy, Precision, Changed Proportion.

    substitutions: list of final replacement words
    source_words: list of original complex words
    all_gold_labels: list of lists of gold labels
    """
    instances = len(substitutions)
    if instances == 0:
        return {'pipeline_precision': 0, 'pipeline_accuracy': 0, 'changed_proportion': 0}

    precision = 0
    accuracy = 0
    changed = 0

    for sub, source, golds in zip(substitutions, source_words, all_gold_labels):
        sub_l = sub.lower()
        source_l = source.lower()
        golds_l = [g.lower() for g in golds]

        # Precision: unchanged OR in gold labels
        if sub_l == source_l or sub_l in golds_l:
            precision += 1

        # Accuracy: changed AND in gold labels
        if sub_l != source_l and sub_l in golds_l:
            accuracy += 1

        # Changed proportion
        if sub_l != source_l:
            changed += 1

    return {
        'pipeline_precision': round(precision / instances, 4),
        'pipeline_accuracy': round(accuracy / instances, 4),
        'changed_proportion': round(changed / instances, 4),
    }


def compute_semantic_metrics(results):
    """
    Semantic preservation metrics from engine results.
    """
    total = len(results)
    if total == 0:
        return {'semantic_preservation_rate': 0, 'avg_semantic_similarity': 0}

    preserved = sum(1 for r in results if r.get('sense_preserved', True))
    sims = [r.get('semantic_similarity', 1.0) for r in results if r.get('changed', False)]

    avg_sim = sum(sims) / len(sims) if sims else 1.0

    return {
        'semantic_preservation_rate': round(preserved / total, 4),
        'avg_semantic_similarity': round(avg_sim, 4),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(engine, dataset_items, dataset_name, max_samples=None):
    """
    Evaluate the engine on a dataset.
    Returns a dict with all metrics and per-instance results.
    """
    if max_samples and max_samples < len(dataset_items):
        dataset_items = dataset_items[:max_samples]

    print(f"\n{'='*60}")
    print(f"  Evaluating on: {dataset_name}")
    print(f"  Samples: {len(dataset_items)}")
    print(f"{'='*60}\n")

    all_candidates = []
    all_substitutions = []
    all_source_words = []
    all_gold_labels = []
    all_results = []
    errors = 0
    total_time = 0

    for i, (sentence, complex_word, gold_labels) in enumerate(dataset_items):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(dataset_items)}] Processing: '{complex_word}' in: {sentence[:60]}...")

        try:
            t0 = time.time()
            result = engine.simplify_word(sentence, complex_word)
            elapsed = time.time() - t0
            total_time += elapsed

            candidates = result.get('candidates', [])
            replacement = result.get('replacement', complex_word)

            all_candidates.append(candidates)
            all_substitutions.append(replacement)
            all_source_words.append(complex_word.lower())
            all_gold_labels.append(gold_labels)
            all_results.append(result)

        except Exception as e:
            errors += 1
            all_candidates.append([])
            all_substitutions.append(complex_word.lower())
            all_source_words.append(complex_word.lower())
            all_gold_labels.append(gold_labels)
            all_results.append({'error': str(e), 'changed': False})
            if errors <= 5:
                print(f"    [ERROR] {e}")

    # Compute metrics
    ss_metrics = compute_ss_metrics(all_candidates, all_gold_labels)
    pipe_metrics = compute_pipeline_metrics(all_substitutions, all_source_words, all_gold_labels)
    sem_metrics = compute_semantic_metrics(all_results)

    avg_time = total_time / len(dataset_items) if dataset_items else 0

    combined = OrderedDict()
    combined['dataset'] = dataset_name
    combined['num_samples'] = len(dataset_items)
    combined['errors'] = errors
    combined['avg_time_per_word_sec'] = round(avg_time, 2)
    combined['total_time_sec'] = round(total_time, 1)
    combined.update(ss_metrics)
    combined.update(pipe_metrics)
    combined.update(sem_metrics)

    # Print results
    print(f"\n{'─'*50}")
    print(f"  Results for: {dataset_name}")
    print(f"{'─'*50}")
    print(f"  Samples:                {combined['num_samples']}")
    print(f"  Errors:                 {combined['errors']}")
    print(f"  Avg time/word:          {combined['avg_time_per_word_sec']}s")
    print()
    print(f"  === Substitution Selection ===")
    print(f"  Potential:              {combined['potential']:.4f}")
    print(f"  Precision:              {combined['precision']:.4f}")
    print(f"  Recall:                 {combined['recall']:.4f}")
    print(f"  F-score:                {combined['f_score']:.4f}")
    print()
    print(f"  === Full Pipeline ===")
    print(f"  Pipeline Precision:     {combined['pipeline_precision']:.4f}")
    print(f"  Pipeline Accuracy:      {combined['pipeline_accuracy']:.4f}")
    print(f"  Changed Proportion:     {combined['changed_proportion']:.4f}")
    print()
    print(f"  === Semantic Quality ===")
    print(f"  Preservation Rate:      {combined['semantic_preservation_rate']:.4f}")
    print(f"  Avg Similarity (WN):    {combined['avg_semantic_similarity']:.4f}")
    print(f"{'─'*50}\n")

    # Collect sample results for report
    sample_results = []
    for i, (sentence, cw, golds) in enumerate(dataset_items[:20]):
        r = all_results[i]
        sample_results.append({
            'sentence': sentence[:80],
            'complex_word': cw,
            'replacement': r.get('replacement', cw),
            'changed': r.get('changed', False),
            'candidates': r.get('candidates', [])[:5],
            'gold_labels': golds[:5],
            'sense_preserved': r.get('sense_preserved', True),
            'semantic_similarity': r.get('semantic_similarity', 1.0),
        })

    combined['sample_results'] = sample_results

    return combined


def generate_report(all_results, output_dir):
    """Generate markdown report and JSON data."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON output
    json_path = os.path.join(output_dir, 'benchmark_results.json')
    # Remove sample_results from JSON for readability (save separately)
    json_data = []
    for r in all_results:
        r_copy = dict(r)
        samples = r_copy.pop('sample_results', [])
        json_data.append(r_copy)

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Markdown report
    md_path = os.path.join(output_dir, 'benchmark_report.md')
    with open(md_path, 'w') as f:
        f.write("# BERT-LS Benchmark Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Metric | " + " | ".join(r['dataset'] for r in all_results) + " |\n")
        f.write("|--------|" + "|".join("-------" for _ in all_results) + "|\n")

        metrics = [
            ('Samples', 'num_samples'),
            ('Errors', 'errors'),
            ('Avg Time (s)', 'avg_time_per_word_sec'),
            ('Potential', 'potential'),
            ('Precision (SS)', 'precision'),
            ('Recall (SS)', 'recall'),
            ('F-score (SS)', 'f_score'),
            ('Precision (Pipeline)', 'pipeline_precision'),
            ('Accuracy', 'pipeline_accuracy'),
            ('Changed %', 'changed_proportion'),
            ('Semantic Preservation', 'semantic_preservation_rate'),
            ('Avg WN Similarity', 'avg_semantic_similarity'),
        ]

        for label, key in metrics:
            values = []
            for r in all_results:
                v = r.get(key, 'N/A')
                if isinstance(v, float):
                    values.append(f"{v:.4f}")
                else:
                    values.append(str(v))
            f.write(f"| {label} | " + " | ".join(values) + " |\n")

        # Sample results
        for result in all_results:
            f.write(f"\n## Sample Results: {result['dataset']}\n\n")
            f.write("| Complex Word | Replacement | Changed | Sense OK | Sim | Gold Labels (top 3) |\n")
            f.write("|-------------|-------------|---------|----------|-----|--------------------|\n")
            for s in result.get('sample_results', []):
                golds = ', '.join(s['gold_labels'][:3])
                sim = f"{s['semantic_similarity']:.3f}" if s['changed'] else "—"
                sense = "✓" if s['sense_preserved'] else "✗"
                changed = "✓" if s['changed'] else "—"
                f.write(f"| {s['complex_word']} | {s['replacement']} | {changed} | {sense} | {sim} | {golds} |\n")

        # Reference comparison
        f.write("\n## Reference: BERT-LS Paper\n\n")
        f.write("| Metric | Paper (BERT-Large) | Our Target (BERT-Base) |\n")
        f.write("|--------|-------------------|------------------------|\n")
        f.write("| Potential | ~0.80 | 0.75-0.85 |\n")
        f.write("| Precision (SS) | ~0.18 | 0.15-0.20 |\n")
        f.write("| Recall (SS) | ~0.28 | 0.22-0.28 |\n")
        f.write("| F-score (SS) | ~0.22 | 0.18-0.24 |\n")
        f.write("| Accuracy | ~0.50 | 0.40-0.55 |\n")

    print(f"\n  Report saved to: {md_path}")
    print(f"  JSON data saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='BERT-LS Evaluation Pipeline')
    parser.add_argument('--dataset', type=str, default='BenchLS.txt',
                        help='Dataset to evaluate: BenchLS.txt, NNSeval.txt, lex.mturk.txt, or "all"')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to evaluate (for quick testing)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for reports')
    parser.add_argument('--skip-fasttext', action='store_true',
                        help='Skip FastText embeddings (faster startup)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, 'datasets')

    # Determine which datasets to evaluate
    if args.dataset == 'all':
        dataset_files = ['BenchLS.txt', 'NNSeval.txt', 'lex.mturk.txt']
    else:
        dataset_files = [args.dataset]

    # Verify datasets exist
    for df in dataset_files:
        path = os.path.join(datasets_dir, df)
        if not os.path.exists(path):
            print(f"[ERROR] Dataset not found: {path}")
            sys.exit(1)

    # Load engine
    print("=" * 60)
    print("  BERT-LS Evaluation Pipeline")
    print("=" * 60)
    print()
    print("  Loading SimplificationEngine...")

    from simplify_engine import SimplificationEngine

    ppdb_path = os.path.join(datasets_dir, 'ppdb.db')
    if not os.path.exists(ppdb_path):
        ppdb_path = None
        print("  [WARN] PPDB not found")

    freq_path = os.path.join(base_dir, 'SUBTLEX_frequency.xlsx')
    if not os.path.exists(freq_path):
        freq_path = None
        print("  [WARN] Word frequency file not found")

    ft_path = None
    if not args.skip_fasttext:
        ft_file = os.path.join(datasets_dir, 'Fasttext', 'crawl-300d-2M-subword.vec')
        if os.path.exists(ft_file):
            ft_path = ft_file

    engine = SimplificationEngine(
        bert_model='bert-base-uncased',
        word_embeddings_path=ft_path,
        word_frequency_path=freq_path,
        ppdb_path=ppdb_path,
        max_seq_length=128,
        num_selections=10,
        use_cuda=False
    )

    print("  Engine loaded!\n")

    # Run evaluations
    all_results = []
    for df in dataset_files:
        path = os.path.join(datasets_dir, df)
        name = df.replace('.txt', '')

        items = read_benchls(path)
        print(f"  Loaded {len(items)} items from {df}")

        result = evaluate_dataset(engine, items, name, args.max_samples)
        all_results.append(result)

    # Generate report
    output_dir = os.path.join(base_dir, args.output)
    generate_report(all_results, output_dir)

    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
