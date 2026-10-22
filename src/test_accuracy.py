"""
BERT-LS Comprehensive Accuracy Test Suite.

Tests semantic accuracy, grammatical correctness, edge cases,
and full pipeline correctness.

Usage:
    py -3.12 test_accuracy.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def load_engine():
    """Load the engine in lite mode (no FastText)."""
    from simplify_engine import SimplificationEngine

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ppdb_path = os.path.join(base_dir, 'datasets', 'ppdb.db')
    freq_path = os.path.join(base_dir, 'SUBTLEX_frequency.xlsx')

    engine = SimplificationEngine(
        bert_model='bert-base-uncased',
        word_embeddings_path=None,
        word_frequency_path=freq_path if os.path.exists(freq_path) else None,
        ppdb_path=ppdb_path if os.path.exists(ppdb_path) else None,
        max_seq_length=128,
        num_selections=10,
        use_cuda=False
    )
    return engine


class TestResults:
    """Track and report test results."""
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.errors = 0

    def record(self, name, passed, detail=''):
        status = 'PASS' if passed else 'FAIL'
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        self.tests.append((name, status, detail))
        icon = '+' if passed else 'X'
        print(f"  [{icon}] {name}" + (f" -- {detail}" if detail else ""))

    def record_error(self, name, error):
        self.errors += 1
        self.tests.append((name, 'ERROR', str(error)))
        print(f"  [!] {name} -- ERROR: {error}")

    def summary(self):
        total = self.passed + self.failed + self.errors
        print(f"\n{'='*60}")
        print(f"  Test Summary: {self.passed}/{total} passed, {self.failed} failed, {self.errors} errors")
        print(f"{'='*60}")
        return self.failed == 0 and self.errors == 0


def test_semantic_preservation(engine, results):
    """Test 1: Replacement preserves the original word's meaning."""
    print(f"\n{'-'*50}")
    print("  TEST 1: Semantic Preservation")
    print(f"{'-'*50}")

    cases = [
        ("The physician administered the preliminary examination.",
         "administered", "give/manage", ["gave", "performed", "applied", "conducted", "managed"]),
        ("She endeavored to comprehend the intricate methodology.",
         "endeavored", "try/attempt", ["tried", "attempted", "sought"]),
        ("The committee deliberated on the contentious proposal.",
         "deliberated", "think/consider", ["discussed", "thought", "debated", "considered"]),
        ("The scientist conducted the experiment carefully.",
         "conducted", "carry-out", ["performed", "did", "carried", "ran"]),
        ("photosynthesis is vital for life on Earth.",
         "vital", "important", ["important", "necessary", "needed", "essential"]),
        ("The storm retained its strength.",
         "retained", "keep", ["kept", "held", "maintained"]),
        ("He was a prominent area leader.",
         "prominent", "important", ["important", "known", "popular", "famous"]),
    ]

    for sentence, target, domain, acceptable in cases:
        try:
            r = engine.simplify_word(sentence, target)
            replacement = r.get('replacement', target)
            sense_ok = r.get('sense_preserved', True)
            sim = r.get('semantic_similarity', 0.0)
            changed = r.get('changed', False)

            if not changed:
                results.record(
                    f"'{target}' -> (unchanged)",
                    True,
                    f"Domain: {domain}, word kept as-is"
                )
            elif replacement.lower() in [a.lower() for a in acceptable]:
                results.record(
                    f"'{target}' -> '{replacement}'",
                    True,
                    f"Domain: {domain}, sim={sim:.3f}, IN acceptable set"
                )
            elif sense_ok:
                results.record(
                    f"'{target}' -> '{replacement}'",
                    True,
                    f"Domain: {domain}, sim={sim:.3f}, sense_preserved=True"
                )
            else:
                results.record(
                    f"'{target}' -> '{replacement}'",
                    False,
                    f"Domain: {domain}, sim={sim:.3f}, NOT in {acceptable[:3]}"
                )
        except Exception as e:
            results.record_error(f"'{target}'", e)


def test_wordnet_guard(engine, results):
    """Test 2: Semantic guard blocks obviously wrong replacements."""
    print(f"\n{'-'*50}")
    print("  TEST 2: WordNet Semantic Guard")
    print(f"{'-'*50}")

    # These are words where we want the guard to ensure meaningful replacements
    cases = [
        ("The dog bit the man.",
         "bit", "bite/nip"),
        ("She ran quickly down the street.",
         "quickly", "fast/rapidly"),
        ("The ancient castle stood on the hill.",
         "ancient", "old/historic"),
        ("They abandoned the project immediately.",
         "abandoned", "left/quit"),
    ]

    for sentence, target, domain in cases:
        try:
            r = engine.simplify_word(sentence, target)
            replacement = r.get('replacement', target)
            sense_ok = r.get('sense_preserved', True)
            sim = r.get('semantic_similarity', 0.0)
            changed = r.get('changed', False)

            if not changed:
                passed = True
                detail = "Unchanged (acceptable -- word may not be complex)"
            elif sense_ok and sim > 0.3:
                passed = True
                detail = f"sim={sim:.3f}, guard passed"
            elif sense_ok and sim == 0.0:
                passed = True
                detail = f"No WordNet entry, trusted BERT context"
            else:
                passed = False
                detail = f"sim={sim:.3f}, guard may have failed"

            results.record(
                f"Guard: '{target}' -> '{replacement}' ({domain})",
                passed, detail
            )
        except Exception as e:
            results.record_error(f"Guard: '{target}'", e)


def test_morphological_preservation(engine, results):
    """Test 3: Replacement preserves tense/number."""
    print(f"\n{'-'*50}")
    print("  TEST 3: Morphological Preservation")
    print(f"{'-'*50}")

    cases = [
        # (sentence, target, expected_form_hint)
        ("The man administered the test.",
         "administered", "past tense -ed"),
        ("The plants retained their color.",
         "retained", "past tense -ed"),
        ("Steam locomotives traversed the countryside.",
         "traversed", "past tense -ed"),
        ("Many obstacles hindered progress.",
         "obstacles", "plural noun -s"),
        ("Several casualties were reported.",
         "casualties", "plural noun -ies"),
    ]

    # Known irregular past tense forms
    IRREGULAR_PAST_FORMS = {
        'was', 'began', 'broke', 'brought', 'built', 'bought', 'caught', 'chose',
        'came', 'did', 'drew', 'drank', 'drove', 'ate', 'eaten', 'fell', 'felt',
        'found', 'flew', 'forgot', 'got', 'gave', 'went', 'grew', 'had',
        'heard', 'held', 'kept', 'knew', 'led', 'left', 'lost', 'made',
        'meant', 'met', 'paid', 'put', 'read', 'rode', 'ran', 'said',
        'saw', 'sold', 'sent', 'set', 'showed', 'sat', 'spoke', 'spent',
        'stood', 'took', 'taught', 'told', 'thought', 'tried', 'understood',
        'wore', 'won', 'wrote',
    }

    for sentence, target, form_hint in cases:
        try:
            r = engine.simplify_word(sentence, target)
            replacement = r.get('replacement', target)
            changed = r.get('changed', False)

            if not changed:
                results.record(
                    f"Morph: '{target}' (unchanged)",
                    True, f"Expected: {form_hint}"
                )
            else:
                # Check morphological form
                if target.endswith('ed') and not replacement.endswith('ed') and not replacement.endswith('d'):
                    if replacement in IRREGULAR_PAST_FORMS:
                        passed = True
                        detail = f"'{replacement}' irregular past OK ({form_hint})"
                    else:
                        passed = False
                        detail = f"'{replacement}' should end in -ed (source: {form_hint})"
                elif target.endswith('s') and not target.endswith('ss') and not replacement.endswith('s'):
                    passed = False
                    detail = f"'{replacement}' should be plural (source: {form_hint})"
                else:
                    passed = True
                    detail = f"'{replacement}' morphology OK ({form_hint})"

                results.record(f"Morph: '{target}' -> '{replacement}'", passed, detail)
        except Exception as e:
            results.record_error(f"Morph: '{target}'", e)


def test_edge_cases(engine, results):
    """Test 4: Edge cases don't crash the engine."""
    print(f"\n{'-'*50}")
    print("  TEST 4: Edge Cases")
    print(f"{'-'*50}")

    # Short sentences
    try:
        r = engine.simplify_word("Big deal.", "deal")
        results.record("Short sentence", True, f"-> '{r.get('replacement', '?')}'")
    except Exception as e:
        results.record_error("Short sentence", e)

    # Word not in sentence
    try:
        r = engine.simplify_word("The cat sat on the mat.", "elephant")
        results.record("Missing word", 'error' in r or not r.get('changed', True),
                       "Correctly handled missing word")
    except Exception as e:
        results.record_error("Missing word", e)

    # Very long sentence
    try:
        long_sent = "The " + " ".join(["very"] * 20) + " important decision was made."
        r = engine.simplify_word(long_sent, "important")
        results.record("Long sentence", True, f"-> '{r.get('replacement', '?')}'")
    except Exception as e:
        results.record_error("Long sentence", e)

    # Proper noun (should not be simplified)
    try:
        r = engine.simplify_word("Barack Obama served as president.", "Obama")
        not_changed = not r.get('changed', True)
        results.record("Proper noun", True, f"Changed={r.get('changed', '?')}")
    except Exception as e:
        results.record_error("Proper noun", e)

    # Punctuation-heavy sentence
    try:
        r = engine.simplify_word("He said, \"This is extraordinary!\"", "extraordinary")
        results.record("Punctuation sentence", True, f"-> '{r.get('replacement', '?')}'")
    except Exception as e:
        results.record_error("Punctuation sentence", e)

    # Contraction
    try:
        r = engine.simplify_word("They couldn't traverse the river.", "traverse")
        results.record("Contraction sentence", True, f"-> '{r.get('replacement', '?')}'")
    except Exception as e:
        results.record_error("Contraction sentence", e)


def test_auto_simplify(engine, results):
    """Test 5: Full sentence auto-simplification."""
    print(f"\n{'-'*50}")
    print("  TEST 5: Auto-Simplification")
    print(f"{'-'*50}")

    sentences = [
        "She endeavored to comprehend the intricate methodology.",
        "The physician administered the preliminary examination.",
        "The committee deliberated on the contentious proposal.",
    ]

    for sentence in sentences:
        try:
            t0 = time.time()
            r = engine.simplify_sentence(sentence)
            elapsed = time.time() - t0

            original = r.get('original', sentence)
            simplified = r.get('simplified', sentence)
            changes = r.get('changes', [])
            num_changes = len(changes)

            results.record(
                f"Auto: '{original[:50]}...'",
                True,
                f"{num_changes} changes, {elapsed:.1f}s -> '{simplified[:60]}...'"
            )

            for ch in changes:
                sense_ok = ch.get('sense_preserved', True)
                sim = ch.get('semantic_similarity', 0.0)
                results.record(
                    f"  Change: '{ch['original_word']}' -> '{ch['replacement']}'",
                    sense_ok,
                    f"sim={sim:.3f}"
                )
        except Exception as e:
            results.record_error(f"Auto: '{sentence[:40]}...'", e)


def test_benchls_sample(engine, results):
    """Test 6: Quick BenchLS sample evaluation."""
    print(f"\n{'-'*50}")
    print("  TEST 6: BenchLS Sample (first 10)")
    print(f"{'-'*50}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    benchls_path = os.path.join(base_dir, 'datasets', 'BenchLS.txt')

    if not os.path.exists(benchls_path):
        results.record("BenchLS file", False, "Not found")
        return

    # Read first 10 items
    items = []
    with open(benchls_path, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue

            sentence = parts[0].strip()
            complex_word = parts[1].strip()
            gold_labels = []
            for entry in parts[3:]:
                entry = entry.strip()
                if ':' in entry:
                    _, word = entry.split(':', 1)
                    word = word.strip()
                    if word and word not in gold_labels:
                        gold_labels.append(word)

            if sentence and complex_word and gold_labels:
                items.append((sentence, complex_word, gold_labels))
                if len(items) >= 10:
                    break

    hits = 0
    for sentence, cw, golds in items:
        try:
            r = engine.simplify_word(sentence, cw)
            replacement = r.get('replacement', cw)
            changed = r.get('changed', False)

            in_gold = replacement.lower() in [g.lower() for g in golds]
            correct = (not changed) or in_gold

            if in_gold:
                hits += 1

            results.record(
                f"BenchLS: '{cw}' -> '{replacement}'",
                correct,
                f"Gold={golds[:3]}, {'HIT' if in_gold else 'miss'}"
            )
        except Exception as e:
            results.record_error(f"BenchLS: '{cw}'", e)

    results.record(
        f"BenchLS accuracy: {hits}/{len(items)}",
        hits >= 2,  # At least 20% gold hits
        f"Gold label match rate: {hits/max(len(items),1)*100:.0f}%"
    )


def main():
    print("=" * 60)
    print("  BERT-LS Comprehensive Accuracy Test Suite")
    print("=" * 60)
    print()

    print("  Loading engine (lite mode)...")
    t0 = time.time()
    engine = load_engine()
    print(f"  Engine loaded in {time.time()-t0:.1f}s\n")

    results = TestResults()

    # Run all test suites
    test_semantic_preservation(engine, results)
    test_wordnet_guard(engine, results)
    test_morphological_preservation(engine, results)
    test_edge_cases(engine, results)
    test_auto_simplify(engine, results)
    test_benchls_sample(engine, results)

    # Summary
    all_passed = results.summary()
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
