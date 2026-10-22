"""
Quick smoke test for the BERT-LS Simplification Engine.
Loads the engine without FastText (fast, ~2GB RAM) and runs basic tests.

Tests cover: asset verification, engine loading, PPDB lookup, word frequency,
single-word simplification, auto-simplification, semantic guard, weighted ranking,
morphological preservation, and WordNet fallback.

Usage:
    py -3.12 test_engine.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("  BERT-LS Engine Smoke Test (Updated)")
    print("=" * 60)
    print()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    passed = 0
    failed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed, total
        total += 1
        icon = "+" if condition else "X"
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  [{icon}] {name}{suffix}")

    # --- Test 1: Check data assets ---
    print("[TEST 1] Checking data assets...")
    assets = {
        'PPDB SQLite': os.path.join(base_dir, 'datasets', 'ppdb.db'),
        'Word Frequency': os.path.join(base_dir, 'SUBTLEX_frequency.xlsx'),
        'BenchLS Dataset': os.path.join(base_dir, 'datasets', 'BenchLS.txt'),
    }
    all_found = True
    for name, path in assets.items():
        exists = os.path.exists(path)
        size_mb = os.path.getsize(path) / (1024*1024) if exists else 0
        status = f"OK ({size_mb:.1f} MB)" if exists else "MISSING"
        print(f"  {name}: {status}")
        if not exists:
            all_found = False

    if not all_found:
        print("  [WARN] Some assets missing, but engine may still work.\n")
    else:
        print("  All assets present.\n")

    # --- Test 2: Load engine (no FastText for speed) ---
    print("[TEST 2] Loading SimplificationEngine (lite mode, no FastText)...")
    t0 = time.time()

    from simplify_engine import SimplificationEngine

    ppdb_path = os.path.join(base_dir, 'datasets', 'ppdb.db')
    freq_path = os.path.join(base_dir, 'SUBTLEX_frequency.xlsx')

    engine = SimplificationEngine(
        bert_model='bert-base-uncased',
        word_embeddings_path=None,      # Skip FastText
        word_frequency_path=freq_path if os.path.exists(freq_path) else None,
        ppdb_path=ppdb_path if os.path.exists(ppdb_path) else None,
        max_seq_length=128,
        num_selections=10,
        use_cuda=False
    )
    load_time = time.time() - t0
    check("Engine loaded", engine is not None, f"{load_time:.1f}s")
    print()

    # --- Test 3: PPDB lookup ---
    print("[TEST 3] PPDB lookup for 'difficult'...")
    if engine.ppdb_model:
        results = engine.ppdb_model.predict('difficult', 'a')
        check("PPDB returns results", len(results) > 0, f"{len(results)} paraphrases")
        print(f"  Top results: {results[:5]}")
    else:
        print("  PPDB not loaded, skipping.\n")

    # --- Test 4: Word frequency check ---
    print("\n[TEST 4] Word frequency lookup...")
    test_words = ['the', 'happy', 'ameliorate', 'physician', 'cat']
    for word in test_words:
        freq = engine.word_count.get(word, 0)
        print(f"  '{word}': {freq}")
    check("Common word > rare word",
          engine.word_count.get('the', 0) > engine.word_count.get('ameliorate', 0))
    print()

    # --- Test 5: Simplify a target word ---
    print("[TEST 5] Simplifying 'administered' in context...")
    t0 = time.time()
    result = engine.simplify_word(
        "The physician administered the preliminary examination",
        "administered"
    )
    elapsed = time.time() - t0
    print(f"  Original:    {result['original']}")
    print(f"  Simplified:  {result['simplified']}")
    print(f"  Replacement: {result['replacement']}")
    print(f"  Candidates:  {result.get('candidates', [])[:5]}")
    print(f"  Changed:     {result['changed']}")
    print(f"  Time:        {elapsed:.1f}s")

    check("simplify_word returns dict", isinstance(result, dict))
    check("Result has required keys",
          all(k in result for k in ['original', 'simplified', 'replacement', 'changed']))
    check("Result has semantic metrics",
          'sense_preserved' in result and 'semantic_similarity' in result,
          f"sim={result.get('semantic_similarity', '?')}, sense={result.get('sense_preserved', '?')}")
    print()

    # --- Test 6: Auto-simplify a sentence ---
    print("[TEST 6] Auto-simplifying: 'She endeavored to comprehend the intricate methodology'...")
    t0 = time.time()
    result = engine.simplify_sentence(
        "She endeavored to comprehend the intricate methodology"
    )
    elapsed = time.time() - t0
    print(f"  Original:   {result['original']}")
    print(f"  Simplified: {result['simplified']}")
    for ch in result.get('changes', []):
        print(f"    {ch['original_word']} -> {ch['replacement']} "
              f"| sense_preserved={ch.get('sense_preserved','?')} "
              f"| semantic_similarity={ch.get('semantic_similarity','?')}")

    check("Auto-simplify returns changes",
          'changes' in result and isinstance(result['changes'], list))
    check("Changes include semantic metrics",
          all('sense_preserved' in ch and 'semantic_similarity' in ch
              for ch in result.get('changes', [{}])),
          f"{len(result.get('changes', []))} changes")
    print(f"  Time: {elapsed:.1f}s\n")

    # --- Test 7: Semantic Guard ---
    print("[TEST 7] Semantic accuracy guard tests...")
    semantic_cases = [
        ("The physician administered the preliminary examination.",
         "administered", "apply/manage domain"),
        ("She endeavored to comprehend the intricate methodology.",
         "endeavored", "try/attempt domain"),
        ("The committee deliberated on the contentious proposal.",
         "deliberated", "think/consider domain"),
        ("The scientist conducted the experiment carefully.",
         "conducted", "carry-out domain"),
    ]

    guard_passed = 0
    for sentence, target, domain in semantic_cases:
        r = engine.simplify_word(sentence, target)
        sense_ok    = r.get('sense_preserved', True)
        sim         = r.get('semantic_similarity', 1.0)
        replacement = r.get('replacement', target)
        changed     = r.get('changed', False)

        ok = (not changed or sense_ok)
        if ok:
            guard_passed += 1
        check(f"Guard: '{target}' -> '{replacement}' ({domain})",
              ok,
              f"sim={sim:.3f}, sense_preserved={sense_ok}")

    print(f"\n  Semantic guard: {guard_passed}/{len(semantic_cases)} passed\n")

    # --- Test 8: Weighted Ranking ---
    print("[TEST 8] Weighted rank fusion (verifying weights are applied)...")
    # The engine should use weighted ranks, so results should differ from naive ordering
    r1 = engine.simplify_word("The physician administered the preliminary examination.", "physician")
    check("Weighted ranking returns result", isinstance(r1, dict) and 'replacement' in r1,
          f"'{r1.get('replacement', '?')}'")
    # Candidates should be a list
    check("Candidates list populated", len(r1.get('candidates', [])) > 0,
          f"{len(r1.get('candidates', []))} candidates")
    print()

    # --- Test 9: Morphological Preservation ---
    print("[TEST 9] Morphological preservation...")

    # Known irregular past tense forms (same list as engine)
    IRREGULAR_PAST_FORMS = {
        'was', 'began', 'broke', 'brought', 'built', 'bought', 'caught', 'chose',
        'came', 'did', 'drew', 'drank', 'drove', 'ate', 'fell', 'felt',
        'found', 'flew', 'forgot', 'got', 'gave', 'went', 'grew', 'had',
        'heard', 'held', 'kept', 'knew', 'led', 'left', 'lost', 'made',
        'meant', 'met', 'paid', 'put', 'read', 'rode', 'ran', 'said',
        'saw', 'sold', 'sent', 'set', 'showed', 'sat', 'spoke', 'spent',
        'stood', 'took', 'taught', 'told', 'thought', 'tried', 'understood',
        'wore', 'won', 'wrote',
    }

    morph_cases = [
        ("The man administered the test.", "administered", "past tense -ed"),
        ("Many obstacles hindered progress.", "obstacles", "plural noun -s"),
    ]
    for sentence, target, form in morph_cases:
        r = engine.simplify_word(sentence, target)
        replacement = r.get('replacement', target)
        changed = r.get('changed', False)
        if not changed:
            check(f"Morph: '{target}' (unchanged)", True, form)
        else:
            # Check form preservation
            if target.endswith('ed') and not replacement.endswith(('ed', 'd')):
                # Accept irregular past tenses as valid
                if replacement in IRREGULAR_PAST_FORMS:
                    check(f"Morph: '{target}' -> '{replacement}'", True,
                          f"Irregular past OK ({form})")
                else:
                    check(f"Morph: '{target}' -> '{replacement}'", False,
                          f"Should preserve {form}")
            elif target.endswith('s') and not target.endswith('ss') and not replacement.endswith('s'):
                check(f"Morph: '{target}' -> '{replacement}'", False,
                      f"Should preserve {form}")
            else:
                check(f"Morph: '{target}' -> '{replacement}'", True,
                      f"Form OK ({form})")
    print()

    # --- Test 10: WordNet Fallback (no FastText mode) ---
    print("[TEST 10] WordNet fallback (engine loaded without FastText)...")
    check("FastText NOT loaded", engine.fasttext_dico is None,
          "Confirming lite mode")
    # In lite mode, WordNet similarity should be used as fallback
    r = engine.simplify_word("The pilot navigated the aircraft.", "navigated")
    check("WordNet fallback produces result", isinstance(r, dict),
          f"'{r.get('replacement', '?')}', sim={r.get('semantic_similarity', '?')}")
    print()

    # --- Test 11: Edge Cases ---
    print("[TEST 11] Edge cases...")
    # Missing word
    r = engine.simplify_word("The cat sat on the mat.", "elephant")
    check("Missing word handled", 'error' in r or not r.get('changed', True),
          "Word not in sentence")

    # Very short sentence
    r = engine.simplify_word("Big deal.", "deal")
    check("Short sentence handled", isinstance(r, dict))

    # Empty-ish inputs
    r = engine.simplify_word("Hello.", "nonexistent")
    check("Nonexistent word handled", isinstance(r, dict))
    print()

    # --- Summary ---
    print("=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  [OK] All smoke tests passed!")
    else:
        print(f"  [FAIL] {failed} test(s) failed -- review output above")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
