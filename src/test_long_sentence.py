"""Quick test: identify complex words + simplify the long sentence."""
import json, urllib.request

sentence = (
    "Although the project was initially intended to be completed by the end of the fiscal year, "
    "we had to extend the deadline to the following spring because the primary stakeholders "
    "requested significant modifications to the final deliverables."
)

print("=" * 70)
print("  SENTENCE ANALYSIS")
print("=" * 70)
print(f"\n  Input ({len(sentence.split())} words):")
print(f"  {sentence}\n")

# Step 1: Identify complex words
print("-" * 70)
print("  STEP 1: Complex Word Identification")
print("-" * 70)

data = json.dumps({"sentence": sentence}).encode()
req = urllib.request.Request(
    "http://localhost:5000/api/identify",
    data=data,
    headers={"Content-Type": "application/json"},
)

try:
    resp = urllib.request.urlopen(req, timeout=60)
    id_result = json.loads(resp.read())
    cws = id_result.get("complex_words", [])
    print(f"  Found {len(cws)} complex words ({id_result.get('time_ms', 0)}ms)\n")
    for cw in cws:
        pct = int(cw["complexity"] * 100)
        bar = "#" * int(cw["complexity"] * 20)
        print(f"    {cw['word']:20s}  {bar:20s}  {pct}%")
    print()
except Exception as e:
    print(f"  ERROR: {e}\n")

# Step 2: Full simplification
print("-" * 70)
print("  STEP 2: Simplification")
print("-" * 70)

req2 = urllib.request.Request(
    "http://localhost:5000/api/simplify_auto",
    data=data,
    headers={"Content-Type": "application/json"},
)

try:
    resp2 = urllib.request.urlopen(req2, timeout=300)
    result = json.loads(resp2.read())

    print(f"\n  Original:")
    print(f"  {result.get('original', '')}\n")
    print(f"  Simplified:")
    print(f"  {result.get('simplified', '')}\n")
    
    changes = result.get("changes", [])
    if changes:
        print(f"  Word Changes ({len(changes)}):")
        for c in changes:
            sim = c.get("semantic_similarity", 0)
            sense = "[OK]" if c.get("sense_preserved") else "[FAIL]"
            cands = c.get("candidates", [])[:4]
            print(f"    {c['original_word']:20s} --> {c['replacement']:16s}  sim={sim:.3f}  {sense}  candidates: {cands}")
    else:
        print("  No changes made.")
    
    t = result.get("time_ms", 0)
    print(f"\n  Processing time: {t/1000:.1f}s")
    
    # Quality analysis
    print("\n" + "-" * 70)
    print("  QUALITY ANALYSIS")
    print("-" * 70)
    for c in changes:
        orig = c["original_word"]
        repl = c["replacement"]
        sim = c.get("semantic_similarity", 0)
        
        expected = {
            "initially": ["first", "originally", "at first"],
            "intended": ["meant", "planned", "supposed"],
            "completed": ["finished", "done"],
            "fiscal": ["financial", "tax", "budget"],
            "extend": ["lengthen", "push", "move"],
            "deadline": ["due date", "date", "time limit"],
            "stakeholders": ["people involved", "partners", "investors"],
            "requested": ["asked for", "asked", "wanted"],
            "significant": ["big", "large", "major", "important"],
            "modifications": ["changes", "updates", "edits"],
            "deliverables": ["results", "outputs", "work", "products"],
            "primary": ["main", "key", "chief"],
        }
        
        exp = expected.get(orig, [])
        if repl in exp:
            verdict = "IDEAL"
        elif sim >= 0.7:
            verdict = "GOOD"
        elif sim >= 0.55:
            verdict = "OK"
        else:
            verdict = "POOR"
        
        print(f"    {orig:20s} --> {repl:16s}  [{verdict}]  (expected: {exp[:3]})")

except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 70)
