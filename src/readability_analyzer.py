"""
Readability-Aware Lexical Simplification.

Wraps SimplificationEngine with readability measurement and grade-level targeting.
Uses the textstat library for readability scores (Flesch-Kincaid Grade, SMOG Index,
ARI, Coleman-Liau Index, Dale-Chall, Flesch Reading Ease) — all O(n) string
operations, no additional models required.

Works in lite mode (no FastText). Only new dependency: textstat (pip install textstat).

Usage:
    # Python API
    from simplify_engine import SimplificationEngine
    from readability_analyzer import ReadabilityAnalyzer

    engine = SimplificationEngine(
        word_frequency_path='SUBTLEX_frequency.xlsx',
        ppdb_path='datasets/ppdb.db',
    )
    analyzer = ReadabilityAnalyzer(engine, target_grade=8)
    result = analyzer.simplify_to_target("The physician administered the examination.")
    print(result.summary())

    # CLI
    python readability_analyzer.py "Your complex text here." --target-grade 8
    python readability_analyzer.py "Text." --analyze-only
    python readability_analyzer.py "Text." --json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

try:
    import textstat
except ImportError:
    raise ImportError(
        "textstat is required: pip install textstat\n"
        "It is a lightweight pure-Python library with no model downloads."
    )

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize
except ImportError:
    def sent_tokenize(text):
        """Minimal sentence splitter fallback when nltk is unavailable."""
        import re
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


# ---------------------------------------------------------------------------
# Grade-level labels
# ---------------------------------------------------------------------------

_GRADE_LABELS = [
    (0,   5,  "Elementary (Grade 1-5)"),
    (5,   7,  "Middle School (Grade 5-7)"),
    (7,   9,  "High School (Grade 7-9)"),
    (9,  12,  "College Prep (Grade 9-12)"),
    (12, 15,  "College Level"),
    (15, 99,  "Graduate / Professional"),
]


def _grade_label(grade: float) -> str:
    for low, high, label in _GRADE_LABELS:
        if low <= grade < high:
            return label
    return "Graduate / Professional"


def _dale_chall_to_grade(score: float) -> float:
    """Convert Dale-Chall score to approximate US grade level."""
    if score <= 4.9:
        return 4.0
    elif score <= 5.9:
        return 5.5
    elif score <= 6.9:
        return 7.5
    elif score <= 7.9:
        return 9.5
    elif score <= 8.9:
        return 11.5
    elif score <= 9.9:
        return 13.0
    else:
        return 16.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReadabilityScores:
    """All six readability scores for a piece of text."""

    flesch_kincaid_grade: float
    smog_index: float
    automated_readability_index: float
    coleman_liau_index: float
    dale_chall_score: float
    flesch_reading_ease: float
    composite_grade: float   # mean of FK + SMOG + ARI + CL
    interpretation: str      # human-readable label

    def delta(self, other: 'ReadabilityScores') -> Dict[str, float]:
        """Return improvement dict (self - other). Positive = self is harder."""
        return {
            'flesch_kincaid_grade': self.flesch_kincaid_grade - other.flesch_kincaid_grade,
            'smog_index': self.smog_index - other.smog_index,
            'automated_readability_index': self.automated_readability_index - other.automated_readability_index,
            'coleman_liau_index': self.coleman_liau_index - other.coleman_liau_index,
            'dale_chall_score': self.dale_chall_score - other.dale_chall_score,
            'flesch_reading_ease': self.flesch_reading_ease - other.flesch_reading_ease,
            'composite_grade': self.composite_grade - other.composite_grade,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'flesch_kincaid_grade': self.flesch_kincaid_grade,
            'smog_index': self.smog_index,
            'automated_readability_index': self.automated_readability_index,
            'coleman_liau_index': self.coleman_liau_index,
            'dale_chall_score': self.dale_chall_score,
            'flesch_reading_ease': self.flesch_reading_ease,
            'composite_grade': self.composite_grade,
            'interpretation': self.interpretation,
        }

    def __repr__(self) -> str:
        return (
            f"ReadabilityScores("
            f"FK={self.flesch_kincaid_grade:.1f}, "
            f"SMOG={self.smog_index:.1f}, "
            f"ARI={self.automated_readability_index:.1f}, "
            f"CL={self.coleman_liau_index:.1f}, "
            f"Flesch={self.flesch_reading_ease:.1f}, "
            f"Composite={self.composite_grade:.1f} [{self.interpretation}])"
        )


@dataclass
class SentenceResult:
    """Per-sentence analysis result."""

    original: str
    simplified: str
    changes: List[Dict[str, Any]]  # [{original_word, replacement, similarity}]
    before_scores: ReadabilityScores
    after_scores: ReadabilityScores
    grade_improvement: float        # composite_grade before - after (positive = easier)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'original': self.original,
            'simplified': self.simplified,
            'changes': self.changes,
            'before_composite_grade': self.before_scores.composite_grade,
            'after_composite_grade': self.after_scores.composite_grade,
            'grade_improvement': self.grade_improvement,
        }


@dataclass
class DocumentResult:
    """Full document analysis result."""

    original_text: str
    simplified_text: str
    before_scores: ReadabilityScores
    after_scores: ReadabilityScores
    sentence_results: List[SentenceResult]
    target_grade: float
    total_changes: int
    sentences_changed: int
    total_sentences: int
    processing_time_sec: float

    @property
    def grade_improvement(self) -> float:
        return self.before_scores.composite_grade - self.after_scores.composite_grade

    @property
    def target_achieved(self) -> bool:
        return self.after_scores.composite_grade <= self.target_grade

    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_text': self.original_text,
            'simplified_text': self.simplified_text,
            'before_scores': self.before_scores.to_dict(),
            'after_scores': self.after_scores.to_dict(),
            'sentence_results': [s.to_dict() for s in self.sentence_results],
            'target_grade': self.target_grade,
            'target_achieved': self.target_achieved,
            'grade_improvement': self.grade_improvement,
            'total_changes': self.total_changes,
            'sentences_changed': self.sentences_changed,
            'total_sentences': self.total_sentences,
            'processing_time_sec': self.processing_time_sec,
        }

    def summary(self) -> str:
        """Return an ASCII-formatted readability analysis report."""
        lines = []
        sep = "=" * 62

        lines.append(sep)
        lines.append("  READABILITY ANALYSIS REPORT")
        lines.append(sep)
        lines.append("")
        lines.append(f"  Target Grade Level:  {self.target_grade:.1f}   |   "
                     f"Processing time: {self.processing_time_sec:.1f}s")
        wc = len(self.original_text.split())
        lines.append(f"  Text length: {wc} words ({self.total_sentences} sentences)")
        if wc < 50:
            lines.append("  [!] Short text — readability scores less reliable below 50 words")
        lines.append("")

        # Score table
        header = f"  {'SCORE':<20}{'BEFORE':>8}  {'AFTER':>8}  {'CHANGE':>8}"
        lines.append(header)
        lines.append("  " + "-" * 50)

        def _row(label, before, after, higher_is_easier=False):
            change = after - before
            sign = "+" if change > 0 else ""
            # For Flesch Ease, higher = easier, so colour logic inverts
            arrow = ""
            if higher_is_easier:
                arrow = " (easier)" if change > 0 else (" (harder)" if change < 0 else "")
            else:
                arrow = " (easier)" if change < 0 else (" (harder)" if change > 0 else "")
            return f"  {label:<20}{before:>8.1f}  {after:>8.1f}  {sign}{change:>7.1f}{arrow}"

        lines.append(_row("FK Grade", self.before_scores.flesch_kincaid_grade,
                          self.after_scores.flesch_kincaid_grade))
        lines.append(_row("SMOG Index", self.before_scores.smog_index,
                          self.after_scores.smog_index))
        lines.append(_row("ARI", self.before_scores.automated_readability_index,
                          self.after_scores.automated_readability_index))
        lines.append(_row("Coleman-Liau", self.before_scores.coleman_liau_index,
                          self.after_scores.coleman_liau_index))
        lines.append(_row("Dale-Chall", self.before_scores.dale_chall_score,
                          self.after_scores.dale_chall_score))
        lines.append(_row("Flesch Ease", self.before_scores.flesch_reading_ease,
                          self.after_scores.flesch_reading_ease, higher_is_easier=True))
        lines.append("  " + "-" * 50)
        lines.append(f"  {'Composite Grade':<20}{self.before_scores.composite_grade:>8.1f}"
                     f"  {self.after_scores.composite_grade:>8.1f}")
        lines.append(f"  Before: {self.before_scores.interpretation}")
        lines.append(f"  After:  {self.after_scores.interpretation}")
        lines.append("")

        # Improvement summary
        imp = self.grade_improvement
        if imp > 0:
            status = "TARGET ACHIEVED" if self.target_achieved else f"TARGET: {self.target_grade:.1f}"
            lines.append(f"  IMPROVEMENT: -{imp:.1f} grade levels  [{status}]")
        elif imp == 0:
            lines.append("  IMPROVEMENT: none (no simplifications made)")
        else:
            lines.append(f"  NOTE: Grade increased by {abs(imp):.1f} levels (unusual — check output)")
        lines.append("")

        # Per-sentence breakdown
        if self.sentence_results:
            lines.append("  PER-SENTENCE BREAKDOWN")
            lines.append("  " + "-" * 50)
            for i, sr in enumerate(self.sentence_results, 1):
                changes_str = ", ".join(
                    f"{c['original_word']}->{c['replacement']}"
                    for c in sr.changes
                ) if sr.changes else "no changes"
                grade_str = f"+{sr.grade_improvement:.1f}" if sr.grade_improvement >= 0 else f"{sr.grade_improvement:.1f}"
                lines.append(f"  S{i} [{grade_str}]: {changes_str}")
                if sr.changes:
                    lines.append(f"    Before: \"{sr.original}\"")
                    lines.append(f"    After:  \"{sr.simplified}\"")
            lines.append("")

        lines.append(f"  TOTAL: {self.total_changes} change(s) across "
                     f"{self.sentences_changed}/{self.total_sentences} sentence(s)")
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ReadabilityAnalyzer:
    """
    Readability-aware lexical simplification wrapper.

    Measures text readability before and after BERT-LS simplification using
    six established formulas. Supports grade-level targeting with convergence
    detection.

    Args:
        engine: A loaded SimplificationEngine instance.
        target_grade: Target composite grade level (default 8.0 = general public).
        max_passes: Maximum simplification passes per sentence (default 3).
        convergence_threshold: Stop when remaining grade above target < this (default 0.2).
        min_improvement_per_pass: Stop if a pass improves composite grade by less than
            this value (default 0.1 grade levels).
    """

    def __init__(
        self,
        engine,
        target_grade: float = 8.0,
        max_passes: int = 3,
        convergence_threshold: float = 0.2,
        min_improvement_per_pass: float = 0.1,
    ):
        self.engine = engine
        self.target_grade = target_grade
        self.max_passes = max_passes
        self.convergence_threshold = convergence_threshold
        self.min_improvement_per_pass = min_improvement_per_pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> ReadabilityScores:
        """
        Compute all six readability scores for a text.

        Args:
            text: The text to score (any length; scores are more reliable for 50+ words).

        Returns:
            ReadabilityScores instance with all six scores plus composite grade.
        """
        if not text or not text.strip():
            return ReadabilityScores(
                flesch_kincaid_grade=0.0, smog_index=0.0,
                automated_readability_index=0.0, coleman_liau_index=0.0,
                dale_chall_score=0.0, flesch_reading_ease=100.0,
                composite_grade=0.0, interpretation="N/A (empty text)"
            )

        fk = textstat.flesch_kincaid_grade(text)
        ari = textstat.automated_readability_index(text)
        cl = textstat.coleman_liau_index(text)
        ease = textstat.flesch_reading_ease(text)
        dc = textstat.dale_chall_readability_score(text)

        # SMOG is unreliable for < 3 sentences; fall back to mean of other indices
        sent_count = textstat.sentence_count(text)
        if sent_count >= 3:
            smog = textstat.smog_index(text)
        else:
            smog = (fk + ari + cl) / 3.0

        # Clamp grade values to reasonable range [0, 20]
        fk   = max(0.0, min(20.0, fk))
        smog = max(0.0, min(20.0, smog))
        ari  = max(0.0, min(20.0, ari))
        cl   = max(0.0, min(20.0, cl))

        composite = (fk + smog + ari + cl) / 4.0
        label = _grade_label(composite)

        return ReadabilityScores(
            flesch_kincaid_grade=round(fk, 1),
            smog_index=round(smog, 1),
            automated_readability_index=round(ari, 1),
            coleman_liau_index=round(cl, 1),
            dale_chall_score=round(dc, 1),
            flesch_reading_ease=round(ease, 1),
            composite_grade=round(composite, 2),
            interpretation=label,
        )

    def analyze_only(self, text: str) -> ReadabilityScores:
        """Score text without simplifying. Useful for benchmarking input texts."""
        return self.score_text(text)

    def simplify_to_target(self, text: str) -> DocumentResult:
        """
        Simplify text until the target grade level is reached or improvement plateaus.

        Splits text into sentences, simplifies each sentence iteratively, then
        measures document-level readability before and after.

        Args:
            text: Full text to simplify.

        Returns:
            DocumentResult with before/after scores, per-sentence analysis, and summary.
        """
        start_time = time.time()

        before_scores = self.score_text(text)
        sentences = sent_tokenize(text)
        sentence_results: List[SentenceResult] = []

        simplified_sentences = []
        for sentence in sentences:
            sr = self._simplify_sentence_to_target(sentence, pass_num=0)
            sentence_results.append(sr)
            simplified_sentences.append(sr.simplified)

        simplified_text = " ".join(simplified_sentences)
        after_scores = self.score_text(simplified_text)

        total_changes = sum(len(sr.changes) for sr in sentence_results)
        sentences_changed = sum(1 for sr in sentence_results if sr.changes)
        elapsed = time.time() - start_time

        return DocumentResult(
            original_text=text,
            simplified_text=simplified_text,
            before_scores=before_scores,
            after_scores=after_scores,
            sentence_results=sentence_results,
            target_grade=self.target_grade,
            total_changes=total_changes,
            sentences_changed=sentences_changed,
            total_sentences=len(sentences),
            processing_time_sec=round(elapsed, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simplify_sentence_to_target(
        self, sentence: str, pass_num: int = 0
    ) -> SentenceResult:
        """
        Iteratively simplify one sentence toward the target grade level.

        Returns a SentenceResult even if no changes are made (grade already at target,
        no simplifiable words, or improvement plateaued).
        """
        before_scores = self.score_text(sentence)

        # Already at or below target — no simplification needed
        if before_scores.composite_grade <= self.target_grade + self.convergence_threshold:
            return SentenceResult(
                original=sentence,
                simplified=sentence,
                changes=[],
                before_scores=before_scores,
                after_scores=before_scores,
                grade_improvement=0.0,
            )

        # Max passes reached
        if pass_num >= self.max_passes:
            return SentenceResult(
                original=sentence,
                simplified=sentence,
                changes=[],
                before_scores=before_scores,
                after_scores=before_scores,
                grade_improvement=0.0,
            )

        # Run one simplification pass via the engine
        engine_result = self._run_engine_on_sentence(sentence)
        simplified = engine_result.get('simplified', sentence)
        changes = engine_result.get('changes', [])

        # No changes made — converged (no simplifiable words found)
        if not changes or simplified == sentence:
            return SentenceResult(
                original=sentence,
                simplified=sentence,
                changes=[],
                before_scores=before_scores,
                after_scores=before_scores,
                grade_improvement=0.0,
            )

        after_scores = self.score_text(simplified)
        grade_improvement = before_scores.composite_grade - after_scores.composite_grade

        # Improvement plateaued — stop recursing
        if grade_improvement < self.min_improvement_per_pass:
            return SentenceResult(
                original=sentence,
                simplified=simplified,
                changes=changes,
                before_scores=before_scores,
                after_scores=after_scores,
                grade_improvement=round(grade_improvement, 2),
            )

        # Still above target and more passes allowed — recurse
        if (after_scores.composite_grade > self.target_grade + self.convergence_threshold
                and pass_num + 1 < self.max_passes):
            deeper = self._simplify_sentence_to_target(simplified, pass_num + 1)
            # Combine changes from this pass with deeper passes
            all_changes = changes + [
                c for c in deeper.changes
                if c['original_word'] not in {ch['original_word'] for ch in changes}
            ]
            total_improvement = before_scores.composite_grade - deeper.after_scores.composite_grade
            return SentenceResult(
                original=sentence,
                simplified=deeper.simplified,
                changes=all_changes,
                before_scores=before_scores,
                after_scores=deeper.after_scores,
                grade_improvement=round(total_improvement, 2),
            )

        return SentenceResult(
            original=sentence,
            simplified=simplified,
            changes=changes,
            before_scores=before_scores,
            after_scores=after_scores,
            grade_improvement=round(grade_improvement, 2),
        )

    def _run_engine_on_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Call SimplificationEngine.simplify_sentence() and normalise the result.

        The engine's simplify_sentence method returns a dict with keys
        'simplified', 'changes', etc. This method normalises that output
        into the format expected by _simplify_sentence_to_target.
        """
        try:
            result = self.engine.simplify_sentence(sentence)
        except Exception as exc:
            # If the engine fails on this sentence, return unchanged
            return {'simplified': sentence, 'changes': [], 'error': str(exc)}

        if result is None:
            return {'simplified': sentence, 'changes': []}

        simplified = result.get('simplified', sentence)
        raw_changes = result.get('changes', [])

        # Normalise change records to {original_word, replacement, similarity}
        normalised_changes = []
        for ch in raw_changes:
            if isinstance(ch, dict):
                normalised_changes.append({
                    'original_word': ch.get('original_word', ch.get('target_word', '?')),
                    'replacement': ch.get('replacement', '?'),
                    'similarity': ch.get('similarity', ch.get('sense_score', 0.0)),
                })
            elif isinstance(ch, (list, tuple)) and len(ch) >= 2:
                normalised_changes.append({
                    'original_word': ch[0],
                    'replacement': ch[1],
                    'similarity': ch[2] if len(ch) > 2 else 0.0,
                })

        return {'simplified': simplified, 'changes': normalised_changes}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_lite_engine():
    """Attempt to build a SimplificationEngine in lite mode using default asset paths."""
    try:
        from simplify_engine import SimplificationEngine
    except ImportError:
        print("ERROR: simplify_engine.py not found. Run from the BERT-LS-master directory.")
        sys.exit(1)

    # Try to find assets relative to script location
    base = os.path.dirname(os.path.abspath(__file__))
    freq_path = os.path.join(base, 'SUBTLEX_frequency.xlsx')
    ppdb_path = os.path.join(base, 'datasets', 'ppdb.db')

    if not os.path.isfile(freq_path):
        print(f"WARNING: Word frequency file not found at {freq_path}")
        freq_path = None
    if not os.path.isfile(ppdb_path):
        print(f"WARNING: PPDB file not found at {ppdb_path}")
        ppdb_path = None

    print("Loading SimplificationEngine (lite mode — this takes ~17s)...")
    engine = SimplificationEngine(
        word_frequency_path=freq_path,
        ppdb_path=ppdb_path,
    )
    return engine


def main():
    parser = argparse.ArgumentParser(
        description='Readability-Aware Lexical Simplification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python readability_analyzer.py "The physician administered the examination." --target-grade 8
  python readability_analyzer.py "Complex text." --analyze-only
  python readability_analyzer.py "Text." --json --target-grade 6
        """
    )
    parser.add_argument(
        'text', nargs='?',
        help='Text to analyze/simplify. Reads from stdin if omitted.'
    )
    parser.add_argument(
        '--target-grade', type=float, default=8.0,
        help='Target composite reading grade level (default: 8.0)'
    )
    parser.add_argument(
        '--max-passes', type=int, default=3,
        help='Maximum simplification passes per sentence (default: 3)'
    )
    parser.add_argument(
        '--analyze-only', action='store_true',
        help='Score the text without simplifying'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output results as JSON'
    )
    args = parser.parse_args()

    # Get input text
    if args.text:
        text = args.text
    else:
        print("Enter text to analyze (Ctrl+D / Ctrl+Z to finish):")
        text = sys.stdin.read().strip()

    if not text:
        print("ERROR: No text provided.")
        sys.exit(1)

    # Analyze-only mode: no engine needed
    if args.analyze_only:
        # Create a dummy analyzer just to score
        class _DummyEngine:
            pass
        analyzer = ReadabilityAnalyzer(_DummyEngine(), target_grade=args.target_grade)
        scores = analyzer.score_text(text)
        if args.json:
            print(json.dumps(scores.to_dict(), indent=2))
        else:
            print("\nREADABILITY SCORES")
            print("=" * 40)
            print(f"  Flesch-Kincaid Grade: {scores.flesch_kincaid_grade:.1f}")
            print(f"  SMOG Index:           {scores.smog_index:.1f}")
            print(f"  ARI:                  {scores.automated_readability_index:.1f}")
            print(f"  Coleman-Liau:         {scores.coleman_liau_index:.1f}")
            print(f"  Dale-Chall:           {scores.dale_chall_score:.1f}")
            print(f"  Flesch Reading Ease:  {scores.flesch_reading_ease:.1f}")
            print(f"  Composite Grade:      {scores.composite_grade:.1f}")
            print(f"  Interpretation:       {scores.interpretation}")
        return

    # Full simplification mode: load engine
    engine = _build_lite_engine()
    analyzer = ReadabilityAnalyzer(
        engine,
        target_grade=args.target_grade,
        max_passes=args.max_passes,
    )

    result = analyzer.simplify_to_target(text)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())
        print()
        print("Simplified text:")
        print(result.simplified_text)


if __name__ == '__main__':
    main()
