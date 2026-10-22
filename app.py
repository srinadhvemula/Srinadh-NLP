"""
Flask Web App for BERT-LS Lexical Simplification.

Usage:
    python app.py                    # Full mode (with FastText, ~8GB RAM)
    python app.py --skip-fasttext    # Lite mode (no FastText, ~2GB RAM)

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import time
import argparse
from flask import Flask, render_template, request, jsonify

# Add src/ to path — all modules and data files live there
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)

# CORS support — allow cross-origin requests for API usage
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Global engine instance (loaded once at startup)
engine = None
engine_loading = False
engine_error = None
skip_fasttext = False


def get_engine():
    """Lazy-load the simplification engine."""
    global engine, engine_loading, engine_error

    if engine is not None:
        return engine

    if engine_loading:
        return None

    engine_loading = True
    load_start = time.time()

    try:
        from simplify_engine import SimplificationEngine

        base_dir = os.path.join(os.path.dirname(__file__), 'src')

        # PPDB — prefer SQLite (299MB) over flat file (12GB)
        ppdb_sqlite = os.path.join(base_dir, 'datasets', 'ppdb.db')
        ppdb_flat = os.path.join(base_dir, 'datasets', 'ppdb-2.0-tldr')

        if os.path.exists(ppdb_sqlite):
            ppdb_path = ppdb_sqlite
            print(f"[STARTUP] Found PPDB SQLite: {ppdb_sqlite}")
        elif os.path.exists(ppdb_flat):
            ppdb_path = ppdb_flat
            print(f"[STARTUP] Found PPDB flat file: {ppdb_flat}")
        else:
            ppdb_path = None
            print("[STARTUP] WARNING — No PPDB found. Paraphrase candidates disabled.")

        # FastText embeddings (4.5GB file, ~8GB RAM)
        fasttext_path = None
        if not skip_fasttext:
            ft_file = os.path.join(base_dir, 'datasets', 'Fasttext', 'crawl-300d-2M-subword.vec')
            if os.path.exists(ft_file):
                fasttext_path = ft_file
                print(f"[STARTUP] Found FastText: {ft_file}")
                print("[STARTUP] FastText loading will take 5-10 minutes...")
            else:
                print("[STARTUP] WARNING — FastText file not found. Semantic ranking disabled.")
        else:
            print("[STARTUP] FastText SKIPPED (--skip-fasttext mode)")

        # Word frequency
        freq_path = os.path.join(base_dir, 'SUBTLEX_frequency.xlsx')
        if os.path.exists(freq_path):
            print(f"[STARTUP] Found word frequency: {freq_path}")
        else:
            freq_path = None
            print("[STARTUP] WARNING — Word frequency file not found.")

        engine = SimplificationEngine(
            bert_model='bert-base-uncased',
            word_embeddings_path=fasttext_path,
            word_frequency_path=freq_path,
            ppdb_path=ppdb_path,
            max_seq_length=128,
            num_selections=10,
            use_cuda=False
        )

        elapsed = time.time() - load_start
        engine_loading = False
        print(f"[STARTUP] Engine ready in {elapsed:.1f}s")
        return engine

    except Exception as e:
        engine_error = str(e)
        engine_loading = False
        print(f"[ERROR] Failed to load engine: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Check if the engine is loaded and ready."""
    if engine is not None:
        return jsonify({
            'status': 'ready',
            'device': str(engine.device),
            'has_fasttext': engine.fasttext_dico is not None,
            'has_ppdb': engine.ppdb_model is not None,
            'vocab_size': len(engine.word_count)
        })
    elif engine_loading:
        return jsonify({'status': 'loading'})
    elif engine_error:
        return jsonify({'status': 'error', 'message': engine_error})
    else:
        return jsonify({'status': 'not_started'})


@app.route('/api/simplify', methods=['POST'])
def simplify():
    """
    Simplify a specific word in a sentence.

    Request JSON: {"sentence": "...", "target_word": "..."}
    Response JSON: {"original": "...", "simplified": "...", "target_word": "...",
                     "replacement": "...", "candidates": [...], "changed": bool}
    """
    eng = get_engine()
    if eng is None:
        if engine_loading:
            return jsonify({'error': 'Model is still loading. Please wait...'}), 503
        return jsonify({'error': engine_error or 'Engine failed to initialize'}), 500

    data = request.get_json()
    if not data or 'sentence' not in data or 'target_word' not in data:
        return jsonify({'error': 'Missing "sentence" or "target_word" in request body'}), 400

    sentence = data['sentence'].strip()
    target_word = data['target_word'].strip()

    if not sentence or not target_word:
        return jsonify({'error': 'Empty sentence or target word'}), 400

    # Input length guard
    MAX_SENTENCE_LEN = 500
    if len(sentence) > MAX_SENTENCE_LEN:
        return jsonify({'error': f'Sentence too long (max {MAX_SENTENCE_LEN} characters)'}), 400

    start_time = time.time()
    try:
        result = eng.simplify_word(sentence, target_word)
        result['time_ms'] = round((time.time() - start_time) * 1000)
    except Exception as e:
        return jsonify({'error': f'Simplification failed: {str(e)}'}), 500

    return jsonify(result)


@app.route('/api/identify', methods=['POST'])
def identify():
    """
    Identify complex words in a sentence (step 1 of two-step flow).

    Request JSON: {"sentence": "..."}
    Response JSON: {"sentence": "...", "complex_words": [{"word": "...", "complexity": float, "index": int}]}
    """
    eng = get_engine()
    if eng is None:
        if engine_loading:
            return jsonify({'error': 'Model is still loading. Please wait...'}), 503
        return jsonify({'error': engine_error or 'Engine failed to initialize'}), 500

    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({'error': 'Missing "sentence" in request body'}), 400

    sentence = data['sentence'].strip()
    if not sentence:
        return jsonify({'error': 'Empty sentence'}), 400

    MAX_SENTENCE_LEN = 500
    if len(sentence) > MAX_SENTENCE_LEN:
        return jsonify({'error': f'Sentence too long (max {MAX_SENTENCE_LEN} characters)'}), 400

    start_time = time.time()
    try:
        result = eng.identify_complex_words(sentence)
        result['time_ms'] = round((time.time() - start_time) * 1000)
    except Exception as e:
        return jsonify({'error': f'Identification failed: {str(e)}'}), 500

    return jsonify(result)


@app.route('/api/simplify_auto', methods=['POST'])
def simplify_auto():
    """
    Auto-detect complex words and simplify the sentence (full pipeline).

    Request JSON: {"sentence": "..."}
    Response JSON: {"original": "...", "simplified": "...",
                     "complex_words": [...], "changes": [...]}
    """
    eng = get_engine()
    if eng is None:
        if engine_loading:
            return jsonify({'error': 'Model is still loading. Please wait...'}), 503
        return jsonify({'error': engine_error or 'Engine failed to initialize'}), 500

    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({'error': 'Missing "sentence" in request body'}), 400

    sentence = data['sentence'].strip()
    if not sentence:
        return jsonify({'error': 'Empty sentence'}), 400

    # Input length guard
    MAX_SENTENCE_LEN = 500
    if len(sentence) > MAX_SENTENCE_LEN:
        return jsonify({'error': f'Sentence too long (max {MAX_SENTENCE_LEN} characters)'}), 400

    start_time = time.time()
    try:
        result = eng.simplify_sentence(sentence)
        result['time_ms'] = round((time.time() - start_time) * 1000)
    except Exception as e:
        return jsonify({'error': f'Simplification failed: {str(e)}'}), 500

    return jsonify(result)


# ---------------------------------------------------------------------------
# Readability-Aware Simplification
# ---------------------------------------------------------------------------

_readability_analyzer = None


def get_readability_analyzer(target_grade=8.0):
    """Get or create a ReadabilityAnalyzer wrapping the loaded engine."""
    global _readability_analyzer
    eng = get_engine()
    if eng is None:
        return None
    if _readability_analyzer is None or _readability_analyzer.target_grade != target_grade:
        from readability_analyzer import ReadabilityAnalyzer
        _readability_analyzer = ReadabilityAnalyzer(eng, target_grade=target_grade)
    return _readability_analyzer


@app.route('/api/readability_analyze', methods=['POST'])
def readability_analyze():
    """
    Analyze readability of text without simplifying.

    Request JSON: {"text": "..."}
    Response JSON: {readability scores}
    """
    eng = get_engine()
    if eng is None:
        if engine_loading:
            return jsonify({'error': 'Model is still loading. Please wait...'}), 503
        return jsonify({'error': engine_error or 'Engine failed to initialize'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request body'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    if len(text) > 2000:
        return jsonify({'error': 'Text too long (max 2000 characters)'}), 400

    try:
        analyzer = get_readability_analyzer()
        scores = analyzer.score_text(text)
        return jsonify(scores.to_dict())
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/readability_simplify', methods=['POST'])
def readability_simplify():
    """
    Simplify text to a target grade level using readability-aware feedback loop.

    Request JSON: {"text": "...", "target_grade": 8.0}
    Response JSON: {original_text, simplified_text, before_scores, after_scores,
                     sentence_results, target_achieved, grade_improvement, ...}
    """
    eng = get_engine()
    if eng is None:
        if engine_loading:
            return jsonify({'error': 'Model is still loading. Please wait...'}), 503
        return jsonify({'error': engine_error or 'Engine failed to initialize'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request body'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    if len(text) > 2000:
        return jsonify({'error': 'Text too long (max 2000 characters)'}), 400

    target_grade = data.get('target_grade', 8.0)
    try:
        target_grade = float(target_grade)
        target_grade = max(1.0, min(16.0, target_grade))
    except (TypeError, ValueError):
        target_grade = 8.0

    start_time = time.time()
    try:
        analyzer = get_readability_analyzer(target_grade)
        result = analyzer.simplify_to_target(text)
        response = result.to_dict()
        response['time_ms'] = round((time.time() - start_time) * 1000)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'Readability simplification failed: {str(e)}'}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT-LS Web App')
    parser.add_argument('--skip-fasttext', action='store_true',
                        help='Skip loading FastText embeddings (saves ~8GB RAM, faster startup)')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    args = parser.parse_args()

    skip_fasttext = args.skip_fasttext

    print("=" * 60)
    print("  BERT-LS Lexical Simplification Web App")
    print("=" * 60)
    if skip_fasttext:
        print("  Mode: LITE (no FastText, ~2GB RAM)")
    else:
        print("  Mode: FULL (with FastText, ~8GB RAM)")
    print()
    print("  Loading pipeline components...")
    print()

    # Pre-load engine on startup
    get_engine()

    if engine is not None:
        print()
        print(f"  Server starting at http://localhost:{args.port}")
        print("=" * 60)
        app.run(host='0.0.0.0', port=args.port, debug=False)
    else:
        print()
        print("  [FATAL] Engine failed to load. Check errors above.")
        print("=" * 60)
        sys.exit(1)

