#!/usr/bin/env python3
"""
One-time script to convert PPDB 2.0 flat file to SQLite database.
Reduces memory from ~3-5GB (in-memory dict) to ~0MB (disk-based lookups).

Usage:
    python build_ppdb_sqlite.py

This reads `datasets/ppdb-2.0-tldr` and creates `datasets/ppdb.db`.
"""

import sqlite3
import os
import sys
import time

# Default paths
PPDB_INPUT = os.path.join(os.path.dirname(__file__), 'datasets', 'ppdb-2.0-tldr')
PPDB_OUTPUT = os.path.join(os.path.dirname(__file__), 'datasets', 'ppdb.db')


def build_sqlite(input_path, output_path):
    """Convert PPDB flat file to SQLite database."""

    if os.path.exists(output_path):
        print(f"[INFO] SQLite DB already exists at {output_path}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    print(f"[INFO] Reading PPDB from: {input_path}")
    print(f"[INFO] Writing SQLite to: {output_path}")

    # Remove existing DB if overwriting
    if os.path.exists(output_path):
        os.remove(output_path)

    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE ppdb (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phrase TEXT NOT NULL,
            pos TEXT NOT NULL,
            synonym TEXT NOT NULL,
            scheme TEXT,
            score REAL
        )
    ''')

    # Score threshold (same as original Ppdb class)
    score_threshold = {'AGigaSim': 0.6}

    start_time = time.time()
    row_count = 0
    skipped = 0
    batch = []
    batch_size = 10000

    file_size = os.path.getsize(input_path)
    bytes_read = 0

    with open(input_path, 'rb') as f:
        for line_num, line in enumerate(f, 1):
            bytes_read += len(line)

            try:
                line = line.decode('utf-8')
            except UnicodeDecodeError:
                skipped += 1
                continue

            if '\\ x' in line or 'xc3' in line:
                skipped += 1
                continue

            fields = line.split('|||')
            if len(fields) < 3:
                skipped += 1
                continue

            constituents = fields[0].strip()[1:-1].split('/')
            phrase = fields[1].strip()
            paraphrase = fields[2].strip()

            # Filter multiple words (must be same word count)
            if len(phrase.split()) != len(paraphrase.split()):
                skipped += 1
                continue

            # Extract scores if available
            best_scheme = None
            best_score = None

            if len(fields) == 6:
                features = fields[3].strip().split()
                features = [feat for feat in features for s in score_threshold if s in feat]

                for feature in features:
                    try:
                        scheme, score_str = feature.split('=')
                        score_val = float(score_str)
                        if scheme in score_threshold and score_val > score_threshold[scheme]:
                            best_scheme = scheme
                            best_score = score_val
                    except (ValueError, IndexError):
                        pass

            # Map constituents to POS tags
            from PPDB.part_of_speech import PartOfSpeech
            part_of_speeches = [pos for con in constituents for pos in PartOfSpeech.constituent2pos(con)]

            for pos in part_of_speeches:
                batch.append((phrase, pos, paraphrase, best_scheme, best_score))
                row_count += 1

            # Batch insert
            if len(batch) >= batch_size:
                cursor.executemany(
                    'INSERT INTO ppdb (phrase, pos, synonym, scheme, score) VALUES (?, ?, ?, ?, ?)',
                    batch
                )
                batch = []

                # Progress update
                if line_num % 500000 == 0:
                    pct = (bytes_read / file_size) * 100
                    elapsed = time.time() - start_time
                    print(f"  [{pct:.1f}%] {line_num:,} lines | {row_count:,} rows | {elapsed:.0f}s elapsed")

    # Insert remaining batch
    if batch:
        cursor.executemany(
            'INSERT INTO ppdb (phrase, pos, synonym, scheme, score) VALUES (?, ?, ?, ?, ?)',
            batch
        )

    # Create indexes for fast lookups
    print("[INFO] Creating indexes...")
    cursor.execute('CREATE INDEX idx_phrase_pos ON ppdb (phrase, pos)')
    cursor.execute('CREATE INDEX idx_phrase ON ppdb (phrase)')

    conn.commit()

    elapsed = time.time() - start_time
    db_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n[DONE] Conversion complete!")
    print(f"  Rows inserted: {row_count:,}")
    print(f"  Lines skipped: {skipped:,}")
    print(f"  DB size: {db_size:.1f} MB")
    print(f"  Time: {elapsed:.1f}s")

    conn.close()


if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else PPDB_INPUT
    output_path = sys.argv[2] if len(sys.argv) > 2 else PPDB_OUTPUT

    if not os.path.exists(input_path):
        print(f"[ERROR] PPDB file not found: {input_path}")
        sys.exit(1)

    build_sqlite(input_path, output_path)
