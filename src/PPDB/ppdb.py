

from .part_of_speech import PartOfSpeech
from .word_dictionary import WordDictionary

import sqlite3
import os


class Ppdb(WordDictionary):
    # http://paraphrase.org/#/download
    def __init__(self, dict_path):
        super().__init__(cache=True)

        self.dict_path = dict_path
        self.lang = 'eng'
        self.score_threshold = self.get_default_score_thresholds()
        self.is_synonym = True

        # Detect if path is SQLite DB or flat file
        if dict_path.endswith('.db') or (os.path.exists(dict_path) and self._is_sqlite(dict_path)):
            self._init_sqlite(dict_path)
        else:
            # Check if a .db version exists alongside the flat file
            db_path = dict_path + '.db'
            if os.path.exists(db_path):
                print(f"[PPDB] Found SQLite DB at {db_path}, using it instead of flat file.")
                self._init_sqlite(db_path)
            else:
                print(f"[PPDB] Loading flat file into memory (this may take several minutes)...")
                self._init_memory(dict_path)

    def _is_sqlite(self, path):
        """Check if file is a SQLite database."""
        try:
            with open(path, 'rb') as f:
                header = f.read(16)
                return header[:6] == b'SQLite'
        except:
            return False

    def _init_sqlite(self, db_path):
        """Initialize with SQLite backend (low memory)."""
        self.mode = 'sqlite'
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA cache_size=10000")
        print(f"[PPDB] SQLite mode: {db_path}")

    def _init_memory(self, dict_path):
        """Initialize with in-memory dict (original behavior)."""
        self.mode = 'memory'
        self.dict = {}
        self.read(dict_path)

    @classmethod
    def get_default_score_thresholds(cls):
        return {
            'AGigaSim': 0.6
        }

    def read(self, model_path):
        """Read flat PPDB file into memory dict (original behavior)."""
        with open(model_path, 'rb') as f:
            for line in f:
                line = line.decode('utf-8')

                if '\\ x' in line or 'xc3' in line:
                    continue

                fields = line.split('|||')
                constituents = fields[0].strip()[1:-1].split('/')
                phrase = fields[1].strip()
                paraphrase = fields[2].strip()

                # filter multiple words
                if len(phrase.split()) != len(paraphrase.split()):
                    continue

                scores = []

                if len(fields) == 6:
                    features = fields[3].strip().split()
                    features = [feature for feature in features for s in self.score_threshold if
                                s in feature]

                    for feature in features:
                        scheme, score = feature.split('=')
                        if scheme in self.score_threshold and float(score) > self.score_threshold[scheme]:
                            scores.append((scheme, score))

                if phrase not in self.dict:
                    self.dict[phrase] = {}

                part_of_speeches = [pos for con in constituents for pos in PartOfSpeech.constituent2pos(con)]

                for pos in part_of_speeches:
                    if pos not in self.dict[phrase]:
                        self.dict[phrase][pos] = []

                    self.dict[phrase][pos].append({
                        'phrase': phrase,
                        'part_of_speech': pos,
                        'synonym': paraphrase,
                        'scores': scores
                    })

    def predict(self, word, pos=None):
        """Get paraphrases for a word, optionally filtered by POS tag."""
        if self.mode == 'sqlite':
            return self._predict_sqlite(word, pos)
        else:
            return self._predict_memory(word, pos)

    def _predict_sqlite(self, word, pos=None):
        """SQLite-based lookup — returns synonyms only (backward compat)."""
        return [syn for syn, _ in self._predict_sqlite_scored(word, pos)]

    def _predict_sqlite_scored(self, word, pos=None):
        """SQLite-based lookup — returns (synonym, score) sorted by score desc."""
        cursor = self.conn.cursor()

        if pos is None:
            cursor.execute(
                'SELECT synonym, MAX(score) as best_score '
                'FROM ppdb WHERE phrase = ? '
                'GROUP BY synonym '
                'ORDER BY best_score DESC NULLS LAST',
                (word,)
            )
        else:
            # Also try the POS-expanded forms (e.g. 'a' covers JJ/JJR/JJS)
            expanded_pos = self._expand_pos(pos)
            placeholders = ','.join('?' * len(expanded_pos))
            cursor.execute(
                f'SELECT synonym, MAX(score) as best_score '
                f'FROM ppdb WHERE phrase = ? AND pos IN ({placeholders}) '
                f'GROUP BY synonym '
                f'ORDER BY best_score DESC NULLS LAST',
                (word, *expanded_pos)
            )

        return [(row[0], row[1] or 0.0) for row in cursor.fetchall()]

    @staticmethod
    def _expand_pos(pos):
        """Expand simplified POS tags to the full set stored in the DB."""
        expansions = {
            'a': ['a', 'JJ', 'JJR', 'JJS', 'IN'],
            's': ['s', 'a', 'JJ', 'JJR', 'JJS'],
            'r': ['r', 'RB', 'RBR', 'RBS'],
            'n': ['n', 'NN', 'NNS', 'NNP', 'NNPS', 'NP'],
            'v': ['v', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP'],
        }
        return expansions.get(pos, [pos])

    def _predict_memory(self, word, pos=None):
        """Original in-memory dict lookup."""
        if pos is None:
            candidates = []
            if word not in self.dict:
                return candidates

            for pos in self.dict[word]:
                for candidate in self.dict[word][pos]:
                    candidates.append(candidate['synonym'])

            return candidates

        if word in self.dict and pos in self.dict[word]:
            return [candidate['synonym'] for candidate in self.dict[word][pos]]

        return []

    def _predict_memory_scored(self, word, pos=None):
        """In-memory lookup — returns (synonym, score), sorted by score desc."""
        candidates = []
        if pos is None:
            if word not in self.dict:
                return candidates
            for p in self.dict[word]:
                for c in self.dict[word][p]:
                    score = float(c['scores'][0][1]) if c['scores'] else 0.0
                    candidates.append((c['synonym'], score))
        elif word in self.dict and pos in self.dict[word]:
            for c in self.dict[word][pos]:
                score = float(c['scores'][0][1]) if c['scores'] else 0.0
                candidates.append((c['synonym'], score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def predict_with_scores(self, word, pos=None):
        """Return (synonym, score) pairs sorted by paraphrase quality score desc."""
        if self.mode == 'sqlite':
            return self._predict_sqlite_scored(word, pos)
        else:
            return self._predict_memory_scored(word, pos)

    def pos_tag(self, tokens):
        import nltk
        return nltk.pos_tag(tokens)

    def close(self):
        """Close SQLite connection if open."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
