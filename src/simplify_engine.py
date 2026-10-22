"""
Simplification Engine — Core pipeline extracted from LSBert2.py.
Provides a SimplificationEngine class that can be used by the Flask web app.
"""

import os
import random
import math
import sys
import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import nltk
import openpyxl

from pathlib import Path
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity as cosine

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from PPDB import Ppdb

logger = logging.getLogger(__name__)


class SimplificationEngine:
    """
    Encapsulates the full BERT-LS lexical simplification pipeline.
    Load once, call simplify() many times.
    """

    # ------------------------------------------------------------------
    # Configuration constants (override via constructor or subclass)
    # ------------------------------------------------------------------
    BERT_TOP_K = 80               # Number of BERT vocabulary predictions to consider
    CONTEXT_WINDOW = 11           # Words around target for LM scoring context
    CWI_COMPLEXITY_THRESHOLD = 0.45  # Score above which a word is "complex"
    SEMANTIC_THRESHOLD = 0.55     # Wu-Palmer similarity floor for acceptance
    MAX_SENTENCE_CHARS = 500      # Input length guard
    MAX_AUTO_SIMPLIFICATIONS = 5  # Max complex words to simplify per sentence

    # Weighted Rank Fusion weights (tuned post-paper)
    W_BERT = 1.0   # BERT prediction order
    W_SIS  = 1.5   # Semantic similarity (most important for accuracy)
    W_CNT  = 0.5   # Word frequency
    W_LM   = 1.2   # Language model fluency
    W_PPDB = 0.5   # Paraphrase database membership

    def __init__(self, bert_model: str = 'bert-base-uncased',
                 word_embeddings_path: Optional[str] = None,
                 word_frequency_path: Optional[str] = None,
                 ppdb_path: Optional[str] = None,
                 max_seq_length: int = 128,
                 num_selections: int = 10,
                 prob_mask: float = 0.5,
                 use_cuda: bool = False):

        self.max_seq_length = max_seq_length
        self.num_selections = num_selections
        self.prob_mask = prob_mask
        self.ps = PorterStemmer()

        # LRU-style candidate cache: (sentence_lower, target_lower) -> result dict
        # Avoids re-running BERT for repeated identical queries (e.g. interactive UI).
        self._candidate_cache: dict = {}
        self._CACHE_MAX = 256  # keep memory bounded

        # Device
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        logger.info(f"Device: {self.device}")

        # Ensure NLTK data is available
        for resource, package in [
            ('tokenizers/punkt',                   'punkt'),
            ('tokenizers/punkt_tab',               'punkt_tab'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
            ('corpora/wordnet',                    'wordnet'),
            ('corpora/omw-1.4',                    'omw-1.4'),
        ]:
            try:
                nltk.data.find(resource)
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass

        # Load tokenizer and model
        logger.info(f"Loading BERT model: {bert_model}")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.model = self._load_bert_model(bert_model)
        self.model.to(self.device)
        self.model.eval()
        logger.info("BERT model loaded.")

        # Load FastText embeddings
        self.fasttext_dico = None
        self.fasttext_emb = None
        self.fasttext_index = {}   # word -> index for O(1) lookup
        if word_embeddings_path and os.path.exists(word_embeddings_path):
            logger.info(f"Loading FastText embeddings: {word_embeddings_path}")
            self.fasttext_dico, self.fasttext_emb = self._load_word_embeddings(word_embeddings_path)
            self.fasttext_index = {w: i for i, w in enumerate(self.fasttext_dico)}
            logger.info(f"FastText loaded: {len(self.fasttext_dico)} words (indexed)")
        else:
            logger.warning("No word embeddings loaded. Will use WordNet similarity fallback.")

        # Load word frequency
        self.word_count = {}
        if word_frequency_path and os.path.exists(word_frequency_path):
            logger.info(f"Loading word frequency: {word_frequency_path}")
            self.word_count = self._load_word_frequency(word_frequency_path)
            logger.info(f"Word frequency loaded: {len(self.word_count)} words")

        # Load PPDB
        self.ppdb_model = None
        if ppdb_path:
            logger.info(f"Loading PPDB: {ppdb_path}")
            self.ppdb_model = Ppdb(ppdb_path)
            logger.info("PPDB loaded.")

        logger.info("Initialization complete!")

    def _load_bert_model(self, bert_model):
        """Load BertForMaskedLM with fallback for broken download URLs."""
        # Try the bundled from_pretrained first
        model = BertForMaskedLM.from_pretrained(bert_model, output_attentions=True)
        if model is not None:
            return model

        logger.info("from_pretrained failed, using torch.hub fallback...")

        # Download config and weights via torch.hub (handles redirects properly)
        config_url = f"https://huggingface.co/{bert_model}/resolve/main/config.json"
        weights_url = f"https://huggingface.co/{bert_model}/resolve/main/pytorch_model.bin"

        import json as json_mod
        from pytorch_pretrained_bert.modeling import BertConfig

        # Download and parse config
        config_path = torch.hub.download_url_to_file(
            config_url,
            os.path.join(torch.hub.get_dir(), f'{bert_model}-config.json'),
            progress=True
        )
        config_file = os.path.join(torch.hub.get_dir(), f'{bert_model}-config.json')
        config = BertConfig.from_json_file(config_file)

        # Create model from config
        model = BertForMaskedLM(config, output_attentions=True)

        # Download weights
        logger.info("Downloading BERT weights (~440MB)...")
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url,
            model_dir=torch.hub.get_dir(),
            map_location='cpu',
            file_name=f'{bert_model}-pytorch_model.bin'
        )

        # Handle gamma/beta -> weight/bias renaming
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load state dict (allow missing keys for output_attentions extra params)
        model.load_state_dict(state_dict, strict=False)
        logger.info("BERT weights loaded via torch.hub fallback.")
        return model

    def _load_word_embeddings(self, path):
        """Load FastText/word2vec embeddings."""
        words = []
        We = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for n, line in enumerate(f):
                if n == 0:
                    continue  # Skip header
                parts = line.rstrip().split(' ', 1)
                if len(parts) != 2:
                    continue
                word, vect_str = parts
                vect = np.array(vect_str.split(), dtype=np.float32)
                if len(vect) > 0:
                    We.append(vect)
                    words.append(word)
        return words, We

    def _load_word_frequency(self, path):
        """Load SUBTLEX word frequency from xlsx."""
        word2count = {}
        if path.endswith('.xlsx'):
            xlsx_file = Path('', path)
            wb_obj = openpyxl.load_workbook(xlsx_file)
            sheet = wb_obj.active
            last_column = sheet.max_column - 1
            for i, row in enumerate(sheet.iter_rows(values_only=True)):
                if i == 0:
                    continue
                word2count[row[0]] = round(float(row[last_column]), 3)
        else:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        word2count[parts[0]] = float(parts[1])
        return word2count

    def _convert_sentence_to_token(self, sentence):
        """Tokenize sentence and track word positions."""
        tokenized_text = self.tokenizer.tokenize(sentence.lower())

        if len(tokenized_text) >= self.max_seq_length - 2:
            tokenized_text = tokenized_text[:self.max_seq_length - 3]

        nltk_sent = nltk.word_tokenize(sentence.lower())
        position2 = []
        token_index = 0
        start_pos = len(tokenized_text) + 2
        pre_word = ""

        for i, word in enumerate(nltk_sent):
            if token_index >= len(tokenized_text):
                break

            if word == "n't" and pre_word and pre_word[-1] == "n":
                word = "'t"

            if tokenized_text[token_index] == "\"":
                len_token = 2
            else:
                len_token = len(tokenized_text[token_index])

            if tokenized_text[token_index] == word or len_token >= len(word):
                position2.append(start_pos + token_index)
                pre_word = tokenized_text[token_index]
                token_index += 1
            else:
                new_pos = [start_pos + token_index]
                new_word = tokenized_text[token_index]

                while new_word != word and token_index < len(tokenized_text) - 1:
                    token_index += 1
                    new_word += tokenized_text[token_index].replace('##', '')
                    new_pos.append(start_pos + token_index)
                    if len(new_word) == len(word):
                        break
                token_index += 1
                pre_word = new_word
                position2.append(new_pos)

        return tokenized_text, nltk_sent, position2

    def _convert_token_to_feature(self, tokens_a, mask_position):
        """Create masked LM features for a single token position."""
        tokens = ["[CLS]"]
        input_type_ids = [0]

        len_tokens = len(tokens_a)
        num_mask = max(1, int(self.prob_mask * len_tokens))
        first_sentence_mask_random = random.sample(range(0, len_tokens), min(num_mask, len_tokens))

        for i in range(len_tokens):
            if i == (mask_position - len_tokens - 2):
                tokens.append(tokens_a[i])
            elif i in first_sentence_mask_random:
                tokens.append('[MASK]')
            else:
                tokens.append(tokens_a[i])
            input_type_ids.append(0)

        tokens.append("[SEP]")
        input_type_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(1)

        tokens.append("[SEP]")
        input_type_ids.append(1)

        tokens[mask_position] = '[MASK]'

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        # Truncate if too long
        input_ids = input_ids[:self.max_seq_length]
        input_mask = input_mask[:self.max_seq_length]
        input_type_ids = input_type_ids[:self.max_seq_length]

        return input_ids, input_mask, input_type_ids

    def _convert_whole_word_to_feature(self, tokens_a, mask_position):
        """Create masked LM features for a multi-token word."""
        tokens = ["[CLS]"]
        input_type_ids = [0]

        len_tokens = len(tokens_a)
        num_mask = max(1, int(self.prob_mask * len_tokens))
        first_sentence_mask_random = random.sample(range(0, len_tokens), min(num_mask, len_tokens))

        mask_index = [pos - len_tokens - 2 for pos in mask_position]

        for i in range(len_tokens):
            if i in mask_index:
                tokens.append(tokens_a[i])
            elif i in first_sentence_mask_random:
                tokens.append('[MASK]')
            else:
                tokens.append(tokens_a[i])
            input_type_ids.append(0)

        tokens.append("[SEP]")
        input_type_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(1)

        tokens.append("[SEP]")
        input_type_ids.append(1)

        # Mask the target positions (reverse order to preserve indices)
        mask_position_length = len(mask_position)
        count = 0
        while count in range(mask_position_length):
            index = mask_position_length - 1 - count
            pos = mask_position[index]
            if index == 0:
                tokens[pos] = '[MASK]'
            else:
                if pos < len(tokens):
                    del tokens[pos]
                    del input_type_ids[pos]
            count += 1

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        input_ids = input_ids[:self.max_seq_length]
        input_mask = input_mask[:self.max_seq_length]
        input_type_ids = input_type_ids[:self.max_seq_length]

        return input_ids, input_mask, input_type_ids

    def _bert_candidate_generation(self, source_word, pre_tokens, pre_scores, num_selection=10):
        """Generate candidate substitutions from BERT predictions."""
        cur_tokens = []
        source_stem = self.ps.stem(source_word)

        for i in range(len(pre_tokens)):
            token = pre_tokens[i]
            if token[:2] == "##":
                continue
            if token == source_word:
                continue

            token_stem = self.ps.stem(token)
            if token_stem == source_stem:
                continue
            if len(token_stem) >= 3 and token_stem[:3] == source_stem[:3]:
                continue

            cur_tokens.append(token)
            if len(cur_tokens) == num_selection:
                break

        if len(cur_tokens) == 0:
            cur_tokens = pre_tokens[:num_selection + 1]

        return cur_tokens

    def _multi_mask_bert_prediction(self, tokens_a, mask_position, context_window=2):
        """
        Multi-mask BERT ensemble prediction.

        Runs multiple BERT forward passes and averages the raw logit
        distributions at the target position:
          - Pass 0: standard single-mask (target word only, in segment B)
          - Passes 1..K: same as pass 0 but additionally mask one context
            word at a time in segment A (the visible first sentence).

        Masking context words in segment A forces BERT to rely on broader
        syntactic/semantic structure rather than the immediate neighbour
        tokens, which reduces frequency bias (e.g. "gave" beating "performed").

        Args:
            tokens_a:       List of BERT token strings for the sentence.
            mask_position:  int (single-token word) or list[int] (multi-token).
            context_window: Number of positions on each side to mask (default 2).

        Returns:
            pre_tokens:         list[str]  — top-80 tokens from averaged distribution.
            pre_scores:         np.ndarray — corresponding averaged logit values.
            single_pre_tokens:  list[str]  — top-5 tokens from baseline single-mask
                                             (kept for comparison output).
        """
        mask_id = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        # Build the standard masked feature and determine segment-B target position
        if isinstance(mask_position, list):
            input_ids, input_mask, input_type_ids = self._convert_whole_word_to_feature(
                tokens_a, mask_position)
            target_pos_B = mask_position[0]
            target_pos_A = mask_position[0] - len(tokens_a) - 1
        else:
            input_ids, input_mask, input_type_ids = self._convert_token_to_feature(
                tokens_a, mask_position)
            target_pos_B = mask_position
            # Segment A occupies positions 1..len(tokens_a);
            # start_pos in _convert_sentence_to_token = len(tokens_a)+2,
            # so token index i in segment B maps to position 1+i in segment A.
            target_pos_A = mask_position - len(tokens_a) - 1

        tokens_tensor  = torch.tensor([input_ids]).to(self.device)
        type_ids_tensor = torch.tensor([input_type_ids]).to(self.device)
        attn_mask_tensor = torch.tensor([input_mask]).to(self.device)

        # --- Pass 0: baseline single-mask ---
        with torch.no_grad():
            _, prediction_scores = self.model(tokens_tensor, type_ids_tensor,
                                              attn_mask_tensor)

        baseline_logits = prediction_scores[0, target_pos_B].cpu().numpy()
        all_logits = [baseline_logits]

        # Save single-mask top-5 for comparison (before any ensemble averaging)
        single_top = torch.tensor(baseline_logits, dtype=torch.float32).topk(5)
        single_pre_tokens = self.tokenizer.convert_ids_to_tokens(
            single_top[1].cpu().numpy())

        # --- Passes 1..K: additionally mask context words in segment A ---
        first_sent_start = 1
        first_sent_end   = len(tokens_a)          # inclusive

        for offset in range(-context_window, context_window + 1):
            if offset == 0:
                continue
            ctx_pos_A = target_pos_A + offset
            if ctx_pos_A < first_sent_start or ctx_pos_A > first_sent_end:
                continue
            # Skip positions that are already [MASK] (from prob_mask randomisation)
            # or zero-padded
            if input_ids[ctx_pos_A] == mask_id or input_ids[ctx_pos_A] == 0:
                continue

            ctx_input_ids = list(input_ids)       # shallow copy
            ctx_input_ids[ctx_pos_A] = mask_id

            ctx_tensor = torch.tensor([ctx_input_ids]).to(self.device)

            with torch.no_grad():
                _, ctx_scores = self.model(ctx_tensor, type_ids_tensor,
                                           attn_mask_tensor)

            all_logits.append(ctx_scores[0, target_pos_B].cpu().numpy())

        # Average logits across all passes (before softmax for stability)
        avg_logits   = np.mean(all_logits, axis=0)
        avg_tensor   = torch.tensor(avg_logits, dtype=torch.float32)
        predicted_top = avg_tensor.topk(80)

        pre_tokens = self.tokenizer.convert_ids_to_tokens(
            predicted_top[1].cpu().numpy())
        pre_scores = predicted_top[0].cpu().numpy()

        return pre_tokens, pre_scores, single_pre_tokens

    # ------------------------------------------------------------------
    # Semantic similarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_wordnet_similarity(word1, word2, pos=None):
        """
        Return the best WordNet Wu-Palmer similarity between word1 and word2.
        Wu-Palmer is preferred over path_similarity because it accounts for
        depth in the taxonomy tree, producing more meaningful scores.
        Returns float in [0, 1] or 0.0 if no synsets found.
        """
        synsets1 = wn.synsets(word1, pos=pos) if pos else wn.synsets(word1)
        synsets2 = wn.synsets(word2, pos=pos) if pos else wn.synsets(word2)

        if not synsets1 or not synsets2:
            return 0.0

        best = 0.0
        for s1 in synsets1:
            for s2 in synsets2:
                try:
                    sim = s1.wup_similarity(s2)
                    if sim and sim > best:
                        best = sim
                except Exception:
                    pass
        return best

    @staticmethod
    def _is_antonym(word1, word2):
        """
        Check if word2 is an antonym of word1 using WordNet antonym relations.
        Returns True if any synset/lemma of word1 lists word2 as an antonym.
        """
        for syn in wn.synsets(word1):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    if ant.name().lower() == word2.lower():
                        return True
        # Also check reverse direction
        for syn in wn.synsets(word2):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    if ant.name().lower() == word1.lower():
                        return True
        return False

    @staticmethod
    def _shares_synset(word1, word2, pos=None):
        """
        Check if word1 and word2 appear together in at least one synset,
        meaning they are true synonyms in at least one sense.
        """
        synsets1 = wn.synsets(word1, pos=pos) if pos else wn.synsets(word1)
        for syn in synsets1:
            lemma_names = [l.name().lower() for l in syn.lemmas()]
            if word2.lower() in lemma_names:
                return True
        return False

    @staticmethod
    def _nltk_pos_to_wn(nltk_tag):
        """Map an NLTK POS tag to a WordNet POS constant."""
        if nltk_tag.startswith('V'):
            return wn.VERB
        if nltk_tag.startswith('N'):
            return wn.NOUN
        if nltk_tag.startswith('J'):
            return wn.ADJ
        if nltk_tag.startswith('R'):
            return wn.ADV
        return None

    def _get_bert_embedding(self, word: str):
        """
        Return the static BERT token embedding for a word (averaged over
        its WordPiece subwords).  Uses the already-loaded BERT embedding
        matrix — zero additional RAM or disk cost.
        Returns a (1, hidden) numpy array, or None if tokenisation fails.
        """
        tokens = self.tokenizer.tokenize(word)
        if not tokens:
            return None
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        id_tensor = torch.tensor(token_ids, device=self.device)
        with torch.no_grad():
            emb = self.model.bert.embeddings.word_embeddings(id_tensor)
        return emb.mean(0).unsqueeze(0).cpu().numpy()   # shape (1, 768)

    # ------------------------------------------------------------------

    def _cross_entropy_word(self, X, i, pos):
        """Calculate cross-entropy loss for a word position."""
        X = softmax(X, axis=1)
        loss = -np.log10(X[i, pos])
        return loss

    def _get_score(self, sentence):
        """Get perplexity-like score for a sentence."""
        tokenize_input = self.tokenizer.tokenize(sentence)
        len_sen = len(tokenize_input)

        tokenize_input.insert(0, '[CLS]')
        tokenize_input.append('[SEP]')
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenize_input)

        sentence_loss = 0
        for i, word in enumerate(tokenize_input):
            if word == '[CLS]' or word == '[SEP]':
                continue

            original_word = tokenize_input[i]
            tokenize_input[i] = '[MASK]'
            mask_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
            mask_input = mask_input.to(self.device)

            with torch.no_grad():
                att, pre_word = self.model(mask_input)
            word_loss = self._cross_entropy_word(pre_word[0].cpu().numpy(), i, input_ids[i])
            sentence_loss += word_loss
            tokenize_input[i] = original_word

        return np.exp(sentence_loss / max(len_sen, 1))

    def _lm_score(self, source_word, source_context, substitution_selection):
        """Calculate LM scores for substitutions in context."""
        source_sentence = ' '.join(source_context).strip()
        LM = []
        source_loss = self._get_score(source_sentence)

        for sub in substitution_selection:
            # FIX: replace only the FIRST occurrence to avoid corrupting
            # sentences where the target word appears multiple times
            sub_sentence = source_sentence.replace(source_word, sub, 1)
            score = self._get_score(sub_sentence)
            LM.append(score)

        return LM, source_loss

    def _preprocess_sr(self, source_word, substitution_selection, ppdb_scores=None):
        """
        Filter candidates by frequency and compute semantic similarity.
        Uses FastText cosine similarity when available, falls back to
        WordNet Wu-Palmer similarity otherwise.
        ppdb_scores: optional dict {synonym: score} from PPDB quality scores.
        """
        ss = []
        sis_scores = []
        count_scores = []
        ppdb_score_list = []

        # Use O(1) dict lookup instead of O(N) list.index()
        isFast = self.fasttext_index and source_word in self.fasttext_index
        if isFast:
            source_idx = self.fasttext_index[source_word]
            source_emb = np.array(self.fasttext_emb[source_idx]).reshape(1, -1)
        else:
            # BERT static-embedding fallback — much better than Wu-Palmer.
            # Computed once for the source word, reused for every candidate.
            _bert_src_emb = self._get_bert_embedding(source_word)

        ppdb_scores = ppdb_scores or {}

        for sub in substitution_selection:
            if sub not in self.word_count:
                continue
            sub_count = self.word_count[sub]
            if sub_count <= 3:
                continue

            # Reject known antonyms early
            if self._is_antonym(source_word, sub):
                continue

            if isFast:
                if sub not in self.fasttext_index:
                    continue
                sub_idx = self.fasttext_index[sub]
                sub_emb = np.array(self.fasttext_emb[sub_idx]).reshape(1, -1)
                sis = cosine(source_emb, sub_emb)
                sis_scores.append(sis)
            else:
                # BERT static-embedding cosine similarity — better than Wu-Palmer.
                # Falls back to Wu-Palmer only if BERT tokenisation fails for either word.
                if _bert_src_emb is not None:
                    sub_emb = self._get_bert_embedding(sub)
                    if sub_emb is not None:
                        sis = cosine(_bert_src_emb, sub_emb)
                    else:
                        wn_sim = self._get_wordnet_similarity(source_word, sub)
                        sis = np.array([[max(wn_sim, 0.01)]])
                else:
                    wn_sim = self._get_wordnet_similarity(source_word, sub)
                    sis = np.array([[max(wn_sim, 0.01)]])
                sis_scores.append(sis)

            ss.append(sub)
            count_scores.append(sub_count)
            ppdb_score_list.append(ppdb_scores.get(sub, 0.0))

        return ss, sis_scores, count_scores, ppdb_score_list

    def _extract_context(self, words, mask_index, window=11):
        """Extract context window around target word."""
        length = len(words)
        half = window // 2

        if length <= window:
            return words
        elif half <= mask_index < length - half:
            return words[mask_index - half:mask_index + half + 1]
        elif mask_index < half:
            return words[:window]
        else:
            return words[length - window:length]

    def _preprocess_tag(self, tag):
        """Map POS tag to simplified form."""
        if tag[0] in ("V", "N"):
            return tag
        if tag[0] == "R":
            return "r"
        if tag[0] in ("J", "I"):
            return 'a'
        return 's'

    def simplify_word(self, sentence: str, target_word: str,
                      use_multi_mask: bool = True) -> Dict[str, Any]:
        """
        Simplify a specific word in a sentence.

        Args:
            sentence:        Input sentence string.
            target_word:     The word to simplify.
            use_multi_mask:  When True (default), use the multi-mask BERT ensemble
                             to generate candidates instead of single-mask.  The
                             ensemble averages logit distributions across multiple
                             forward passes where context words in segment A are
                             additionally masked, reducing frequency bias.

        Returns:
            dict with keys: original, simplified, target_word, replacement,
                  candidates, changed, sense_preserved, semantic_similarity,
                  single_mask_top5 (baseline BERT top-5 when use_multi_mask=True),
                  multi_mask_top5  (ensemble top-5 when use_multi_mask=True).
        """
        # Check cache before running BERT
        _cache_key = (sentence.lower(), target_word.lower())
        if _cache_key in self._candidate_cache:
            return self._candidate_cache[_cache_key]

        try:
            tokens, words, position = self._convert_sentence_to_token(sentence)
        except (AssertionError, Exception) as e:
            return {
                'original': sentence,
                'simplified': sentence,
                'target_word': target_word,
                'replacement': target_word,
                'candidates': [],
                'changed': False,
                'error': str(e)
            }

        if len(words) != len(position):
            return {
                'original': sentence, 'simplified': sentence,
                'target_word': target_word, 'replacement': target_word,
                'candidates': [], 'changed': False,
                'error': 'Token/position mismatch'
            }

        # Find target word in tokenized sentence
        target_lower = target_word.lower()
        if target_lower not in words:
            return {
                'original': sentence, 'simplified': sentence,
                'target_word': target_word, 'replacement': target_word,
                'candidates': [], 'changed': False,
                'error': f'Word "{target_word}" not found in tokenized sentence'
            }

        mask_index = words.index(target_lower)
        mask_context = self._extract_context(words, mask_index)
        mask_position = position[mask_index]

        # Generate features and get BERT predictions
        # Multi-mask ensemble (default) gives richer contextual signal and
        # reduces frequency bias vs. single-mask.
        single_mask_top5 = None
        multi_mask_top5  = None

        if use_multi_mask:
            pre_tokens, pre_scores_arr, single_mask_top5 = \
                self._multi_mask_bert_prediction(tokens, mask_position, context_window=3)
            multi_mask_top5 = list(pre_tokens[:5])
        else:
            if isinstance(mask_position, list):
                input_ids, input_mask, input_type_ids = self._convert_whole_word_to_feature(
                    tokens, mask_position)
            else:
                input_ids, input_mask, input_type_ids = self._convert_token_to_feature(
                    tokens, mask_position)

            tokens_tensor  = torch.tensor([input_ids]).to(self.device)
            token_type_ids = torch.tensor([input_type_ids]).to(self.device)
            attention_mask = torch.tensor([input_mask]).to(self.device)

            with torch.no_grad():
                _, prediction_scores = self.model(tokens_tensor, token_type_ids,
                                                   attention_mask)

            if isinstance(mask_position, list):
                predicted_top = prediction_scores[0, mask_position[0]].topk(80)
            else:
                predicted_top = prediction_scores[0, mask_position].topk(80)

            pre_tokens    = self.tokenizer.convert_ids_to_tokens(predicted_top[1].cpu().numpy())
            pre_scores_arr = predicted_top[0].cpu().numpy()

        # BERT candidate generation
        cgBERT = self._bert_candidate_generation(target_lower, pre_tokens,
                                                  pre_scores_arr,
                                                  self.num_selections)

        # Get PPDB candidates (with quality scores)
        cgPPDB = []
        ppdb_score_map = {}    # {synonym: quality_score}
        raw_nltk_tag = 'NN'    # fallback
        if self.ppdb_model:
            words_tag = nltk.pos_tag(words)
            if mask_index < len(words_tag):
                raw_nltk_tag = words_tag[mask_index][1]
                complex_word_tag = self._preprocess_tag(raw_nltk_tag)
                scored = self.ppdb_model.predict_with_scores(target_lower, complex_word_tag)
                for syn, sc in scored:
                    cgPPDB.append(syn)
                    ppdb_score_map[syn] = sc

        # Substitution ranking (BERT candidates, weighted by PPDB quality scores)
        ss, sis_scores, count_scores, ppdb_score_list = self._preprocess_sr(
            target_lower, cgBERT, ppdb_scores=ppdb_score_map
        )

        if len(ss) == 0:
            return {
                'original': sentence, 'simplified': sentence,
                'target_word': target_word, 'replacement': target_word,
                'candidates': cgBERT[:5], 'changed': False,
                'sense_preserved': True
            }

        # ------------------------------------------------------------------
        # Weighted Rank Fusion
        # Uses class-level weights (W_BERT, W_SIS, W_CNT, W_LM, W_PPDB)
        # that emphasise semantic similarity and LM fluency over raw BERT order.
        # ------------------------------------------------------------------

        if len(sis_scores) > 0 and len(sis_scores) == len(ss):
            sis_rank = self._rank_with_ties(sis_scores, reverse=True)
        else:
            sis_rank = None

        count_rank = self._rank_with_ties(count_scores, reverse=True)

        lm_score, source_lm = self._lm_score(target_lower, mask_context, ss)
        lm_rank = self._rank_with_ties(lm_score, reverse=False)

        bert_rank = list(range(1, len(ss) + 1))

        # PPDB rank: reward both membership AND high quality score
        max_ppdb = max(ppdb_score_list) if ppdb_score_list and max(ppdb_score_list) > 0 else 1.0
        ppdb_rank = [
            max(1, int((1 - (ppdb_score_list[i] / max_ppdb)) * (len(ss) // 2)))
            if ppdb_score_list[i] > 0 else len(ss) // 3
            for i in range(len(ss))
        ]

        if sis_rank is not None:
            all_ranks = [
                self.W_BERT*b + self.W_SIS*s + self.W_CNT*c + self.W_LM*l + self.W_PPDB*p
                for b, s, c, l, p
                in zip(bert_rank, sis_rank, count_rank, lm_rank, ppdb_rank)
            ]
        else:
            all_ranks = [
                self.W_BERT*b + self.W_CNT*c + self.W_LM*l + self.W_PPDB*p
                for b, c, l, p
                in zip(bert_rank, count_rank, lm_rank, ppdb_rank)
            ]

        pre_index = all_ranks.index(min(all_ranks))
        pre_count = count_scores[pre_index]

        source_count = self.word_count.get(target_lower, 0)
        pre_lm = lm_score[pre_index]

        if source_lm > pre_lm or pre_count > source_count:
            chosen = ss[pre_index]
        else:
            chosen = target_lower

        # ------------------------------------------------------------------
        # Semantic Guard (Wu-Palmer + Antonym Detection)
        # Before committing a replacement, verify:
        #   1. The substitute is NOT an antonym of the source word
        #   2. The substitute shares the same conceptual domain
        # Threshold raised to 0.55 for better accuracy.
        # ------------------------------------------------------------------
        SEMANTIC_THRESHOLD = self.SEMANTIC_THRESHOLD
        wn_pos = self._nltk_pos_to_wn(raw_nltk_tag)
        sense_preserved = True
        replacement = target_lower          # default: keep original
        best_sim = 0.0

        if chosen != target_lower:
            # Build candidate ranking order for the guard
            order = sorted(range(len(ss)), key=lambda i: all_ranks[i])
            for idx in order:
                candidate = ss[idx]
                # Accept if LM or frequency qualifies AND concept is preserved
                if lm_score[idx] >= source_lm and count_scores[idx] <= source_count:
                    continue
                # REJECT antonyms immediately
                if self._is_antonym(target_lower, candidate):
                    continue
                sim = self._get_wordnet_similarity(target_lower, candidate, pos=wn_pos)
                # Bonus: if candidate shares a synset, it's a true synonym
                if self._shares_synset(target_lower, candidate, pos=wn_pos):
                    replacement = candidate
                    sense_preserved = True
                    best_sim = max(sim, 0.9)
                    break
                if sim >= SEMANTIC_THRESHOLD:
                    replacement = candidate
                    sense_preserved = True
                    best_sim = sim
                    break
                elif sim == 0.0:
                    # WordNet has no entry → rely on BERT context, but only
                    # if this is the top-ranked candidate
                    if idx == order[0]:
                        replacement = candidate
                        sense_preserved = True  # trust BERT
                        best_sim = 0.0
                        break

        # ------------------------------------------------------------------
        # Morphological post-processing: preserve tense/number
        # ------------------------------------------------------------------
        if replacement != target_lower:
            replacement = self._match_morphology(target_lower, replacement, raw_nltk_tag)

        # Build simplified sentence
        simplified = sentence
        if replacement != target_lower:
            import re as re_mod
            pattern = re_mod.compile(re_mod.escape(target_word), re_mod.IGNORECASE)
            simplified = pattern.sub(replacement, sentence, count=1)

        final_sim = best_sim if best_sim > 0 else (
            round(self._get_wordnet_similarity(target_lower, replacement, pos=wn_pos), 3)
            if replacement != target_lower else 1.0
        )

        result = {
            'original': sentence,
            'simplified': simplified,
            'target_word': target_word,
            'replacement': replacement,
            'candidates': ss[:10],
            'changed': replacement != target_lower,
            'sense_preserved': sense_preserved,
            'semantic_similarity': final_sim,
        }
        if single_mask_top5 is not None:
            result['single_mask_top5'] = list(single_mask_top5)
        if multi_mask_top5 is not None:
            result['multi_mask_top5'] = multi_mask_top5

        # Store in cache; evict oldest entry when limit is reached
        if len(self._candidate_cache) >= self._CACHE_MAX:
            self._candidate_cache.pop(next(iter(self._candidate_cache)))
        self._candidate_cache[_cache_key] = result
        return result

    @staticmethod
    def _rank_with_ties(values: list, reverse: bool = True) -> List[int]:
        """
        Compute ordinal ranks for a list of values, with proper tie-breaking.
        Tied values get consecutive ranks (1, 2, 3) instead of all getting
        the same rank, which prevents rank fusion bias.

        Args:
            values:  List of numeric values to rank.
            reverse: If True, highest value gets rank 1 (for scores where higher is better).
                     If False, lowest value gets rank 1 (for scores where lower is better).

        Returns:
            List of integer ranks (1-indexed), same length as values.
        """
        indexed = list(enumerate(values))
        indexed.sort(key=lambda x: x[1], reverse=reverse)
        ranks = [0] * len(values)
        for rank, (original_idx, _) in enumerate(indexed, 1):
            ranks[original_idx] = rank
        return ranks

    @staticmethod
    def _match_morphology(source: str, replacement: str, pos_tag: str) -> str:
        """
        Attempt to match the morphological form (tense, number) of the
        replacement to the source word.

        Uses simple heuristic rules plus an irregular verb lookup.
        Prevents double-inflection of already-inflected words.

        Key fix: When the source word uses regular past tense (-ed), we prefer
        regular -ed inflection on the candidate rather than accepting an
        unrelated irregular past form (e.g., "took" for "administered").
        """
        if source == replacement:
            return replacement

        # Common irregular past tenses: base -> past
        IRREGULAR_PAST = {
            'be': 'was', 'begin': 'began', 'break': 'broke', 'bring': 'brought',
            'build': 'built', 'buy': 'bought', 'catch': 'caught', 'choose': 'chose',
            'come': 'came', 'do': 'did', 'draw': 'drew', 'drink': 'drank',
            'drive': 'drove', 'eat': 'ate', 'fall': 'fell', 'feel': 'felt',
            'find': 'found', 'fly': 'flew', 'forget': 'forgot', 'get': 'got',
            'give': 'gave', 'go': 'went', 'grow': 'grew', 'have': 'had',
            'hear': 'heard', 'hold': 'held', 'keep': 'kept', 'know': 'knew',
            'lead': 'led', 'leave': 'left', 'lose': 'lost', 'make': 'made',
            'mean': 'meant', 'meet': 'met', 'pay': 'paid', 'put': 'put',
            'read': 'read', 'ride': 'rode', 'run': 'ran', 'say': 'said',
            'see': 'saw', 'sell': 'sold', 'send': 'sent', 'set': 'set',
            'show': 'showed', 'sit': 'sat', 'speak': 'spoke', 'spend': 'spent',
            'stand': 'stood', 'take': 'took', 'teach': 'taught', 'tell': 'told',
            'think': 'thought', 'try': 'tried', 'understand': 'understood',
            'wear': 'wore', 'win': 'won', 'write': 'wrote',
        }
        # All known irregular past / past participle forms
        PAST_FORMS = set(IRREGULAR_PAST.values()) | {
            'been', 'begun', 'broken', 'brought', 'built', 'bought', 'caught',
            'chosen', 'come', 'done', 'drawn', 'drunk', 'driven', 'eaten',
            'fallen', 'felt', 'found', 'flown', 'forgotten', 'gotten', 'given',
            'gone', 'grown', 'had', 'heard', 'held', 'kept', 'known', 'led',
            'left', 'lost', 'made', 'meant', 'met', 'paid', 'put', 'read',
            'ridden', 'run', 'said', 'seen', 'sold', 'sent', 'set', 'shown',
            'sat', 'spoken', 'spent', 'stood', 'taken', 'taught', 'told',
            'thought', 'tried', 'understood', 'worn', 'won', 'written',
        }

        # ------------------------------------------------------------------
        # GUARD 1: If replacement is already inflected (regular past, progressive,
        # OR any known irregular past/participial form), do NOT re-inflect.
        # This prevents double-inflection bugs like "eaten" → "eatenned".
        # ------------------------------------------------------------------
        if replacement.endswith('ed') or replacement.endswith('ing') or replacement in PAST_FORMS:
            return replacement  # Already inflected, leave as-is

        # Past tense: source ends in -ed or is an irregular past form
        if pos_tag.startswith('VB'):
            source_is_past = source.endswith('ed') or pos_tag in ('VBD', 'VBN')

            if source_is_past:
                if replacement in IRREGULAR_PAST:
                    # Convert base form to past tense via lookup
                    replacement = IRREGULAR_PAST[replacement]
                elif replacement.endswith('e'):
                    replacement = replacement + 'd'
                elif (replacement.endswith('y')
                      and len(replacement) > 1
                      and replacement[-2] not in 'aeiou'):
                    # consonant + y → ied  (carry→carried, apply→applied)
                    replacement = replacement[:-1] + 'ied'
                elif (len(replacement) > 2
                      and replacement[-1] not in 'aeiou'
                      and replacement[-2] in 'aeiou'
                      and replacement[-3:] not in ('ght', 'nce', 'rse', 'use', 'ase', 'ise', 'ose')):
                    # CVC doubling rule — but exclude common exceptions
                    replacement = replacement + replacement[-1] + 'ed'
                else:
                    replacement = replacement + 'ed'
            elif source.endswith('ing') and not replacement.endswith('ing'):
                if replacement.endswith('e'):
                    replacement = replacement[:-1] + 'ing'
                else:
                    replacement = replacement + 'ing'
            elif source.endswith('s') and not source.endswith('ss') and not replacement.endswith('s'):
                if pos_tag == 'VBZ':
                    if replacement.endswith(('s', 'x', 'z', 'ch', 'sh')):
                        replacement = replacement + 'es'
                    else:
                        replacement = replacement + 's'

        # Plural nouns: source ends in -s/-es
        elif pos_tag.startswith('NN'):
            if pos_tag == 'NNS' and not replacement.endswith('s'):
                if replacement.endswith(('s', 'x', 'z', 'ch', 'sh')):
                    replacement = replacement + 'es'
                elif replacement.endswith('y') and len(replacement) > 1 and replacement[-2] not in 'aeiou':
                    replacement = replacement[:-1] + 'ies'
                else:
                    replacement = replacement + 's'

        # Comparative/superlative adjectives
        elif pos_tag == 'JJR' and source.endswith('er') and not replacement.endswith('er'):
            if replacement.endswith('e'):
                replacement = replacement + 'r'
            else:
                replacement = replacement + 'er'
        elif pos_tag == 'JJS' and source.endswith('est') and not replacement.endswith('est'):
            if replacement.endswith('e'):
                replacement = replacement + 'st'
            else:
                replacement = replacement + 'est'

        return replacement

    def _identify_complex_words(self, words):
        """
        Identify complex words in a tokenized word list.
        Uses trained CWI model when available, otherwise a multi-signal
        frequency + length + syllable heuristic.

        Returns:
            list of (word, complexity_score, word_index) sorted by complexity desc.
        """
        # Skip function words (closed-class POS tags)
        SKIP_POS = {'DT', 'IN', 'CC', 'PRP', 'PRP$', 'WDT', 'WP', 'WP$',
                     'TO', 'MD', 'EX', 'PDT', 'UH', 'CD', 'LS', 'SYM',
                     'POS', '-LRB-', '-RRB-', ',', '.', ':', '``', "''"}

        complex_candidates = []

        # --- Attempt trained CWI model ---
        gpu_model_path = os.path.join(os.path.dirname(__file__), 'gpu_attention.model')
        if os.path.exists(gpu_model_path):
            try:
                import sys as _sys
                _sys.path.insert(0, os.path.dirname(__file__))
                import complex_word as _cw
                probs = _cw.get_complex_words(words)
                COMPLEXITY_THRESHOLD = 0.5
                tags_all = nltk.pos_tag(words)
                for i, word in enumerate(words):
                    if len(word) < 3 or tags_all[i][1] in SKIP_POS:
                        continue
                    if i < len(probs) and probs[i] >= COMPLEXITY_THRESHOLD:
                        complex_candidates.append((word, float(probs[i]), i))
                complex_candidates.sort(key=lambda x: x[1], reverse=True)
                if complex_candidates:
                    return complex_candidates
            except Exception as _e:
                logger.debug(f"CWI model failed ({_e}), using heuristic fallback")

        # --- Improved heuristic fallback ---
        tags_all = nltk.pos_tag(words)
        for i, word in enumerate(words):
            if len(word) < 3 or tags_all[i][1] in SKIP_POS:
                continue

            # Multi-signal complexity score:
            freq = self.word_count.get(word.lower(), 0)
            # 1) Zipf frequency: lower = more complex
            freq_score = max(0, (6.0 - freq) / 6.0) if freq > 0 else 1.0
            # 2) Word length: longer words tend to be harder
            len_score = min(1.0, max(0, (len(word) - 4)) / 10.0)
            # 3) Syllable count estimate (vowel clusters)
            vowels = sum(1 for j, c in enumerate(word)
                         if c in 'aeiou' and (j == 0 or word[j-1] not in 'aeiou'))
            syl_score = min(1.0, max(0, (vowels - 1)) / 4.0)

            # Weighted combination
            complexity = 0.50 * freq_score + 0.25 * len_score + 0.25 * syl_score

            # Threshold: consider complex if score > 0.45
            if complexity > self.CWI_COMPLEXITY_THRESHOLD:
                complex_candidates.append((word, round(complexity, 3), i))

        complex_candidates.sort(key=lambda x: x[1], reverse=True)
        return complex_candidates

    def identify_complex_words(self, sentence: str) -> Dict[str, Any]:
        """
        Public API: Identify complex words in a sentence.

        Returns:
            dict with 'sentence', 'complex_words' (list of
            {'word', 'complexity', 'index'}).
        """
        try:
            tokens, words, position = self._convert_sentence_to_token(sentence)
        except Exception as e:
            return {
                'sentence': sentence,
                'complex_words': [],
                'error': str(e)
            }

        if len(words) != len(position):
            return {
                'sentence': sentence,
                'complex_words': [],
                'error': 'Token/position mismatch'
            }

        candidates = self._identify_complex_words(words)

        return {
            'sentence': sentence,
            'complex_words': [
                {'word': w, 'complexity': score, 'index': idx}
                for w, score, idx in candidates
            ]
        }

    def simplify_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Auto-detect complex words and simplify them one at a time.
        Returns complex words detected AND the simplifications made.

        Returns:
            dict with keys: original, simplified, complex_words, changes
        """
        changes = []
        current_sentence = sentence

        try:
            tokens, words, position = self._convert_sentence_to_token(current_sentence)
        except Exception as e:
            return {
                'original': sentence,
                'simplified': sentence,
                'complex_words': [],
                'changes': [],
                'error': str(e)
            }

        if len(words) != len(position):
            return {
                'original': sentence,
                'simplified': sentence,
                'complex_words': [],
                'changes': [],
                'error': 'Token/position mismatch'
            }

        # Step 1: Identify complex words
        all_complex = self._identify_complex_words(words)
        complex_word_list = [
            {'word': w, 'complexity': score, 'index': idx}
            for w, score, idx in all_complex
        ]

        # Step 2: Simplify the most complex words
        for word, _, _ in all_complex[:self.MAX_AUTO_SIMPLIFICATIONS]:
            result = self.simplify_word(current_sentence, word)
            if result['changed']:
                changes.append({
                    'original_word': word,
                    'replacement': result['replacement'],
                    'candidates': result['candidates'][:5],
                    'sense_preserved': result.get('sense_preserved', True),
                    'semantic_similarity': result.get('semantic_similarity', 1.0)
                })
                current_sentence = result['simplified']

        return {
            'original': sentence,
            'simplified': current_sentence,
            'complex_words': complex_word_list,
            'changes': changes
        }
