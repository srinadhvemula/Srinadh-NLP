#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
LSBert1.py — Multi-mask BERT Ensemble for Lexical Simplification
=================================================================
Innovation: Instead of masking only the complex word (single-mask, LSBert2.py),
this module additionally masks surrounding context words in segment A and
averages the resulting logit distributions.  This reduces the frequency bias
that causes BERT to prefer common-but-wrong substitutes (e.g. "gave" instead
of "performed" for "administered").

Key new function:
    multi_mask_BERT_prediction(tokens_a, mask_position, tokenizer, maskedLM,
                               seq_length, prob_mask, context_window, device)

Comparison runner (CLI):
    python LSBert1.py --sentence "The physician administered the preliminary examination" \
                      --target_word administered \
                      --bert_model bert-base-uncased \
                      --word_frequency ../datasets/SUBTLEX_frequency.xlsx \
                      --ppdb ../datasets/ppdb.db

Full evaluation (same flags as LSBert2.py, adds --multi_mask):
    python LSBert1.py --eval_dir data/test.txt --bert_model bert-base-uncased \
                      --output_SR_file results_multi.txt \
                      --word_embeddings datasets/Fasttext/crawl-300d-2M-subword.vec \
                      --word_frequency datasets/SUBTLEX_frequency.xlsx \
                      --ppdb datasets/ppdb.db --do_eval --multi_mask
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import nltk
import openpyxl
from pathlib import Path
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity as cosine
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Add parent/src to path so local modules resolve regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from PPDB import Ppdb


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class InputFeatures:
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


# ---------------------------------------------------------------------------
# Tokenisation helpers  (identical to LSBert2.py)
# ---------------------------------------------------------------------------

def convert_sentence_to_token(sentence, seq_length, tokenizer):
    tokenized_text = tokenizer.tokenize(sentence.lower())
    assert len(tokenized_text) < seq_length - 2

    nltk_sent = nltk.word_tokenize(sentence.lower())
    position2 = []
    token_index = 0
    start_pos = len(tokenized_text) + 2
    pre_word = ""

    for i, word in enumerate(nltk_sent):
        if word == "n't" and pre_word and pre_word[-1] == "n":
            word = "'t"

        len_token = 2 if tokenized_text[token_index] == '"' else len(tokenized_text[token_index])

        if tokenized_text[token_index] == word or len_token >= len(word):
            position2.append(start_pos + token_index)
            pre_word = tokenized_text[token_index]
            token_index += 1
        else:
            new_pos = [start_pos + token_index]
            new_word = tokenized_text[token_index]
            while new_word != word:
                token_index += 1
                new_word += tokenized_text[token_index].replace('##', '')
                new_pos.append(start_pos + token_index)
                if len(new_word) == len(word):
                    break
            token_index += 1
            pre_word = new_word
            position2.append(new_pos)

    return tokenized_text, nltk_sent, position2


def convert_token_to_feature(tokens_a, mask_position, seq_length, tokenizer, prob_mask):
    """Build sentence-pair masked-LM feature for a single-token word."""
    tokens = ["[CLS]"]
    input_type_ids = [0]

    len_tokens = len(tokens_a)
    first_sentence_mask_random = random.sample(range(0, len_tokens),
                                               int(prob_mask * len_tokens))

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

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0, tokens=tokens, input_ids=input_ids,
                         input_mask=input_mask, input_type_ids=input_type_ids)


def convert_whole_word_to_feature(tokens_a, mask_position, seq_length, tokenizer, prob_mask):
    """Build sentence-pair masked-LM feature for a multi-token word."""
    tokens = ["[CLS]"]
    input_type_ids = [0]

    len_tokens = len(tokens_a)
    first_sentence_mask_random = random.sample(range(0, len_tokens),
                                               int(prob_mask * len_tokens))

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

    mask_position_length = len(mask_position)
    count = 0
    while count in range(mask_position_length):
        index = mask_position_length - 1 - count
        pos = mask_position[index]
        if index == 0:
            tokens[pos] = '[MASK]'
        else:
            del tokens[pos]
            del input_type_ids[pos]
        count += 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0, tokens=tokens, input_ids=input_ids,
                         input_mask=input_mask, input_type_ids=input_type_ids)


# ---------------------------------------------------------------------------
# NEW: Multi-mask BERT ensemble prediction
# ---------------------------------------------------------------------------

def multi_mask_BERT_prediction(tokens_a, mask_position, tokenizer, maskedLM,
                                seq_length, prob_mask=0.5, context_window=2,
                                device=None):
    """
    Multi-mask BERT ensemble prediction.

    Algorithm
    ---------
    Pass 0  — standard single-mask: target word masked in segment B, segment A
              carries the complete sentence (with random prob_mask noise).

    Passes 1..K — for each context position within ±context_window of the
              target in segment A, create a new input where that context token
              is *additionally* masked in segment A, then run a second forward
              pass.  This forces BERT to rely on broader syntactic cues rather
              than immediately neighbouring words, mitigating frequency bias.

    The raw logit distributions at the target position from all passes are
    averaged (before softmax), then top-80 predictions are taken.

    Why it helps
    ------------
    Single-mask BERT for "The physician [MASK] the preliminary examination"
    assigns the highest probability to "gave" because "give" is extremely
    frequent after "physician" in the training corpus.  When we additionally
    mask "examination" in segment A, BERT can no longer exploit the noun as
    an easy hint, and contextually appropriate verbs like "performed" and
    "conducted" are boosted in the ensemble average.

    Parameters
    ----------
    tokens_a        : list[str]   BERT token strings for the sentence.
    mask_position   : int | list[int]  Position of the target in full sequence.
    tokenizer       : BertTokenizer
    maskedLM        : BertForMaskedLM
    seq_length      : int         Maximum sequence length (e.g. 128).
    prob_mask       : float       Proportion of random masks in segment A.
    context_window  : int         Positions on each side of target to consider.
    device          : torch.device or None (auto-detects).

    Returns
    -------
    pre_tokens       : list[str]   Top-80 tokens from averaged distribution.
    pre_scores       : np.ndarray  Corresponding averaged logit values.
    single_pre_tokens: list[str]   Top-5 tokens from baseline single-mask
                                   (returned for side-by-side comparison).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

    # Build the baseline (single-mask) feature
    if isinstance(mask_position, list):
        feature = convert_whole_word_to_feature(
            tokens_a, mask_position, seq_length, tokenizer, prob_mask)
        target_pos_B = mask_position[0]
        target_pos_A = mask_position[0] - len(tokens_a) - 1
    else:
        feature = convert_token_to_feature(
            tokens_a, mask_position, seq_length, tokenizer, prob_mask)
        target_pos_B = mask_position
        # Segment A occupies indices 1..len(tokens_a) in the full sequence.
        # start_pos = len(tokens_a)+2, so BERT-token i → position start_pos+i.
        # Rearranging: position p in segment B → i = p - len(tokens_a) - 2,
        # and the same token in segment A is at position 1 + i = p - len(tokens_a) - 1.
        target_pos_A = mask_position - len(tokens_a) - 1

    base_ids   = feature.input_ids
    type_ids   = feature.input_type_ids
    attn_mask  = feature.input_mask

    tokens_tensor   = torch.tensor([base_ids]).to(device)
    type_ids_tensor = torch.tensor([type_ids]).to(device)
    attn_tensor     = torch.tensor([attn_mask]).to(device)

    # --- Pass 0: baseline single-mask ---
    with torch.no_grad():
        _, prediction_scores = maskedLM(tokens_tensor, type_ids_tensor, attn_tensor)

    baseline_logits = prediction_scores[0, target_pos_B].cpu().numpy()
    all_logits = [baseline_logits]

    # Save single-mask top-5 before ensemble averaging (for comparison)
    single_top = torch.tensor(baseline_logits, dtype=torch.float32).topk(5)
    single_pre_tokens = tokenizer.convert_ids_to_tokens(single_top[1].cpu().numpy())

    # --- Passes 1..K: additionally mask context words in segment A ---
    first_sent_start = 1
    first_sent_end   = len(tokens_a)           # inclusive

    for offset in range(-context_window, context_window + 1):
        if offset == 0:
            continue
        ctx_pos = target_pos_A + offset
        if ctx_pos < first_sent_start or ctx_pos > first_sent_end:
            continue
        # Skip positions already [MASK] (from prob_mask) or padding (id=0)
        if base_ids[ctx_pos] == mask_id or base_ids[ctx_pos] == 0:
            continue

        ctx_ids = list(base_ids)
        ctx_ids[ctx_pos] = mask_id

        ctx_tensor = torch.tensor([ctx_ids]).to(device)

        with torch.no_grad():
            _, ctx_scores = maskedLM(ctx_tensor, type_ids_tensor, attn_tensor)

        all_logits.append(ctx_scores[0, target_pos_B].cpu().numpy())

    # Average logits across all passes (before softmax for numerical stability)
    avg_logits    = np.mean(all_logits, axis=0)
    avg_tensor    = torch.tensor(avg_logits, dtype=torch.float32)
    predicted_top = avg_tensor.topk(80)

    pre_tokens = tokenizer.convert_ids_to_tokens(predicted_top[1].cpu().numpy())
    pre_scores = predicted_top[0].cpu().numpy()

    return pre_tokens, pre_scores, single_pre_tokens


# ---------------------------------------------------------------------------
# Candidate generation  (identical to LSBert2.py)
# ---------------------------------------------------------------------------

def BERT_candidate_generation(source_word, pre_tokens, pre_scores, ps, num_selection=10):
    cur_tokens = []
    source_stem = ps.stem(source_word)

    assert num_selection <= len(pre_tokens)

    for i in range(len(pre_tokens)):
        token = pre_tokens[i]
        if token[:2] == "##":
            continue
        if token == source_word:
            continue
        token_stem = ps.stem(token)
        if token_stem == source_stem:
            continue
        if len(token_stem) >= 3 and token_stem[:3] == source_stem[:3]:
            continue
        cur_tokens.append(token)
        if len(cur_tokens) == num_selection:
            break

    if len(cur_tokens) == 0:
        cur_tokens = pre_tokens[:num_selection + 1]

    assert len(cur_tokens) > 0
    return cur_tokens


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def compare_single_vs_multi(sentence, target_word, tokenizer, maskedLM,
                             seq_length, prob_mask, context_window, ps, device):
    """
    Run both single-mask (LSBert2 baseline) and multi-mask (LSBert1 ensemble)
    on a single sentence+target pair and print a side-by-side top-10 table.

    Returns
    -------
    single_candidates : list[str]   top-10 from single-mask
    multi_candidates  : list[str]   top-10 from multi-mask ensemble
    """
    tokens, words, position = convert_sentence_to_token(sentence, seq_length, tokenizer)
    target_lower = target_word.lower()

    if target_lower not in words:
        print(f'[ERROR] "{target_word}" not found in tokenised sentence.')
        print(f'  Tokenised words: {words}')
        return [], []

    mask_index    = words.index(target_lower)
    mask_position = position[mask_index]

    # ---- Single-mask (LSBert2 path) ----------------------------------------
    if isinstance(mask_position, list):
        feature_s = convert_whole_word_to_feature(
            tokens, mask_position, seq_length, tokenizer, prob_mask)
        target_pos_B = mask_position[0]
    else:
        feature_s = convert_token_to_feature(
            tokens, mask_position, seq_length, tokenizer, prob_mask)
        target_pos_B = mask_position

    t_ids  = torch.tensor([feature_s.input_ids]).to(device)
    tt_ids = torch.tensor([feature_s.input_type_ids]).to(device)
    t_mask = torch.tensor([feature_s.input_mask]).to(device)

    with torch.no_grad():
        _, pred_s = maskedLM(t_ids, tt_ids, t_mask)

    top_s = pred_s[0, target_pos_B].topk(80)
    single_raw   = tokenizer.convert_ids_to_tokens(top_s[1].cpu().numpy())
    single_scores = top_s[0].cpu().numpy()
    single_candidates = BERT_candidate_generation(
        target_lower, single_raw, single_scores, ps, num_selection=10)

    # ---- Multi-mask (LSBert1 ensemble) -------------------------------------
    multi_raw, multi_scores_arr, _ = multi_mask_BERT_prediction(
        tokens, mask_position, tokenizer, maskedLM,
        seq_length, prob_mask, context_window, device)
    multi_candidates = BERT_candidate_generation(
        target_lower, multi_raw, multi_scores_arr, ps, num_selection=10)

    # ---- Print table --------------------------------------------------------
    print(f'\nSentence  : {sentence}')
    print(f'Target    : {target_word}')
    print(f'Context window for multi-mask: ±{context_window}\n')
    print(f'{"Rank":<6} {"Single-mask (baseline)":<28} {"Multi-mask (ensemble)":<28}')
    print('-' * 64)
    for i, (s, m) in enumerate(
            zip(single_candidates, multi_candidates), start=1):
        marker = ' <--' if s != m else ''
        print(f'{i:<6} {s:<28} {m:<28}{marker}')
    print()

    return single_candidates, multi_candidates


# ---------------------------------------------------------------------------
# Utility loaders  (identical to LSBert2.py)
# ---------------------------------------------------------------------------

def getWordmap(wordVecPath):
    words, We = [], []
    with open(wordVecPath, 'r') as f:
        for n, line in enumerate(f):
            if n == 0:
                print(line)
                continue
            word, vect = line.rstrip().split(' ', 1)
            We.append(np.fromstring(vect, sep=' '))
            words.append(word)
    return words, We


def getWordCount(word_count_path):
    word2count = {}
    xlsx_file = Path('', word_count_path)
    wb_obj = openpyxl.load_workbook(xlsx_file)
    sheet = wb_obj.active
    last_column = sheet.max_column - 1
    for i, row in enumerate(sheet.iter_rows(values_only=True)):
        if i == 0:
            continue
        word2count[row[0]] = round(float(row[last_column]), 3)
    return word2count


def read_eval_index_dataset(data_path):
    sentences, mask_words, mask_labels = [], [], []
    with open(data_path, 'r', encoding='ISO-8859-1') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            sentence, words = line.split('\t', 1)
            mask_word, labels = words.strip().split('\t', 1)
            label = labels.split('\t')
            sentences.append(sentence)
            mask_words.append(mask_word)
            one_labels = []
            for la in label[1:]:
                if la not in one_labels:
                    _, la_word = la.split(':')
                    one_labels.append(la_word)
            mask_labels.append(one_labels)
    return sentences, mask_words, mask_labels


def read_eval_dataset(data_path):
    sentences, mask_words, mask_labels = [], [], []
    with open(data_path, 'r', encoding='ISO-8859-1') as reader:
        for i, line in enumerate(reader):
            if i == 0:
                continue
            line = line.strip()
            if not line:
                continue
            sentence, words = line.split('\t', 1)
            mask_word, labels = words.strip().split('\t', 1)
            label = labels.split('\t')
            sentences.append(sentence)
            mask_words.append(mask_word)
            one_labels = []
            for la in label:
                if la not in one_labels:
                    one_labels.append(la)
            mask_labels.append(one_labels)
    return sentences, mask_words, mask_labels


# ---------------------------------------------------------------------------
# Scoring helpers  (identical to LSBert2.py)
# ---------------------------------------------------------------------------

def cross_entropy_word(X, i, pos):
    X = softmax(X, axis=1)
    return -np.log10(X[i, pos])


def get_score(sentence, tokenizer, maskedLM, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenize_input = tokenizer.tokenize(sentence)
    len_sen = len(tokenize_input)
    tokenize_input = ['[CLS]'] + tokenize_input + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokenize_input)
    sentence_loss = 0
    for i, word in enumerate(tokenize_input):
        if word in ('[CLS]', '[SEP]'):
            continue
        orig = tokenize_input[i]
        tokenize_input[i] = '[MASK]'
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
        with torch.no_grad():
            att, pre_word = maskedLM(mask_input)
        sentence_loss += cross_entropy_word(pre_word[0].cpu().numpy(), i, input_ids[i])
        tokenize_input[i] = orig
    return np.exp(sentence_loss / len_sen)


def LM_score(source_word, source_context, substitution_selection,
             tokenizer, maskedLM, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_sentence = ' '.join(source_context).strip()
    LM = []
    source_loss = get_score(source_sentence, tokenizer, maskedLM, device)
    for sub in substitution_selection:
        sub_sentence = source_sentence.replace(source_word, sub, 1)
        LM.append(get_score(sub_sentence, tokenizer, maskedLM, device))
    return LM, source_loss


def preprocess_SR(source_word, substitution_selection, fasttext_dico,
                  fasttext_emb, word_count):
    ss, sis_scores, count_scores = [], [], []
    isFast = source_word in fasttext_dico
    if isFast:
        source_emb = fasttext_emb[fasttext_dico.index(source_word)].reshape(1, -1)

    for sub in substitution_selection:
        if sub not in word_count:
            continue
        sub_count = word_count[sub]
        if sub_count <= 3:
            continue
        if isFast:
            if sub not in fasttext_dico:
                continue
            sis = cosine(source_emb, fasttext_emb[fasttext_dico.index(sub)].reshape(1, -1))
            sis_scores.append(sis)
        ss.append(sub)
        count_scores.append(sub_count)
    return ss, sis_scores, count_scores


def substitution_ranking(source_word, source_context, substitution_selection,
                         fasttext_dico, fasttext_emb, word_count, ssPPDB,
                         tokenizer, maskedLM, labels, device=None):
    ss, sis_scores, count_scores = preprocess_SR(
        source_word, substitution_selection, fasttext_dico, fasttext_emb, word_count)
    if len(ss) == 0:
        return source_word

    if len(sis_scores) > 0:
        seq = sorted(sis_scores, reverse=True)
        sis_rank = [seq.index(v) + 1 for v in sis_scores]

    rank_count = sorted(count_scores, reverse=True)
    count_rank = [rank_count.index(v) + 1 for v in count_scores]

    lm_score, source_lm = LM_score(source_word, source_context, ss,
                                    tokenizer, maskedLM, device)
    rank_lm = sorted(lm_score)
    lm_rank = [rank_lm.index(v) + 1 for v in lm_score]

    bert_rank, ppdb_rank = [], []
    for i in range(len(ss)):
        bert_rank.append(i + 1)
        ppdb_rank.append(1 if ss[i] in ssPPDB else len(ss) / 3)

    if len(sis_scores) > 0:
        all_ranks = [b + s + c + l + p
                     for b, s, c, l, p
                     in zip(bert_rank, sis_rank, count_rank, lm_rank, ppdb_rank)]
    else:
        all_ranks = [b + c + l + p
                     for b, c, l, p
                     in zip(bert_rank, count_rank, lm_rank, ppdb_rank)]

    pre_index  = all_ranks.index(min(all_ranks))
    pre_count  = count_scores[pre_index]
    source_count = word_count.get(source_word, 0)
    pre_lm     = lm_score[pre_index]

    return ss[pre_index] if (source_lm > pre_lm or pre_count > source_count) else source_word


def extract_context(words, mask_index, window):
    length = len(words)
    half   = int(window / 2)
    assert 0 <= mask_index < length
    if length <= window:
        return words
    if mask_index >= half and mask_index < length - half:
        return words[mask_index - half: mask_index + half + 1]
    if mask_index < half:
        return words[0:window]
    return words[length - window: length]


def preprocess_tag(tag):
    if tag[0] in ('V', 'N'):
        return tag
    if tag[0] == 'R':
        return 'r'
    if tag[0] in ('J', 'I'):
        return 'a'
    return 's'


def evaulation_SS_scores(ss, labels):
    assert len(ss) == len(labels)
    potential = precision = precision_all = recall = recall_all = 0
    instances = len(ss)
    for i in range(instances):
        common = list(set(ss[i]).intersection(labels[i]))
        if len(common) >= 1:
            potential += 1
        precision     += len(common)
        recall        += len(common)
        precision_all += len(ss[i])
        recall_all    += len(labels[i])
    potential  /= instances
    precision  /= precision_all
    recall     /= recall_all
    F_score = 2 * precision * recall / (precision + recall)
    return potential, precision, recall, F_score


def evaulation_pipeline_scores(substitution_words, source_words, gold_words):
    instances = len(substitution_words)
    precision = accuracy = changed = 0
    for sub, source, gold in zip(substitution_words, source_words, gold_words):
        if sub == source or sub in gold:
            precision += 1
        if sub != source and sub in gold:
            accuracy += 1
        if sub != source:
            changed += 1
    return precision / instances, accuracy / instances, changed / instances


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='LSBert1 — Multi-mask BERT ensemble for lexical simplification.')

    # Single-sentence comparison mode
    parser.add_argument('--sentence', default=None, type=str,
                        help='Sentence to simplify (single-sentence mode).')
    parser.add_argument('--target_word', default=None, type=str,
                        help='Target word to simplify (single-sentence mode).')

    # Full evaluation mode (same as LSBert2.py)
    parser.add_argument('--eval_dir', default=None, type=str,
                        help='Evaluation data file path.')
    parser.add_argument('--output_SR_file', default=None, type=str,
                        help='Output file for substitution results.')
    parser.add_argument('--word_embeddings', default=None, type=str,
                        help='Path to FastText word vectors (.vec).')
    parser.add_argument('--ppdb', default='./ppdb-2.0-tldr', type=str,
                        help='Path to PPDB file or SQLite database.')
    parser.add_argument('--do_eval', action='store_true',
                        help='Run full evaluation loop.')

    # Shared parameters
    parser.add_argument('--bert_model', default=None, type=str, required=True,
                        help='Pretrained BERT model name or path.')
    parser.add_argument('--word_frequency', default=None, type=str, required=True,
                        help='Path to SUBTLEX word frequency .xlsx file.')
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--num_selections', default=10, type=int)
    parser.add_argument('--prob_mask', default=0.5, type=float,
                        help='Fraction of segment-A tokens randomly masked.')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', default=42, type=int)

    # Multi-mask specific
    parser.add_argument('--multi_mask', action='store_true', default=True,
                        help='Use multi-mask BERT ensemble (default: True).')
    parser.add_argument('--no_multi_mask', dest='multi_mask', action='store_false',
                        help='Fall back to single-mask (LSBert2 behaviour).')
    parser.add_argument('--context_window', default=2, type=int,
                        help='Context positions on each side to mask (default 2).')

    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'Loading BERT model: {args.bert_model}')
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    model = BertForMaskedLM.from_pretrained(args.bert_model, output_attentions=True)
    model.to(device)
    model.eval()

    ps = PorterStemmer()

    print('Loading word frequency ...')
    word_count = getWordCount(args.word_frequency)

    # ------------------------------------------------------------------ #
    # Single-sentence comparison mode                                      #
    # ------------------------------------------------------------------ #
    if args.sentence and args.target_word:
        compare_single_vs_multi(
            sentence=args.sentence,
            target_word=args.target_word,
            tokenizer=tokenizer,
            maskedLM=model,
            seq_length=args.max_seq_length,
            prob_mask=args.prob_mask,
            context_window=args.context_window,
            ps=ps,
            device=device,
        )
        return

    # ------------------------------------------------------------------ #
    # Full evaluation mode                                                 #
    # ------------------------------------------------------------------ #
    if not args.do_eval:
        parser.error('Provide --sentence/--target_word or --do_eval.')

    if args.word_embeddings:
        print('Loading word embeddings ...')
        fasttext_dico, fasttext_emb = getWordmap(args.word_embeddings)
    else:
        fasttext_dico, fasttext_emb = [], []

    print('Loading PPDB ...')
    ppdb_model = Ppdb(args.ppdb)

    fileName = os.path.splitext(os.path.basename(args.eval_dir))[0]
    if fileName == 'lex.mturk':
        eval_examples, mask_words, mask_labels = read_eval_dataset(args.eval_dir)
    else:
        eval_examples, mask_words, mask_labels = read_eval_index_dataset(args.eval_dir)

    eval_size = len(eval_examples)
    print(f'Running evaluation on {eval_size} examples '
          f'(multi_mask={args.multi_mask}, context_window=±{args.context_window})')

    output_sr_file = open(args.output_SR_file, 'a+')
    CGBERT = []
    substitution_words = []
    window_context = 11

    for i in range(eval_size):
        print(f'Sentence {i}: {eval_examples[i]}  |  target: {mask_words[i]}')

        tokens, words, position = convert_sentence_to_token(
            eval_examples[i], args.max_seq_length, tokenizer)

        assert len(words) == len(position)
        mask_index    = words.index(mask_words[i])
        mask_context  = extract_context(words, mask_index, window_context)
        mask_position = position[mask_index]

        if args.multi_mask:
            pre_tokens, pre_scores, single_top5 = multi_mask_BERT_prediction(
                tokens, mask_position, tokenizer, model,
                args.max_seq_length, args.prob_mask, args.context_window, device)
            print(f'  single-mask top-5: {list(single_top5)}')
            print(f'  multi-mask  top-5: {list(pre_tokens[:5])}')
        else:
            if isinstance(mask_position, list):
                feature = convert_whole_word_to_feature(
                    tokens, mask_position, args.max_seq_length, tokenizer, args.prob_mask)
                target_pos_B = mask_position[0]
            else:
                feature = convert_token_to_feature(
                    tokens, mask_position, args.max_seq_length, tokenizer, args.prob_mask)
                target_pos_B = mask_position

            t_ids  = torch.tensor([feature.input_ids]).to(device)
            tt_ids = torch.tensor([feature.input_type_ids]).to(device)
            t_mask = torch.tensor([feature.input_mask]).to(device)

            with torch.no_grad():
                _, pred = model(t_ids, tt_ids, t_mask)

            top = pred[0, target_pos_B].topk(80)
            pre_tokens = tokenizer.convert_ids_to_tokens(top[1].cpu().numpy())
            pre_scores = top[0].cpu().numpy()

        sentence_lower = eval_examples[i].lower()
        words_lower    = word_tokenize(sentence_lower)
        words_tag      = nltk.pos_tag(words_lower)

        complex_word_index = words_lower.index(mask_words[i])
        complex_word_tag   = preprocess_tag(words_tag[complex_word_index][1])
        cgPPDB = ppdb_model.predict(mask_words[i], complex_word_tag)

        cgBERT = BERT_candidate_generation(
            mask_words[i], pre_tokens, pre_scores, ps, args.num_selections)
        print(f'  candidates: {cgBERT}')
        CGBERT.append(cgBERT)

        pre_word = substitution_ranking(
            mask_words[i], mask_context, cgBERT,
            fasttext_dico, fasttext_emb, word_count,
            cgPPDB, tokenizer, model, mask_labels[i], device)
        substitution_words.append(pre_word)

    potential, precision, recall, F_score = evaulation_SS_scores(CGBERT, mask_labels)
    print('\n=== Candidate generation scores ===')
    print(f'Potential={potential:.4f}  Precision={precision:.4f}  '
          f'Recall={recall:.4f}  F={F_score:.4f}')

    p, acc, chg = evaulation_pipeline_scores(substitution_words, mask_words, mask_labels)
    print('\n=== Full pipeline scores ===')
    print(f'Precision={p:.4f}  Accuracy={acc:.4f}  Changed={chg:.4f}')

    output_sr_file.write(
        f'{args.num_selections}\t{potential}\t{precision}\t{recall}\t{F_score}\t'
        f'{p}\t{acc}\t{chg}\t'
        f'multi_mask={args.multi_mask},window={args.context_window}\n')
    output_sr_file.close()


if __name__ == '__main__':
    main()
