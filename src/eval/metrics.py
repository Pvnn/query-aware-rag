# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file 
# in the root directory of the DPR source tree.


import collections
import logging
import regex
import string
import unicodedata
from collections import Counter
from typing import List

logger = logging.getLogger(__name__)

# =========================================================
# 1. NORMALIZATION
# =========================================================
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _normalize(text):
    return unicodedata.normalize('NFD', text)

# =========================================================
# 2. ANSWER SURVIVAL (Facebook DPR)
# =========================================================
class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def has_answer(answers: List[str], text: str, tokenizer: SimpleTokenizer) -> bool:
    """Check if a document contains any of the answer strings."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

# =========================================================
# 3. EXTRACTIVE COMPRESSION METRICS
# =========================================================
def context_overlap_scores(compressed_text: str, gold_text: str) -> dict:
    """
    Evaluates how well the compressor isolated the gold supporting facts.
    Returns Precision, Recall, and F1 at the token level.
    """
    comp_tokens = normalize_answer(compressed_text).split()
    gold_tokens = normalize_answer(gold_text).split()
    
    if not comp_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(comp_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    precision = num_same / len(comp_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}