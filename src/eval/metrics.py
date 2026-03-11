import re
import string
from collections import Counter
from typing import List
import numpy as np
import spacy
from rouge_score import rouge_scorer

# 1. EXACT MATCH (EM) CORRECTNESS (LLMLingua / Soft EM)
def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD/LLMLingua evaluation script."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def answer_em_correctness(output: str, short_answers: List[str]) -> float:
    """Computes exact match correctness based on inclusion (Soft EM)."""
    normalized_prediction = normalize_answer(output)
    
    for candidate in short_answers:
        normalized_ground_truth = normalize_answer(candidate)
        if normalized_ground_truth in normalized_prediction:
            return 1.0
    return 0.0


# 2. SQuAD TOKEN F1 (From LLMLingua)
def token_f1_score(prediction: str, ground_truth: str) -> float:
    """Computes Token F1 score as used in SQuAD and LLMLingua."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def answer_f1_correctness(output: str, short_answers: List[str]) -> float:
    """Returns the maximum Token F1 score across all acceptable ground truths."""
    scores = [token_f1_score(output, gt) for gt in short_answers]
    return float(np.max(scores)) if scores else 0.0


# 3. ROUGE CORRECTNESS
def answer_rouge_correctness(output: str, gt_answers: List[str], rouge_type: str = "rougeL") -> float:
    """Computes ROUGE score (default ROUGE-L) between generated output and ground truth."""
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = []
    for gt in gt_answers:
        score = scorer.score(gt, output)
        scores.append(score[rouge_type].fmeasure)
    return float(np.max(scores)) if scores else 0.0


# 4. DISAMBIG-F1 CORRECTNESS (SPACY NER)
class DisambigF1Evaluator:
    """Estimates the Disambig-F1 using Spacy Named Entity Recognition (NER)."""
    def __init__(self, model_name: str = "en_core_web_sm"):
        print(f"Loading Spacy NER model ({model_name})...")
        self.nlp = spacy.load(model_name)

    def _ner(self, s: str) -> List[str]:
        doc = self.nlp(s)
        return [normalize_answer(e.text) for e in doc.ents]

    def evaluate(self, answer: str, gt_answers: List[str]) -> float:
        scores = []
        for gt in gt_answers:
            pred_ents = self._ner(answer)
            ref_ents = self._ner(gt)

            pred_counter = Counter(pred_ents)
            ref_counter = Counter(ref_ents)

            tp = sum((pred_counter & ref_counter).values())
            fp = sum((pred_counter - ref_counter).values())
            fn = sum((ref_counter - pred_counter).values())

            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0

            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append(2 * (precision * recall) / (precision + recall))
                
        return float(np.max(scores)) if scores else 0.0