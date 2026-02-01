from dataclasses import dataclass
from typing import List, Tuple
import re

from sentence_transformers import SentenceTransformer, util


@dataclass
class EvidenceUnit:
    """
    Represents a coherent multi-sentence evidence span.
    """
    sentences: List[str]
    start_idx: int
    end_idx: int

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


class EPExitCompressor:
    """
    Evidence-Preserving EXIT (EP-EXIT)

    EXIT logic is preserved.
    Only the unit of filtering is changed:
    sentence-level → evidence-span-level.
    """

    def __init__(
        self,
        window_size: int = 2,
        threshold: float = 0.5,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)

    # ---------------------------------------------------------
    # STEP 1: Sentence Segmentation
    # ---------------------------------------------------------
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split raw document text into sentences.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]

    # ---------------------------------------------------------
    # STEP 2: Evidence Unit Construction
    # ---------------------------------------------------------
    def build_evidence_units(self, sentences: List[str]) -> List[EvidenceUnit]:
        """
        Group adjacent sentences into evidence spans.
        """
        units = []

        for i in range(0, len(sentences), self.window_size):
            window = sentences[i:i + self.window_size]
            unit = EvidenceUnit(
                sentences=window,
                start_idx=i,
                end_idx=i + len(window) - 1
            )
            units.append(unit)

        return units

    # ---------------------------------------------------------
    # STEP 3: Evidence-Level Scoring (EXIT logic unchanged)
    # ---------------------------------------------------------
    def score_units(
        self,
        query: str,
        units: List[EvidenceUnit]
    ) -> List[Tuple[EvidenceUnit, float]]:
        """
        Apply EXIT-style relevance scoring at the evidence-span level.
        """
        query_emb = self.model.encode(query, convert_to_tensor=True)
        unit_texts = [u.text for u in units]
        unit_embs = self.model.encode(unit_texts, convert_to_tensor=True)

        scores = util.cos_sim(query_emb, unit_embs)[0]

        return [(unit, float(score)) for unit, score in zip(units, scores)]

    # ---------------------------------------------------------
    # STEP 4: Threshold-Based Evidence Selection (EXIT-style)
    # ---------------------------------------------------------
    def filter_units(
        self,
        scored_units: List[Tuple[EvidenceUnit, float]]
    ) -> List[EvidenceUnit]:
        """
        Keep evidence units with score > threshold.
        """
        return [
            unit for unit, score in scored_units
            if score > self.threshold
        ]

    # ---------------------------------------------------------
    # STEP 5: Context Reconstruction
    # ---------------------------------------------------------
    def reconstruct_context(self, units: List[EvidenceUnit]) -> str:
        """
        Reconstruct compressed context preserving original order.
        """
        ordered = sorted(units, key=lambda u: u.start_idx)
        return " ".join(u.text for u in ordered)

    # ---------------------------------------------------------
    # FULL EP-EXIT PIPELINE
    # ---------------------------------------------------------
    def compress(
        self,
        query: str,
        document: str
    ) -> str:
        sentences = self.split_into_sentences(document)
        units = self.build_evidence_units(sentences)
        scored_units = self.score_units(query, units)
        selected_units = self.filter_units(scored_units)
        return self.reconstruct_context(selected_units)
