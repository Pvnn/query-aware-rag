"""
EP-EXIT: Evidence-Preserving EXIT
--------------------------------
This module extends EXIT by grouping adjacent sentences
into coherent evidence units before scoring.
"""

from typing import List


class EvidenceUnit:
    """
    A small group of adjacent sentences that form
    a coherent evidence span.
    """
    def __init__(self, sentences: List[str], start_idx: int):
        self.sentences = sentences
        self.start_idx = start_idx
        self.text = " ".join(sentences)

    def __repr__(self):
        return f"EvidenceUnit(start={self.start_idx}, size={len(self.sentences)})"


class EPExitCompressor:
    """
    Evidence-Preserving EXIT compressor.
    """
    def __init__(self, window_size: int = 2):
        """
        Args:
            window_size: Number of adjacent sentences per evidence unit
        """
        self.window_size = window_size

    def build_evidence_units(self, sentences: List[str]) -> List[EvidenceUnit]:
        """
        Step 2: Evidence Unit Construction

        Groups adjacent sentences into overlapping windows.
        """
        evidence_units = []

        for i in range(len(sentences)):
            window = sentences[i:i + self.window_size]
            if len(window) < self.window_size:
                break

            unit = EvidenceUnit(
                sentences=window,
                start_idx=i
            )
            evidence_units.append(unit)

        return evidence_units
