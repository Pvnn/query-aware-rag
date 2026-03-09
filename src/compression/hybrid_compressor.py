import torch
import time
import spacy
from typing import List, Union, Dict

from src.compression.quitox_filter import QuitoxCoarseFilter
from src.compression.ep_exit import EPExitCompressor
from src.eval.interfaces import SearchResult


class HybridCompressor:
    """
    The Master Module: Adaptive Two-Stage Compression Pipeline.

    Flow:
    Input Doc -> [Stage 3: QUITO-X Coarse Filter] -> Coarse Doc -> [Stage 4: EP-EXIT Fine Filter] -> Final Doc
    """

    def __init__(
        self,
        exit_token: str,
        quitox_model: str = "google/flan-t5-small",
        exit_model: str = "doubleyyh/exit-gemma-2b",
        device: str = None,
    ):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("\n🔗 Initializing HYBRID COMPRESSION PIPELINE...")

        # Load spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spacy model...")
            from spacy.cli import download

            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Stage 3
        print("--- [Stage 3] Loading QUITO-X ---")
        self.quitox = QuitoxCoarseFilter(model_name=quitox_model, device=self.device)

        # Stage 4
        print("--- [Stage 4] Loading EP-EXIT ---")
        self.exit = EPExitCompressor(
            token=exit_token,
            model_name=exit_model,
            threshold=0.5,
        )

        print("✅ Hybrid Pipeline Ready.\n")

    def _split_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    def compress(
        self,
        query: str,
        context: Union[str, List[str], List[SearchResult]],
        coarse_ratio: float = 0.7,
        fine_threshold: float = 0.5,
        use_coarse: bool = True,
        use_fine: bool = True,
    ) -> Dict:

        start_time = time.time()

        # -----------------------------
        # INPUT NORMALIZATION
        # -----------------------------

        if isinstance(context, str):
            sentences = self._split_sentences(context)

        elif isinstance(context, list) and len(context) > 0 and isinstance(context[0], SearchResult):
            full_text = " ".join([doc.text for doc in context])
            sentences = self._split_sentences(full_text)

        else:
            sentences = context

        original_count = len(sentences)
        current_sentences = sentences

        quitox_time = 0.0
        exit_time = 0.0

        quitox_tokens = 0
        exit_tokens = 0

        quitox_details = []

        # -----------------------------
        # STAGE 3 — QUITO-X
        # -----------------------------
        if use_coarse:
            t1 = time.time()

            quitox_result = self.quitox.compress(
                query=query,
                sentences=current_sentences,
                compression_ratio=coarse_ratio,
            )

            current_sentences = quitox_result["filtered_sentences"]
            quitox_tokens = quitox_result["total_tokens_consumed"]
            quitox_details = quitox_result.get("quitox_details", [])

            quitox_time = time.time() - t1

        stage3_count = len(current_sentences)

        # -----------------------------
        # STAGE 4 — EP-EXIT
        # -----------------------------
        if use_fine:

            coarse_context_str = " ".join(current_sentences)

            self.exit.threshold = fine_threshold

            t2 = time.time()

            exit_result = self.exit.compress_with_stats(query, coarse_context_str)

            exit_time = time.time() - t2

            final_text = exit_result["compressed_text"]
            final_sentence_count = exit_result["sentences_kept"]
            exit_tokens = exit_result.get("total_tokens_consumed", 0)

            ep_exit_details = {
                "evidence_units_total": exit_result.get("evidence_units_total", 0),
                "evidence_units_kept_count": exit_result.get("evidence_units_kept_count", 0),
                "evidence_units_removed_count": exit_result.get("evidence_units_removed_count", 0),
                "kept_units": exit_result.get("kept_units", []),
                "removed_units": exit_result.get("removed_units", []),
            }

        else:

            final_text = " ".join(current_sentences)
            final_sentence_count = len(current_sentences)

            ep_exit_details = {
                "evidence_units_total": 0,
                "evidence_units_kept_count": 0,
                "evidence_units_removed_count": 0,
                "kept_units": [],
                "removed_units": [],
            }

        total_time = time.time() - start_time
        total_compression_tokens = quitox_tokens + exit_tokens

        # -----------------------------
        # BUILD compressed_docs
        # -----------------------------

        compressed_docs = [
            SearchResult(
                evi_id=0,
                docid=0,
                title="compressed",
                text=final_text,
            )
        ]

        # -----------------------------
        # FINAL REPORT
        # -----------------------------

        stats = {
            "final_text": final_text,
            "compressed_docs": compressed_docs,
            "metrics": {
                "original_sentence_count": original_count,
                "coarse_sentence_count": stage3_count,
                "final_sentence_count": final_sentence_count,
                "time_quitox": round(quitox_time, 2),
                "time_exit": round(exit_time, 2),
                "time_total": round(total_time, 2),
                "tokens_quitox": quitox_tokens,
                "tokens_exit": exit_tokens,
                "tokens_total_compression": total_compression_tokens,
            },
            "quitox_details": quitox_details,
            "ep_exit_details": ep_exit_details,
        }

        return stats