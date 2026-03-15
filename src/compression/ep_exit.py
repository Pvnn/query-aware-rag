from dataclasses import dataclass
from typing import List
import networkx as nx
from sentence_transformers import SentenceTransformer, util

from src.compression.exit_baseline import ExitBaselineCompressor

@dataclass
class EvidenceUnit:
    sentences: List[str]
    indices: List[int] 
    start_idx: int
    end_idx: int

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

class EPExitCompressor:
    def __init__(
        self,
        token,
        model_name="doubleyyh/exit-gemma-2b",
        threshold=0.5,
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.45,
        locality_window=2
    ):
        print("Initializing EPExitCompressor...")
        self.similarity_threshold = similarity_threshold
        self.locality_window = locality_window
        self.threshold = threshold
        
        self.embedder = SentenceTransformer(embedding_model)
        self.exit = ExitBaselineCompressor(
            token=token, 
            model_name=model_name, 
            threshold=threshold
        )
        print("✓ EP-EXIT initialized\n")

    def decompose_sentences(self, text):
        return self.exit.decompose_sentences(text)

    def classify_sentence(self, query, sentence, document):
        return self.exit.classify_sentence(query, sentence, document)

    def build_similarity_graph(self, sentences):
        embeddings = self.embedder.encode(sentences, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeddings, embeddings)
        n = len(sentences)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(i - j) > self.locality_window:
                    continue
                if float(sim_matrix[i][j]) >= self.similarity_threshold:
                    G.add_edge(i, j)
        return G

    def extract_evidence_units(self, graph, sentences):
        units = []
        for component in nx.connected_components(graph):
            idxs = sorted(component)
            unit_sentences = [sentences[i] for i in idxs]
            units.append(
                EvidenceUnit(
                    sentences=unit_sentences,
                    indices=idxs, 
                    start_idx=idxs[0],
                    end_idx=idxs[-1],
                )
            )
        return units

    def compress(self, query: str, document: str) -> str:
        sentences = self.decompose_sentences(document)
        if not sentences:
            return ""

        graph = self.build_similarity_graph(sentences)
        units = self.extract_evidence_units(graph, sentences)

        # Batch classify all evidence units in one shot
        unit_texts = [unit.text for unit in units]
        queries = [query] * len(unit_texts)
        yes_probs, _ = self.exit._predict_batch(queries, unit_texts, document)

        kept_indices = set()
        for unit, score in zip(units, yes_probs):
            if score > self.threshold:
                kept_indices.update(unit.indices)

        return " ".join(sentences[i] for i in sorted(kept_indices))

    def compress_with_stats(self, query, document):
        sentences = self.decompose_sentences(document)
        
        if not sentences:
            return {
                "compressed_text": "",
                "original_length": 0,
                "compressed_length": 0,
                "compression_ratio": 0.0,
                "sentences_kept": 0,
                "sentences_total": 0,
                "evidence_units_total": 0,
                "evidence_units_kept_count": 0,
                "evidence_units_removed_count": 0,
                "total_tokens_consumed": 0,
                "all_units": [],
                "kept_units": [],
                "removed_units": []
            }

        graph = self.build_similarity_graph(sentences)
        units = self.extract_evidence_units(graph, sentences)

        unit_texts = [unit.text for unit in units]
        queries = [query] * len(unit_texts)
        yes_probs, total_tokens = self.exit._predict_batch(queries, unit_texts, document)

        kept_indices = set()
        all_units_info, kept_units_info, removed_units_info = [], [], []

        for unit, score in zip(units, yes_probs):
            kept = score > self.threshold
            info = {
                "text": unit.text, "sentences": unit.sentences,
                "indices": unit.indices, "start_idx": unit.start_idx,
                "end_idx": unit.end_idx, "score": score
            }
            all_units_info.append(info)
            if kept:
                kept_units_info.append(info)
                kept_indices.update(unit.indices)
            else:
                removed_units_info.append(info)

        compressed_text = " ".join(sentences[i] for i in sorted(kept_indices))
        return {
            "compressed_text": compressed_text,
            "original_length": len(document),
            "compressed_length": len(compressed_text),
            "compression_ratio": len(compressed_text) / len(document) if document else 0.0,
            "sentences_kept": len(kept_indices),
            "sentences_total": len(sentences),
            "evidence_units_total": len(units),
            "evidence_units_kept_count": len(kept_units_info),
            "evidence_units_removed_count": len(removed_units_info),
            "total_tokens_consumed": total_tokens,
            "all_units": all_units_info,
            "kept_units": kept_units_info,
            "removed_units": removed_units_info
        }