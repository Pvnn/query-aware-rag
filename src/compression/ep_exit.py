from dataclasses import dataclass
from typing import List
import networkx as nx
from sentence_transformers import SentenceTransformer, util

from src.compression.exit_baseline import ExitBaselineCompressor


# ---------------------------------------------------------
# Evidence Unit Dataclass
# ---------------------------------------------------------
@dataclass
class EvidenceUnit:
    sentences: List[str]
    start_idx: int
    end_idx: int

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


# ---------------------------------------------------------
# EP-EXIT Compressor
# ---------------------------------------------------------
class EPExitCompressor:
    """
    Evidence-Preserving EXIT (EP-EXIT)

    1) Group semantically similar nearby sentences using a graph
    2) Apply original EXIT classifier on evidence units
    3) Use the same EXIT threshold logic (0.5)
    """

    def __init__(
        self,
        token: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.65,
        locality_window: int = 2,
    ):
        """
        Args:
            token: HuggingFace token for EXIT model
            embedding_model: sentence embedding model for grouping
            similarity_threshold: threshold for semantic edge creation
            locality_window: max sentence distance allowed for grouping
        """

        print("Initializing EP-EXIT...")

        # Embedding model for grouping only
        self.embedder = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.locality_window = locality_window

        # Reuse original EXIT classifier
        self.exit = ExitBaselineCompressor(token=token)

        print("✓ EP-EXIT initialized\n")

    # ---------------------------------------------------------
    # STEP 1: Sentence Segmentation (reuse EXIT splitter)
    # ---------------------------------------------------------
    def split_into_sentences(self, text: str) -> List[str]:
        return self.exit.decompose_sentences(text)

    # ---------------------------------------------------------
    # STEP 2: Build Semantic Similarity Graph
    # ---------------------------------------------------------
    def build_similarity_graph(self, sentences: List[str]) -> nx.Graph:

        embeddings = self.embedder.encode(sentences, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeddings, embeddings)

        n = len(sentences)
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):

                # Locality constraint (prevents long-distance grouping)
                if abs(i - j) > self.locality_window:
                    continue

                if float(sim_matrix[i][j]) >= self.similarity_threshold:
                    G.add_edge(i, j)

        return G

    # ---------------------------------------------------------
    # STEP 3: Extract Evidence Units (Connected Components)
    # ---------------------------------------------------------
    def extract_evidence_units(
        self,
        graph: nx.Graph,
        sentences: List[str],
    ) -> List[EvidenceUnit]:

        units = []

        for component in nx.connected_components(graph):
            idxs = sorted(component)

            unit_sentences = [sentences[i] for i in idxs]

            units.append(
                EvidenceUnit(
                    sentences=unit_sentences,
                    start_idx=idxs[0],
                    end_idx=idxs[-1],
                )
            )

        return units

    # ---------------------------------------------------------
    # STEP 4: EXIT Classification on Evidence Units
    # ---------------------------------------------------------
    def filter_units(
        self,
        query: str,
        document: str,
        units: List[EvidenceUnit],
    ) -> List[EvidenceUnit]:

        selected_units = []

        for unit in units:
            score = self.exit.classify_sentence(
                query=query,
                sentence=unit.text,
                document=document,
            )

            # EXIT threshold logic (unchanged)
            if score > self.exit.threshold:
                selected_units.append(unit)

        return selected_units

    # ---------------------------------------------------------
    # STEP 5: Context Reconstruction
    # ---------------------------------------------------------
    def reconstruct_context(self, units: List[EvidenceUnit]) -> str:
        ordered = sorted(units, key=lambda u: u.start_idx)
        return " ".join(u.text for u in ordered)

    # ---------------------------------------------------------
    # FULL EP-EXIT PIPELINE
    # ---------------------------------------------------------
    def compress(self, query: str, document: str) -> str:

        # Step 1: Sentence split
        sentences = self.split_into_sentences(document)
        if not sentences:
            return ""

        # Step 2: Build similarity graph
        graph = self.build_similarity_graph(sentences)

        # Step 3: Extract evidence units
        units = self.extract_evidence_units(graph, sentences)

        # Step 4: Apply EXIT classification
        selected_units = self.filter_units(query, document, units)

        # Step 5: Reconstruct final compressed context
        return self.reconstruct_context(selected_units)