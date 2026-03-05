from dataclasses import dataclass
from typing import List
import networkx as nx
from sentence_transformers import SentenceTransformer, util

from src.compression.exit_baseline import ExitBaselineCompressor

@dataclass
class EvidenceUnit:
  sentences: List[str]
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
          start_idx=idxs[0],
          end_idx=idxs[-1],
        )
      )
    return units

  def compress(self, query, document):
    sentences = self.decompose_sentences(document)
    if not sentences:
      return ""
      
    graph = self.build_similarity_graph(sentences)
    units = self.extract_evidence_units(graph, sentences)
    
    selected_units = []
    for unit in units:
      score, _ = self.classify_sentence(query, unit.text, document)
      if score > self.threshold:
        selected_units.append(unit)
        
    ordered = sorted(selected_units, key=lambda u: u.start_idx)
    return " ".join(u.text for u in ordered)

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
    
    all_units_info = []
    kept_units_info = []
    removed_units_info = []
    sentences_kept_count = 0
    total_tokens_consumed = 0
    
    for unit in units:
      score, token_count = self.classify_sentence(query, unit.text, document)
      total_tokens_consumed += token_count
      kept = score > self.threshold
      
      unit_info = {
        "text": unit.text,
        "sentences": unit.sentences,
        "start_idx": unit.start_idx,
        "end_idx": unit.end_idx,
        "score": score
      }
      
      all_units_info.append(unit_info)
      
      if kept:
        kept_units_info.append(unit_info)
        sentences_kept_count += len(unit.sentences)
      else:
        removed_units_info.append(unit_info)
        
    ordered_kept = sorted(kept_units_info, key=lambda u: u["start_idx"])
    compressed_text = " ".join(u["text"] for u in ordered_kept)
    
    return {
      "compressed_text": compressed_text,
      "original_length": len(document),
      "compressed_length": len(compressed_text),
      "compression_ratio": len(compressed_text) / len(document) if len(document) > 0 else 0.0,
      "sentences_kept": sentences_kept_count,
      "sentences_total": len(sentences),
      "evidence_units_total": len(units),
      "evidence_units_kept_count": len(kept_units_info),
      "evidence_units_removed_count": len(removed_units_info),
      "total_tokens_consumed": total_tokens_consumed,
      "all_units": all_units_info,
      "kept_units": kept_units_info,
      "removed_units": removed_units_info
    }