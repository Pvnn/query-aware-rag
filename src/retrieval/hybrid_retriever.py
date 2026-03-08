from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import torch

class HybridRetriever:
    def __init__(self, model_name="facebook/contriever-msmarco"):
        print(f"Initializing HybridRetriever with {model_name} and BM25...")
        
        # 1. Initialize Dense Retriever
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = SentenceTransformer(model_name, device=device)
        
        if device == 'cuda':
            print(f"✓ Dense Retriever on GPU: {torch.cuda.get_device_name(0)}")

        self.corpus = []
        self.embeddings = None
        self.bm25 = None  # 2. Placeholder for Sparse Retriever

    def _tokenize(self, text):
        """Simple tokenizer for BM25 (lowercases and splits by spaces)"""
        return text.lower().split()

    def index_documents(self, documents):
        """Index a corpus of documents for both Dense and Sparse retrieval."""
        print(f"Indexing {len(documents)} documents...")

        if isinstance(documents[0], dict):
            texts = [doc['text'] for doc in documents]
            self.corpus = documents
        else:
            texts = documents
            self.corpus = [{'text': doc} for doc in documents]

        # --- DENSE INDEXING ---
        self.embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        # --- SPARSE (BM25) INDEXING ---
        tokenized_corpus = [self._tokenize(doc) for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"✓ Indexed {len(self.corpus)} documents (Dense + Sparse)")

    def retrieve(self, query, top_k=5, rrf_k=60):
        """
        Retrieve using Reciprocal Rank Fusion (RRF).
        rrf_k is a smoothing constant (60 is the industry standard).
        """
        if self.embeddings is None or self.bm25 is None:
            raise ValueError("No documents indexed. Call index_documents() first.")

        # ==========================================
        # 1. DENSE RETRIEVAL (Semantic Context)
        # ==========================================
        query_emb = self.encoder.encode(query, convert_to_tensor=True)
        # Fetch more than top_k initially to ensure good fusion overlap
        dense_hits = util.semantic_search(query_emb, self.embeddings, top_k=top_k * 2)[0]
        
        # Create a dictionary of {corpus_id: dense_rank}
        dense_ranks = {hit['corpus_id']: rank for rank, hit in enumerate(dense_hits)}

        # ==========================================
        # 2. SPARSE RETRIEVAL (Exact Keyword Match)
        # ==========================================
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get the top indices based on BM25 scores
        sparse_indices = bm25_scores.argsort()[-(top_k * 2):][::-1]
        
        # Create a dictionary of {corpus_id: sparse_rank}
        sparse_ranks = {idx: rank for rank, idx in enumerate(sparse_indices)}

        # ==========================================
        # 3. RECIPROCAL RANK FUSION (RRF)
        # ==========================================
        rrf_scores = {}
        # Get all unique document IDs retrieved by either method
        all_retrieved_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))

        for doc_id in all_retrieved_ids:
            score = 0.0
            # Add dense RRF contribution (if it wasn't found, rank is effectively infinity)
            if doc_id in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[doc_id])
            # Add sparse RRF contribution
            if doc_id in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[doc_id])
            
            rrf_scores[doc_id] = score

        # ==========================================
        # 4. SORT AND RETURN TOP-K
        # ==========================================
        # Sort documents by their combined RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        results = [
            (self.corpus[doc_id], rrf_score)
            for doc_id, rrf_score in sorted_docs[:top_k]
        ]

        return results