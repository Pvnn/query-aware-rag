from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import torch
import re

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

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by",
        "for", "if", "in", "into", "is", "it", "no", "not", "of",
        "on", "or", "such", "that", "the", "their", "then", "there",
        "these", "they", "this", "to", "was", "will", "with", "what",
        "how", "where", "who", "why", "which"
    }

    def _tokenize(self, text):
        """
        Improved tokenizer for BM25.
        1. Lowercases text.
        2. Removes punctuation (keeps only alphanumeric words).
        3. Removes common stop words.
        """
        # Extract only alphanumeric words using regex
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stopwords
        return [w for w in words if w not in self.STOP_WORDS]

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
        Normalized so the top theoretical score equals 1.0 for the UI.
        """
        if self.embeddings is None or self.bm25 is None:
            raise ValueError("No documents indexed. Call index_documents() first.")

        query_emb = self.encoder.encode(query, convert_to_tensor=True)
        dense_hits = util.semantic_search(query_emb, self.embeddings, top_k=top_k * 2)[0]
        
        # 1-indexed ranks (Rank 1 is best)
        dense_ranks = {hit['corpus_id']: rank + 1 for rank, hit in enumerate(dense_hits)}

        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_indices = bm25_scores.argsort()[-(top_k * 2):][::-1]
        
        # 1-indexed ranks (Rank 1 is best)
        sparse_ranks = {idx: rank + 1 for rank, idx in enumerate(sparse_indices)}

        rrf_scores = {}
        all_retrieved_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))

        for doc_id in all_retrieved_ids:
            score = 0.0
            if doc_id in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[doc_id])
            
            rrf_scores[doc_id] = score

        # The absolute maximum possible score (Rank 1 in both dense and sparse)
        max_possible_rrf = 2.0 / (rrf_k + 1)
        
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        results = []
        for doc_id, raw_score in sorted_docs[:top_k]:
            # Normalize so the UI progress bars look correct (scale from 0 to 1)
            normalized_score = raw_score / max_possible_rrf
            results.append((self.corpus[doc_id], normalized_score))

        return results