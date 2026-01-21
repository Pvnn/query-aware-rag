from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class DenseRetriever:
    def __init__(self, model_name="facebook/contriever-msmarco"):
        print(f"Initializing DenseRetriever with {model_name}...")
        self.encoder = SentenceTransformer(model_name)

        if torch.cuda.is_available():
            self.encoder = self.encoder.to('cuda')
            print(f"✓ Retriever on GPU: {torch.cuda.get_device_name(0)}")

        self.corpus = []
        self.embeddings = None

    def index_documents(self, documents):
        """
        Index a corpus of documents.

        Args:
            documents: List of document strings or dicts with 'text' key
        """
        print(f"Indexing {len(documents)} documents...")

        # Extract text if documents are dicts
        if isinstance(documents[0], dict):
            texts = [doc['text'] for doc in documents]
            self.corpus = documents
        else:
            texts = documents
            self.corpus = [{'text': doc} for doc in documents]

        # Encode documents
        self.embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        print(f"✓ Indexed {len(self.corpus)} documents")

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k most relevant documents.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")

        # Encode query
        query_emb = self.encoder.encode(
            query,
            convert_to_tensor=True
        )

        # Compute similarity scores
        scores = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0),
            self.embeddings
        )

        # Get top-k
        top_k = min(top_k, len(self.corpus))
        top_indices = torch.topk(scores, k=top_k).indices.cpu().numpy()
        top_scores = scores[top_indices].cpu().numpy()

        results = [
            (self.corpus[idx], float(score))
            for idx, score in zip(top_indices, top_scores)
        ]

        return results
