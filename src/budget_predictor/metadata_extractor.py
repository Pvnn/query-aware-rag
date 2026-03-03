import numpy as np
import spacy
from collections import Counter
from sentence_transformers import SentenceTransformer

class MetadataExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def extract(self, query, retrieved_docs):
        """
        query: str
        retrieved_docs: list of dicts { "text": str, "score": float }

        returns: numpy array of shape (46,)
        """
        # ---------------------------
        # A. QUERY FEATURES (30 dims)
        # ---------------------------
        doc = self.nlp(query)
        tokens = [t.text for t in doc]

        # Basic text stats
        q_len = len(tokens)
        q_char_len = len(query)
        avg_tok_len = q_char_len / q_len if q_len > 0 else 0

        # WH-words
        wh_words = sum(t.lower_ in ["what", "who", "when", "where", "why", "how"] for t in doc)

        # Named entities
        ents = doc.ents
        ent_count = len(ents)

        ent_type_counts = Counter([ent.label_ for ent in ents])
        ent_features = [
            ent_type_counts.get(label, 0)
            for label in ["PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT"]
        ]

        # POS tag distribution
        pos_counts = Counter([t.pos_ for t in doc])
        all_pos = ["NOUN", "VERB", "PROPN", "ADJ", "ADV",
                   "ADP", "DET", "PRON", "NUM", "AUX"]

        pos_features = [
            pos_counts.get(pos, 0) / q_len if q_len > 0 else 0
            for pos in all_pos
        ]

        # punctuation
        punct_count = sum(t.is_punct for t in doc)

        # readability proxy: avg sentence length
        sent_lens = [len(s.text.split()) for s in doc.sents]
        read_score = np.mean(sent_lens) if sent_lens else q_len

        # question type flags (one-hot)
        qt = query.lower()
        qtype = [
            int(qt.startswith("what")),
            int(qt.startswith("who")),
            int(qt.startswith("when")),
            int(qt.startswith("why")),
            int(qt.startswith("how")),
            int(not any(qt.startswith(w) for w in ["what", "who", "when", "why", "how"]))
        ]

        # token entropy (simple approx)
        freq = Counter(tokens)
        probs = np.array([freq[t] / q_len for t in tokens]) if q_len > 0 else np.array([1])
        entropy = -np.sum(probs * np.log(probs + 1e-9))

        query_features = [
            q_len, q_char_len, avg_tok_len,
            wh_words, ent_count, punct_count,
            read_score, entropy
        ] + ent_features + pos_features + qtype

        assert len(query_features) == 30

        # ------------------------------------
        # B. RETRIEVAL FEATURES (16 dims)
        # ------------------------------------
        scores = [d["score"] for d in retrieved_docs]
        docs = [d["text"] for d in retrieved_docs]

        # Score stats
        min_s = float(np.min(scores))
        max_s = float(np.max(scores))
        mean_s = float(np.mean(scores))
        var_s = float(np.var(scores))

        # Doc length stats
        doc_lens = [len(d.split()) for d in docs]

        min_l = float(np.min(doc_lens))
        max_l = float(np.max(doc_lens))
        mean_l = float(np.mean(doc_lens))
        var_l = float(np.var(doc_lens))

        # Entity density in docs
        doc_ent_counts = [len(self.nlp(d).ents) for d in docs]
        mean_ent = float(np.mean(doc_ent_counts))
        max_ent = float(np.max(doc_ent_counts))

        # sentence count in docs
        sent_counts = [len(list(self.nlp(d).sents)) for d in docs]
        mean_sents = float(np.mean(sent_counts))
        max_sents = float(np.max(sent_counts))

        # semantic diversity
        embeddings = self.embedder.encode(docs)
        semantic_div = float(np.mean(np.var(embeddings, axis=0)))

        # score slope (smoothness)
        sorted_scores = sorted(scores, reverse=True)
        score_slope = sorted_scores[0] - sorted_scores[-1]

        retrieval_features = [
            min_s, max_s, mean_s, var_s,
            min_l, max_l, mean_l, var_l,
            mean_ent, max_ent,
            semantic_div, score_slope,
            max_s - min_s, max_l - min_l,
            mean_sents, max_sents
        ]

        assert len(retrieval_features) == 16

        # -----------------------------
        # CONCAT → 46-D VECTOR
        # -----------------------------
        vector = np.array(query_features + retrieval_features, dtype=np.float32)
        return vector