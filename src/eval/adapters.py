"""
Adapters for wrapping various compression baselines into a unified interface
for the GenerativeEvaluator.
"""
from src.eval.interfaces import SearchResult

class NoOpCompressor:
    def compress(self, query, docs):
        return {"compressed_docs": docs}

class ExitAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        compressed_docs = []
        for doc in docs:
            # Check if compressor returns a dict or directly SearchResults (like the official one)
            result = self.compressor.compress(query, [doc] if not isinstance(docs, list) else docs)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], SearchResult):
                return {"compressed_docs": result}
            
            text = result.get("compressed_text", result.get("final_text", "")) if isinstance(result, dict) else result
            compressed_docs.append(SearchResult(evi_id=doc.evi_id, docid=doc.docid, title=doc.title, text=text))
        return {"compressed_docs": compressed_docs}

class QuitoAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        compressed_docs = []
        for doc in docs:
            sentences = [s.strip() + "." for s in doc.text.split(".") if s.strip()]
            result = self.compressor.compress(query=query, sentences=sentences, compression_ratio=0.5)
            text = " ".join(result["filtered_sentences"])
            compressed_docs.append(SearchResult(evi_id=doc.evi_id, docid=doc.docid, title=doc.title, text=text))
        return {"compressed_docs": compressed_docs}

class HybridAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        compressed_docs = []
        for doc in docs:
            result = self.compressor.compress(query=query, context=doc.text, use_coarse=True, use_fine=True)
            compressed_docs.append(SearchResult(evi_id=doc.evi_id, docid=doc.docid, title=doc.title, text=result['final_text']))
        return {"compressed_docs": compressed_docs}

class RefinerAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class RecompAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class LLMLingua2Adapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class CompActAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class RecompExtractiveAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}