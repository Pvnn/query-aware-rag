from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import spacy
from functools import lru_cache
from typing import List, Tuple

class ExitBaselineCompressor:
    def __init__(
        self,
        token,
        model_name="doubleyyh/exit-gemma-2b",
        threshold=0.5,
        batch_size=8
    ):
        self.threshold = threshold
        self.batch_size = batch_size
        print(f"Initializing ExitBaselineCompressor...")
        print(f"Model: {model_name}, Threshold: {threshold}, Batch size: {batch_size}")

        if not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available, using CPU (will be slow)")
        else:
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print("Loading model with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            token=token
        )
        self.model.eval()

        print("Loading spaCy sentence tokenizer...")
        self.nlp = spacy.load("en_core_web_sm")

        # Pre-cache yes/no token IDs — same as official impl
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

        print(f"✓ Compressor initialized on device: {self.model.device}\n")

    def decompose_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    @lru_cache(maxsize=1024)
    def _build_prompt(self, query: str, sentence: str, document: str) -> str:
        """Cached prompt construction — free speedup for repeated queries."""
        return (
            f'<start_of_turn>user\n'
            f'Query:\n{query}\n'
            f'Full context:\n{document}\n'
            f'Sentence:\n{sentence}\n'
            f'Is this sentence useful in answering the query? '
            f'Answer only "Yes" or "No".<end_of_turn>\n'
            f'<start_of_turn>model\n'
        )

    def _predict_batch(
        self,
        queries: List[str],
        sentences: List[str],
        document: str
    ) -> Tuple[List[float], int]:
        """
        Score a batch of sentences in a single forward pass.
        Returns (list of yes_probs, total_tokens_in_batch).
        """
        prompts = [
            self._build_prompt(q, s, document)
            for q, s in zip(queries, sentences)
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,          # pad shorter sequences to match longest
            truncation=True,
            max_length=4096,
            return_attention_mask=True
        )
        total_tokens = inputs["input_ids"].numel()  # batch_size × seq_len
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = self.model(**inputs)
            # Only need the last token logits for Yes/No classification
            last_token_logits = outputs.logits[:, -1, :]
            relevant_logits = torch.stack([
                last_token_logits[:, self.yes_token_id],
                last_token_logits[:, self.no_token_id]
            ], dim=1)
            probs = torch.softmax(relevant_logits, dim=1)
            yes_probs = probs[:, 0].tolist()

        return yes_probs, total_tokens

    def classify_sentence(
        self, query: str, sentence: str, document: str
    ) -> Tuple[float, int]:
        """
        Single-sentence classification — unchanged signature.
        Internally just calls _predict_batch with batch_size=1.
        """
        yes_probs, token_count = self._predict_batch([query], [sentence], document)
        return yes_probs[0], token_count

    def compress(self, query: str, document: str, threshold=None) -> str:
        """Unchanged signature. Now processes sentences in batches."""
        active_threshold = threshold if threshold is not None else self.threshold
        sentences = self.decompose_sentences(document)
        if not sentences:
            return ""

        selected = []
        # Process in chunks of batch_size
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            queries = [query] * len(batch)
            yes_probs, _ = self._predict_batch(queries, batch, document)
            for sent, score in zip(batch, yes_probs):
                if score > active_threshold:
                    selected.append(sent)

        return " ".join(selected)

    def compress_with_stats(self, query: str, document: str) -> dict:
        """Unchanged signature. Batched internally."""
        sentences = self.decompose_sentences(document)
        if not sentences:
            return {
                "compressed_text": "", "original_length": 0,
                "compressed_length": 0, "compression_ratio": 0.0,
                "sentences_kept": 0, "sentences_total": 0,
                "total_tokens_consumed": 0, "sentence_scores": []
            }

        sentence_scores = []
        selected = []
        total_tokens = 0

        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            queries = [query] * len(batch)
            yes_probs, token_count = self._predict_batch(queries, batch, document)
            total_tokens += token_count

            for sent, score in zip(batch, yes_probs):
                kept = score > self.threshold
                sentence_scores.append({"sentence": sent, "score": score, "kept": kept})
                if kept:
                    selected.append(sent)

        compressed_text = " ".join(selected)
        return {
            "compressed_text": compressed_text,
            "original_length": len(document),
            "compressed_length": len(compressed_text),
            "compression_ratio": len(compressed_text) / len(document) if document else 0,
            "sentences_kept": len(selected),
            "sentences_total": len(sentences),
            "total_tokens_consumed": total_tokens,
            "sentence_scores": sentence_scores
        }