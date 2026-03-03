from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import spacy


class ExitBaselineCompressor:
    """
    EXIT baseline compressor using pre-trained doubleyyh/exit-gemma-2b model.
    Performs context-aware sentence-level extractive compression.
    """

    def __init__(
        self,
        token,
        model_name="doubleyyh/exit-gemma-2b",
        threshold=0.5
    ):
        print(f"Initializing ExitBaselineCompressor...")
        print(f"Model: {model_name}")
        print(f"Threshold: {threshold}")

        if not torch.cuda.is_available():
            print("⚠️ Warning: CUDA not available, using CPU (will be slow)")
        else:
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")

        self.threshold = threshold

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

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

        print("Loading spaCy sentence tokenizer...")
        self.nlp = spacy.load("en_core_web_sm")

        print(f"✓ Compressor initialized on device: {self.model.device}\n")

    def decompose_sentences(self, text):
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return [s for s in sentences if len(s) > 0]

    def classify_sentence(self, query, sentence, document):
        prompt = f"Query: {query}\nDocument: {document}\nSentence: {sentence}\nRelevant?"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]

        yes_token = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token = self.tokenizer.encode("No", add_special_tokens=False)[0]

        probs = torch.softmax(logits[[yes_token, no_token]], dim=0)
        yes_prob = probs[0].item()

        return yes_prob

    def compress(self, query, document):
        sentences = self.decompose_sentences(document)

        if len(sentences) == 0:
            return ""

        selected_sentences = []

        for sent in sentences:
            score = self.classify_sentence(query, sent, document)
            if score > self.threshold:
                selected_sentences.append(sent)

        return " ".join(selected_sentences)

    def compress_with_stats(self, query, document):
        sentences = self.decompose_sentences(document)

        if len(sentences) == 0:
            return {
                "compressed_text": "",
                "original_length": 0,
                "compressed_length": 0,
                "compression_ratio": 0.0,
                "sentences_kept": 0,
                "sentences_total": 0,
                "sentence_scores": []
            }

        selected_sentences = []
        sentence_scores = []

        for sent in sentences:
            score = self.classify_sentence(query, sent, document)

            sentence_scores.append({
                "sentence": sent,
                "score": score,
                "kept": score > self.threshold
            })

            if score > self.threshold:
                selected_sentences.append(sent)

        compressed_text = " ".join(selected_sentences)

        return {
            "compressed_text": compressed_text,
            "original_length": len(document),
            "compressed_length": len(compressed_text),
            "compression_ratio": len(compressed_text) / len(document)
            if len(document) > 0 else 0,
            "sentences_kept": len(selected_sentences),
            "sentences_total": len(sentences),
            "sentence_scores": sentence_scores
        }