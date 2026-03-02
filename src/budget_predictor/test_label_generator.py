from src.data.data_loader import HotpotQALoader
from src.retrieval.retriever import DenseRetriever
from src.generation.reader import RAGReader
from src.budget_predictor.label_generator import LabelGenerator

print("Loading dataset...")
loader = HotpotQALoader("data/hotpotqa/dev")

samples = [loader[i] for i in range(10)]
print(f"Loaded {len(samples)} samples")

print("\nBuilding test corpus...")
corpus = []
for sample in samples:
    for doc in sample.documents:
        corpus.append(doc.text)

print(f"Corpus size: {len(corpus)} documents\n")

retriever = DenseRetriever()
retriever.index_documents(corpus)

reader = RAGReader("llama3.1:8b")

label_gen = LabelGenerator(retriever, reader)

print("\n===== Testing Label Generator =====")

for sample in samples:
    print("\n--------------------------------------")
    print("Question:", sample.question)
    print("Gold Answer:", sample.answer)

    true_K = label_gen.find_smallest_k(sample.question, sample.answer)
    print("TRUE K label:", true_K)

    retrieved = retriever.retrieve(sample.question, top_k=10)
    docs = [doc["text"] for doc, _ in retrieved]
    scores = [score for _, score in retrieved]

    metadata = label_gen.extractor.extract(
        sample.question,
        [{"text": t, "score": s} for t, s in zip(docs, scores)]
    )

    print("Metadata shape:", metadata.shape)