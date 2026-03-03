from dataclasses import dataclass
from typing import List
from datasets import load_from_disk


@dataclass
class Document:
    title: str
    text: str
    score: float = 1.0


@dataclass
class QAExample:
    question: str
    answer: str
    documents: List[Document]
    supporting_facts: dict | None = None


class HotpotQALoader:
    def __init__(self, data_path: str):
        self.dataset = load_from_disk(data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        documents = []
        title_to_text = {}

        # Build documents (unchanged behavior)
        for title, sentences in zip(
            sample["context"]["title"],
            sample["context"]["sentences"]
        ):
            text = " ".join(sentences)
            doc = Document(title=title, text=text)
            documents.append(doc)
            title_to_text[title] = text

        # ---- FIX: enrich supporting_facts robustly
        sf = sample.get("supporting_facts")

        if sf is not None:
            supporting_titles = list(set(sf.get("title", [])))

            supporting_docs = [
                {
                    "title": t,
                    "text": title_to_text[t]
                }
                for t in supporting_titles
                if t in title_to_text
            ]

            gold_context = "\n\n".join(
                d["text"] for d in supporting_docs
            )

            supporting_facts = {
                "titles": supporting_titles,
                "documents": supporting_docs,
                "gold_context": gold_context,
                "raw": sf,  # preserve original HF structure
            }
        else:
            supporting_facts = None

        return QAExample(
            question=sample["question"],
            answer=sample["answer"],
            documents=documents,
            supporting_facts=supporting_facts,
        )