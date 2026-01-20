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
    supporting_facts: List | None = None


class HotpotQALoader:
    def __init__(self, data_path: str):
        self.dataset = load_from_disk(data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        documents = [
            Document(
                title=title,
                text=" ".join(sentences)
            )
            for title, sentences in zip(
                sample["context"]["title"],
                sample["context"]["sentences"]
            )
        ]

        return QAExample(
            question=sample["question"],
            answer=sample["answer"],
            documents=documents,
            supporting_facts=sample.get("supporting_facts")
        )
