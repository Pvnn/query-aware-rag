import torch
import numpy as np
from .metadata_extractor import MetadataExtractor
from .model import BudgetPredictor

class BudgetInference:
    def __init__(self, model_path=None):
        self.extractor = MetadataExtractor()

        self.model = BudgetPredictor()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.k_values = [2, 4, 6, 8]

    def predict_k(self, query, docs, scores):
        features = self.extractor.extract(query, docs, scores)

        x = torch.tensor(features).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            pred_class = torch.argmax(logits, dim=1).item()

        return self.k_values[pred_class]
