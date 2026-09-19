"""Load only trusted local artifacts. No text is logged or persisted."""

import joblib
import numpy as np

from src.metrics import probabilities


class Predictor:
    def __init__(self, path):
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.metadata = artifact["metadata"]

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not 1 <= len(texts) <= 64:
            raise ValueError("Supply a text or a list of 1..64 texts")
        for text in texts:
            if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                raise ValueError("Texts must be nonblank strings of at most 4000 characters")
        p = probabilities(self.model, texts, self.metadata["temperature"])
        classes = np.asarray(self.model.classes_)
        response = []
        for row in p:
            index = int(row.argmax())
            confidence = float(row[index])
            human = confidence < self.metadata["policy"]["threshold"]
            label = str(classes[index])
            response.append(
                {
                    "category": None if human else label,
                    "suggested_category": label,
                    "confidence": confidence,
                    "priority": None,
                    "requires_human": human,
                }
            )
        return response
