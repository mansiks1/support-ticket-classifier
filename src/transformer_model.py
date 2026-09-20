"""Lazy adapter. Requires optional transformer dependencies and local weights."""

import numpy as np

from src.config import MODELS


class TransformerModel:
    def __init__(self, directory, classes):
        self.directory = directory
        self.classes_ = np.asarray(classes)
        self._model = None
        self._tokenizer = None

    def __getstate__(self):
        return {
            "directory": self.directory,
            "classes_": self.classes_,
            "_model": None,
            "_tokenizer": None,
        }

    def decision_function(self, texts):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                MODELS / self.directory, local_files_only=True
            )
            self._model = (
                AutoModelForSequenceClassification.from_pretrained(
                    MODELS / self.directory, local_files_only=True
                )
                .to(device)
                .eval()
            )
        chunks = []
        texts = list(texts)
        with torch.inference_mode():
            for i in range(0, len(texts), 32):
                batch = self._tokenizer(
                    texts[i : i + 32],
                    truncation=True,
                    padding=True,
                    max_length=96,
                    return_tensors="pt",
                ).to(device)
                chunks.append(self._model(**batch).logits.cpu().numpy())
        return np.concatenate(chunks)
