"""Optional actual DistilBERT experiment; test remains sealed."""

import argparse
import json
import time

import numpy as np

from src.config import MODELS, REPORTS, SEED
from src.data import load_split, save_json
from src.metrics import quality

MODEL_ID = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def main(epochs=2):
    import torch
    from scipy.special import softmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    if (REPORTS / "selection.json").exists():
        raise RuntimeError("Selection frozen")
    set_seed(SEED)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = load_split("train"), load_split("validation")
    classes = np.array(sorted(train.category.unique()))
    mapping = {label: i for i, label in enumerate(classes)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        num_labels=len(classes),
        id2label=dict(enumerate(classes)),
        label2id=mapping,
    ).to(device)
    encoded = tokenizer(
        train.text.tolist(),
        truncation=True,
        padding="max_length",
        max_length=96,
        return_tensors="pt",
    )
    labels = torch.tensor(train.category.map(mapping).to_numpy())
    data = torch.utils.data.TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
    generator = torch.Generator().manual_seed(SEED)
    loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for ids, mask, y in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device)
            )
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += output.loss.item() * len(y)
        history.append({"epoch": epoch + 1, "train_loss": total / len(data)})
        print(history[-1], flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    model.eval()

    def predict(texts):
        outputs = []
        with torch.inference_mode():
            for i in range(0, len(texts), 32):
                batch = tokenizer(
                    texts[i : i + 32],
                    truncation=True,
                    padding=True,
                    max_length=96,
                    return_tensors="pt",
                ).to(device)
                outputs.append(model(**batch).logits.cpu().numpy())
        return softmax(np.concatenate(outputs), axis=1)

    texts = val.text.tolist()
    predict(texts[:1])
    timing = []
    p = None
    for _ in range(5):
        start = time.perf_counter()
        p = predict(texts)
        timing.append((time.perf_counter() - start) * 1000 / len(texts))
    single = []
    for _ in range(5):
        start = time.perf_counter()
        predict(texts[:1])
        single.append((time.perf_counter() - start) * 1000)
    directory = MODELS / "distilbert"
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    row = {
        "name": "distilbert_2epochs",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "config": {"method": "transformer", "epochs": epochs},
        "license": "Apache-2.0",
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "epochs": epochs,
        "seed": SEED,
        "lr": 5e-5,
        "batch_size": 16,
        "max_length": 96,
        "train_seconds": train_seconds,
        "validation": quality(val.category, p, classes),
        "batch_ms_per_item": float(np.median(timing)),
        "single_ms": float(np.median(single)),
        "model_bytes": sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()),
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2
        if device.type == "cuda"
        else None,
        "history": history,
        "torch": torch.__version__,
    }
    save_json(REPORTS / "transformer.json", row)
    import joblib
    from src.transformer_model import TransformerModel

    joblib.dump(TransformerModel("distilbert", classes), MODELS / "distilbert_2epochs.joblib")
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    main(parser.parse_args().epochs)
