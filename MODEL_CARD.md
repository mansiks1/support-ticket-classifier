# Model card

- Final model: distilbert_2epochs, fine-tuned distilbert-base-uncased (Apache-2.0).
- Task: English banking intent classification, 77 classes; single label.
- Dataset: BANKING77, PolyAI / Casanueva et al. 2020, CC BY 4.0; no ownership claim.
- Training: 5992 fit samples, seed 2026, 2 epochs, AdamW 5e-5, batch 16, 96 tokens.
- Calibration: 1996 samples; temperature 0.765544; validation threshold 0.75.
- Official test macro-F1: 0.8819; decontaminated test: 0.8726.
- Test accepted accuracy: 95.97% at 80.65% coverage.
- Intended use: supervised portfolio demonstration and in-domain routing experiments.
- Not intended: autonomous financial decisions, fraud detection, priority assessment,
  unrestricted multilingual or open-domain customer support.
- Known failure: weather question accepted as card delivery; confidence is not an OOD guarantee.
- Input limit: 4000 characters / 96 model tokens; batch max 64. Priority unavailable.
- No raw request text is persisted. API has no authentication or rate limiting.
- Load joblib only from trusted release after verifying SHA-256. Weight hashes:
  [selection.json](reports/selection.json). Details: [technical report](reports/technical-report.md).
