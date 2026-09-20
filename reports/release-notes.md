# v1.0.0 — reproducible portfolio release

English BANKING77 intent classification with 17 classical configurations,
one actual two-epoch DistilBERT run, temperature calibration and human routing.

- Final DistilBERT: official test macro-F1 **0.881932**, accuracy **0.883117**.
- Overlap-excluded sensitivity: macro-F1 **0.872594**, 2,713 remaining examples.
- Frozen confidence threshold 0.75: **80.65% coverage**, **95.97% accepted accuracy**.
- Selection uses validation only; training, calibration and validation groups are disjoint.
- 41 automated tests; executed notebooks; FastAPI; CPU Docker deployment.
- Model bundle includes calibrated adapter, fine-tuned DistilBERT weights/tokenizer,
  research adapter and SVM C=0.5 alternative. SHA-256 in MODEL_BUNDLE.sha256.
- Original code MIT; BANKING77 CC BY 4.0; DistilBERT Apache-2.0. Attribution bundled.

Known limits: English/banking only, single seed and one neural configuration,
no priority labels, no reliable OOD rejection, maximum 96 model tokens. A weather
question is a documented confident OOD failure. No real service SLA is claimed.

[Technical report](https://github.com/mansiks1/support-ticket-classifier/blob/v1.0.0/reports/technical-report.md)
· [Article](https://github.com/mansiks1/support-ticket-classifier/blob/v1.0.0/article/article.md)
· [Instructions](https://github.com/mansiks1/support-ticket-classifier/blob/v1.0.0/README.md)
· [Verification evidence](https://github.com/mansiks1/support-ticket-classifier/blob/v1.0.0/reports/verification.json)
