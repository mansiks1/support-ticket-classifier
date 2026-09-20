# Verified progress

- Initialization: source/license inspected; protocol fixed before experiments.
- No trained model or measured quality is claimed at this stage.

- Data audit executed: 10003 raw, 9971 cleaned; 5 tests passed; source hashes saved.
- EDA and grouped split executed: 5992 fit / 1996 calibration / 1983 validation; disjoint groups verified.
- Baselines executed: Dummy validation macro-F1 0.0005, Logistic Regression 0.8097; model roundtrip and training tests passed.
- Classical and 8 one-factor preprocessing experiments executed: 17 total configurations. SVM C=0.5 leads validation at 0.8519 macro-F1. Test remains sealed.
- API implemented and exercised through FastAPI TestClient with trained synthetic fixture: 35 tests passed. Production artifact still awaits final selection.
- 41 local tests passed including calibration, input limits, split integrity and sealed-test guards. Added reproducible dependency locks and CI.
- DistilBERT completed on RTX 4050: validation macro-F1 0.868122, 80.898 s fit, 269013758 bytes. Includes pinned upstream revision and actual weights.
- Selection frozen before test: DistilBERT, temperature 0.7655438, threshold 0.75; validation coverage 0.820474 and accepted accuracy 0.950830. First GitHub Actions run passed.
- Final frozen test executed and exact verification passed: macro-F1 0.881932; overlap-excluded 0.872594; coverage 0.806494, accepted accuracy 0.959742. Real final-model HTTP smoke: 10 checks passed.
- Final-model startup warmup ensures health only reports ready after weights load. CPU serving measured; dependency locks and reproducibility tools prepared. Final Transformer container check pending Docker Engine recovery.
- Technical report, distinct article, model card and five resume bullets generated from measured results; two notebooks executed (5 + 7 cells), plots visually inspected.
- Release verification complete: clean CPU frozen metrics within 1e-6; final DistilBERT Docker image built and 10 real HTTP checks passed. All local documentation links valid.
