# Verified progress

- Initialization: source/license inspected; protocol fixed before experiments.
- No trained model or measured quality is claimed at this stage.

- Data audit executed: 10003 raw, 9971 cleaned; 5 tests passed; source hashes saved.
- EDA and grouped split executed: 5992 fit / 1996 calibration / 1983 validation; disjoint groups verified.
- Baselines executed: Dummy validation macro-F1 0.0005, Logistic Regression 0.8097; model roundtrip and training tests passed.
- Classical and 8 one-factor preprocessing experiments executed: 17 total configurations. SVM C=0.5 leads validation at 0.8519 macro-F1. Test remains sealed.
- API implemented and exercised through FastAPI TestClient with trained synthetic fixture: 35 tests passed. Production artifact still awaits final selection.
