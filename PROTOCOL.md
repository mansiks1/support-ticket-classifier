# Preregistered experiment protocol

Seed 2026; Python 3.12; CPU threads fixed to one for classical experiments.
Input: an English banking query. Output: suggested intent, confidence,
automatic category only when accepted, requires_human, priority=null.
Users: support routing teams. Wrong routing can delay help with lost cards or
unrecognised transactions; confidence is not a safety guarantee or fraud detector.

## Data and evaluation boundary

BANKING77, PolyAI / Casanueva et al. (2020), CC BY 4.0.
Upstream commit 57ec275d8078af65b7731c2a98be812d844a6d6b.
Use only official train.csv before final freeze; download test.csv only after
reports/selection.json exists. Never use test to change a model or threshold.
Validate strings and labels; remove invalid and conflicting normalised duplicates;
collapse consistent normalised duplicates. Build near-duplicate connected components
using character 3-5 TF-IDF cosine >=0.92, then StratifiedGroupKFold to allocate
approximately 60% fit, 20% calibration, 20% validation (holdout fold 0, seed 2026).
The character representation used solely for duplicate detection is not a model
feature transform. No customer IDs, timestamps or dialogue IDs are supplied; this
limits group/time leakage checks. All predictive feature fitting uses fit only.
At final evaluation report official test and sensitivity excluding near/exact
matches against upstream development texts. No post-test retraining or selection.

## Bounded search (validation only)

Baseline: most-frequent Dummy and word (1,2) TF-IDF LogisticRegression C=1.
Classical: LinearSVC C in [0.5,1,2]; ComplementNB alpha in [0.1,1]; LR C in [0.5,2].
One-factor LR ablations against C=1: preserve case, remove punctuation,
English stop words, WordNet noun lemmatisation, unigrams, char_wb (3,5),
max_features=5000, class_weight=balanced. Base max_features=30000, min_df=2.
Defaults documented in persisted parameters. No combinatorial sweep.
Transformer: a separate reproducible DistilBERT script; run only if resources
support a meaningful experiment. Unrun work has no result row and no quality claim.

## Decision rule

Primary metric macro-F1 to weight all intents equally; also accuracy, weighted-F1,
class precision/recall, confusion, top-2 and macro average precision (OVR).
Research winner: highest validation macro-F1. Practical winner: smallest saved
model within 0.01 absolute macro-F1 of best non-dummy; break ties by inference time.
This is an explicit engineering heuristic, not a significance test.
Measure fit wall seconds, median warm batch latency per item, median single-query
latency, model bytes, process RSS (not isolated peak memory). Repeat latency 5x.
Calibrate the practical model with scalar temperature fitted on calibration NLL;
retain calibrated confidence only if validation NLL improves. SVM raw softmax
is a score normalisation, not an already calibrated probability.
Measure NLL, multiclass Brier, ECE (10 equal-width bins). Explore thresholds
0.00..0.99 step .01; choose maximum coverage with accepted validation accuracy
>=0.95 and >=100 accepted examples; if none qualify, abstain on all (threshold=1.01).
The target is empirical, not a guarantee; show counts and test Wilson interval.
Class selection uses frozen fit model, no train+validation refit.

## Analysis and reproducibility

Persist configs, hashes, package versions, splits, timings and metrics as JSON/CSV.
Interpret linear coefficients; inspect at least 10 correct and 10 wrong validation
predictions, ambiguity and low confidence. Error causes are human hypotheses, not
automatically proven label errors. Synthetic robustness probes are tests, not data.
Freeze configuration and artifact hash before one final test evaluation command.
Test labels used only for final metrics. API validates 1..4000 characters and 1..64
texts; supports arbitrary Unicode; blank input rejected. OOD is not solved by
in-domain confidence. Reports must say so explicitly.
