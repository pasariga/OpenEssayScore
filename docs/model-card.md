# Model card / 模型说明

- Canonical toolkit: OpenEssayScore 0.1.0. Historical model ID: `wenheng-score-v1-20261022`; model not renamed or retrained.
- Family: character TF-IDF (1–3 grams, min_df=2, max_features=40000, sublinear_tf) + Ridge alpha=0.1. Vocabulary and IDF fit on 751 training essays only. Dev QWK selects model, MAE breaks ties.
- Targets: two CFMS-34 human holistic ratings summed to 0–10. No teacher feedback generator and no fine-grained output in the selected model.
- Preprocessing: NFKC, strip, email/mobile redaction; essay text only. Predictions clipped to 0–10; QWK rounds with NumPy `rint` (ties to even). API requires ≥20 Chinese characters, ≤20000 characters.
- Default output is a research estimate, with manual review required. No validated probability or deployed risk triage.
- Intended users: researchers and developers with authorized inputs; unsuitable for automatic high-stakes grading, ranking, discipline, or unreviewed student decisions.

| Artifact | SHA-256 |
| --- | --- |
| `best_scorer.joblib` | `c28c5010ae4f1170aba860ecbe9f2393f5a723311c59f89db5ebbc2f773193cb` |
| `lexical_vectorizer.joblib` | `e098d09fb07b81320e87a1c2e96cba1387525f64ca93414c279d66a0070a667b` |

Weights are private pending publication-rights clarification. Fingerprints enable integrity checks; they do not provide download access or prove data rights. Joblib is executable serialization: use only artifacts whose provenance you independently trust. Registry manifests validate individual files before loading, preserve immutable versions, and permit explicit rollback. Do not overwrite a known model ID with new bytes.

Bias/limits: small single-source corpus, rater agreement selection in test, score imbalance, no verified grade/prompt/student split, high-score underestimation, substantial overfit gap, unknown out-of-domain behavior. Title grouping in internal CV is not proof of cross-prompt generalization. A draft 20-point rubric from R3 is a separate unvalidated design; multiplying this model's scores by two does not implement it.
