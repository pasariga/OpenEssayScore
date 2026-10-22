# Evidence and experiments

All reported accuracy numbers come from previously completed research, not the open-source packaging checks. No new model selection, old-test tuning, teacher scores, GPU training, paid API calls, or production deployment took place for packaging.

## R1 — `round1-cfms34-bge-v1`

Publisher revision `933489269eb9dbf5f6f00f8e8be6ba0ccebc3dbb`; official train/dev/test 751/100/100. Four model families compared on dev; fixed Ridge grid 0.1/1/10/100/1000. Highest dev QWK wins, then lower MAE. Character TF-IDF + Ridge selected; BGE-based candidates were not substituted after test inspection. The GPU was used only for frozen semantic comparison features (measured dense stage ~4.413 s wall time, not billing time).

Historical selected-model test: MAE 0.95453, RMSE 1.13993, QWK 0.35989, Pearson about 0.5524. Train MAE about 0.094 and QWK 1.0 reveal overfitting. Clipping prevents outputs outside rubric bounds; it does not repair calibration/high-score underestimation.

QWK implementation: `cohen_kappa_score(np.rint(y), np.rint(np.clip(p,0,10)), labels=list(range(11)), weights='quadratic')`. MAE uses continuous clipped predictions. Different rounding/label assumptions can change kappa. Test distribution is restricted to agreeing raters and totals 6/8/10; broader human-score agreement cannot be inferred. No new test claims may reuse this exposed set for tuning.

## R2 — `round2-audit-cv-v1`

Training-only 5 outer × 3 inner title-group CV, 390 CPU fits. No new independent test. Vocabulary/scalers/estimators fit within folds; title grouping reduces exact-title leakage but true prompt/student groups are unavailable. Lexical MAE 0.925/QWK 0.320, fusion 0.904/0.353, shared low-rank multitask 0.935/0.439. The fusion difference was not stable enough to promote; no v2. Three invalid fine-grained training cells were quarantined, not silently repaired; handwriting trait excluded for text-only work.

## R3 — `round3-rubric-protocol-v1`

Draft `WH-P34-NARR-0.1-draft`: grade-3/4 narrative/person/observation, four 0–5 dimensions, total 20. Original proposal informed by curriculum and CFMS categories; no teacher validation. Empty templates only; zero new teacher records. Separate target from CFMS10. Reused R2 out-of-fold residuals for internal exploratory diagnostics; no new accuracy validation.

## R4 — `round4-trust-review-v1`

Four mutually exclusive roles per rotation: score-replica fitting (~300), risk learning (~150), interval calibration (~150), validation (~150), five rotations over the original 751 training essays. Already used in previous research: internal evidence only. Large error defined as absolute score error >1.0/10. Lexical, length, semantic-distance and model-disagreement features were evaluated; disagreement was not assumed to be true error. Auxiliary invalid labels were not risk supervision.

Full risk AUROC 0.51186; reviewed 22.10%, caught 21.54% of large errors. Random comparison reviewed 20.11%, caught 19.94%. Retained coverage 77.9%, retained MAE 0.9348 vs overall 0.9363; this is selection-conditioned error, not improved scoring. Gate failed; stop tuning and leave routing disabled. CPU ~7.64 s, GPU 0. Global residual interval internal coverage 89.61%, width 3.63/10; local 89.88%, width 3.71. No external coverage or original-v1 guarantee. Public runtime cannot use these intervals for original v1; probabilities remain null.

Privacy-preserving archive removes individual nearest-neighbor embeddings and calibration ranks from exported risk bundles. These **aggregate-only derivatives are not byte-identical, fully resumable R4 experiments**. Original checkpoints remain preserved locally/cloud-side. The review runtime can perform guardrails, not reconstruct removed numerical risk calculations.

## R5 — `round5-service-v1`

29 original CPU service tests passed; real loopback HTTP run on one logical CPU, BLAS/OMP threads=1. Cold start 1.305 s; serial100 P95 1.997 ms; optional-review30 P95 2.252 ms; concurrency4/200 success P95 8.542 ms; concurrency8/200 success P95 14.810 ms; concurrency16 99 success/101 rejected (429), success P95 26.305 ms. 500 repeated requests identical scores, RSS delta zero; sampled RSS peak 241229824 bytes. These are distinct from the new packaging tests, whose counts are recorded in `results/engineering-verification.json` after execution.

Address-space caps 128/256 MiB prevented readiness, 512/1024 MiB allowed loading. RLIMIT_AS is not measured physical RAM or full-container HTTP proof. No existing server's spare resources were audited. A host with at least 1 GiB RAM and ~512 MiB headroom per worker is an engineering starting point, not certified production capacity. No production changes or online model calls.
