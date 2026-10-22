# Reproduction without hidden data or API dependencies

No dataset is bundled. Obtain CFMS-34 from its [pinned publisher revision](https://github.com/Atiyahsama/HiFTS/tree/933489269eb9dbf5f6f00f8e8be6ba0ccebc3dbb/CFMS-34) only if you can comply with its licence, intended use and applicable data rights. Do not upload it in issues, forks, CI artifacts or backup payloads. `feedback.csv` is not a human-label source. This repository never auto-downloads student records.

Historical scripts under `research/` retain algorithms and fixed run IDs. Set `WENHENG_WORKSPACE` to an isolated authorized workspace; this legacy environment-variable name is kept for research compatibility and is not a platform dependency. Scripts create raw/processed/checkpoints/reports beneath that root. Copy the publisher `full/train.csv`, `dev.csv`, `test.csv` and `trait_map.csv` into `data/raw/cfms34/<revision>/`; do not merge full/core (same essays).

```bash
export WENHENG_WORKSPACE="$PWD/workspace"
python research/wenheng_score/prepare_data.py
# CPU lexical-only reproduction using the fixed v1 recipe:
python research/train_lexical.py --workspace "$WENHENG_WORKSPACE" --run-id my-lexical-reproduction
```

The CPU command trains the fixed char1–3/min_df2/max40000/sublinear_tf Ridge0.1 recipe on train, reports dev, and **does not read test**. Its artifacts are locally generated research models, not the published historical weights or automatically stable v2. Different versions/hardware/serialization can change bytes; historical v1's hash-pinned loader does not silently accept new models. New artifacts require separate manifest/loader integration and independent validation.

Full historical comparison (`research/wenheng_score/train_round1.py --run-id <new-id>`) additionally requires PyTorch/Transformers, the separately acquired pinned BGE-small-zh-v1.5 revision, and an authorized GPU budget. It evaluates the historical test after dev selection; replaying it does not create a new unseen test. Do not run it merely to install the CPU tool. No GPU experiment was rerun for this release.

R2/R4 reproduction needs authorized processed **training-only** records plus the original separately retained embedding cache and exact environment; optional torch/transformers research dependencies are not part of runtime installation. `research/round2/reproduce_round2.py --help` and `research/round4/reproduce.py --help` expose run controls. R2 audit helpers expecting original manifests/reports are historical workspace tools, not standalone commands that magically restore missing records. The sanitized GitHub archive intentionally excludes raw data, per-essay features and full R4 references, so it can restore serving and learned state but cannot independently re-run every historical result.

For future research: train-only nested/group CV for development; keep calibration separate; seal a newly sourced teacher-scored test before experiments; preregister metrics, comparisons and stopping rules; report MAE/RMSE/QWK and score-bin error, risk AUROC/precision/recall vs review budget, interval coverage/width and CPU cost. Do not tune using the exposed old test, validate with generated teacher labels, or use unknown auxiliary label meanings.
