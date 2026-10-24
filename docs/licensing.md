# Licensing, data provenance and publication review

Review date: 2026-10-22. This is an evidence record, not a substitute for permission from a rights holder.

## Original source

This repository is an extraction of the project's own research/serving code, adapted into the independent `openessayscore` package. It includes no production website code, database, credentials, vendored dependency implementation, or pretrained backbone weights. Generic interface schemas and original transport tests are included; branded provider calls have been replaced by caller-supplied stubs. The user authorized open sourcing this subproject. Original source and documentation use MIT. Historical research names remain to preserve provenance; no platform is a required component.

## CFMS-34

- Authors/publisher copyright: Shihang Yang, Sanwoo Lee, Ningning Zhao, Yunfang Wu.
- Pinned source: [CFMS-34 README](https://github.com/Atiyahsama/HiFTS/blob/933489269eb9dbf5f6f00f8e8be6ba0ccebc3dbb/CFMS-34/README.md), [publisher LICENSE](https://github.com/Atiyahsama/HiFTS/blob/933489269eb9dbf5f6f00f8e8be6ba0ccebc3dbb/LICENSE).
- Publisher declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), research use, and human oversight for high-stakes decisions. We do not turn that guidance into an invented CC noncommercial licence or imply all educational uses are cleared.
- Changes used in the historical study: normalization/contact redaction, dual-rater total summation, audited official splits, quarantine of invalid auxiliary labels for later experiments. Intermediate generated feedback was excluded from human labels.
- No source essays or individual scores are redistributed. We have not independently inspected student/guardian consent or all underlying privacy/personality rights; a copyright licence does not clear those rights.

## Model files

Original v1 coefficients/vectorizer have unchanged fingerprints in the research export records. The source checkout does not include these binaries or grant a public model license. Visibility check, 2026-10-24: `pasariga/OpenEssayScore-Archive` is public and has existing Release assets; older references to a private archive do not establish access control for that GitHub repository. This does not assert a categorical legal ban on derived weights: public redistribution and intended-use boundaries remain to be clarified before a separate model release. The source MIT licence does not cover dataset rights or implicitly relicense trained artifacts. A vocabulary may retain corpus fragments and is treated as sensitive; private storage is not permission for broader disclosure.

Research exports omit per-essay nearest-neighbor matrices, raw/processed records, predictions, split IDs and calibration ranks. They retain code, aggregate reports, learned coefficients/scalers and permitted model state. Originals remain in the existing controlled research backups; this GitHub archive is a **sanitized reproducibility/serving backup, not a complete raw-data mirror**.

## Pretrained comparisons and dependencies

[BAAI/bge-small-zh-v1.5 model card](https://huggingface.co/BAAI/bge-small-zh-v1.5) declares MIT; historical revision `7999e1d3359715c523056ef9478215996d62a620`. Backbone files and per-essay embeddings are not uploaded. BGE is unnecessary for selected-v1 inference. Semantic reproduction requires separately obtaining the upstream revision and complying with its terms.

Dependency versions/licence families were inspected from the actual research environment; see [notices](../THIRD_PARTY_NOTICES.md). Dependencies are installed from their official distributions, not vendored/relicensed here. Binary wheels may include BLAS, Fortran runtimes and additional notices: retain their own licences when distributing environments. MIT on our source does not replace those terms.

## Publication checklist outcome

Approved: original code/docs, generic schemas, header-only annotation templates, aggregate metrics, historical model hashes. Historically reviewed research export (not a statement of current GitHub access control): v1 and selected research coefficients/scalers, aggregate-only R4 bundles. Excluded: student essays, per-essay labels/predictions/IDs/embeddings, original mixed archives, external backbone weights, private infrastructure, credentials. Public model permission: pending; new teacher data/independent accuracy validation: pending. Automated scanning supplements this manual selection and cannot certify all personal data or IP rights.
