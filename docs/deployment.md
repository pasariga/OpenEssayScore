# CPU deployment and version management

Source and artifact lifecycle are separate. Python3.11 is the verified environment; pin NumPy1.26.4, SciPy1.12.0, scikit-learn1.8.0 and joblib1.4.2 for old serialization. Install from source as shown in README; HTTP extras are optional. The Python/CLI core does not import FastAPI, Torch or any remote provider.

An authorized bundle must contain `best_scorer.joblib` and `lexical_vectorizer.joblib` with model-card hashes. `init-v1` verifies before deserialization and creates:

```text
model-store/
  registry.json
  versions/<immutable-model-id>/manifest.json
  versions/<immutable-model-id>/best_scorer.joblib
  versions/<immutable-model-id>/lexical_vectorizer.joblib
```

Runtime validates manifest/file digests and contract. Existing version bytes are never replaced. The active pointer changes only after successful loading; in-flight requests keep their loaded version, recorded in results. Loader cache holds at most two versions. Use `python -m openessayscore.service.manage activate --store ... --version ...` for an explicitly registered compatible release. Current loader deliberately supports the frozen character-v1 family/hashes; a genuinely new model needs reviewed registration/loader support, not just a renamed folder. Rollback tests use a clearly labelled identical-weight engineering copy, not a fictitious v2.

Prototype resource evidence: cold start~1.30s, peak RSS~230MiB and serialP95~2ms on one logical cloud CPU. Start engineering trials with ≥1vCPU, ≥1GiB host RAM and about512MiB available per worker, then measure on the actual host. BLAS thread env vars `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` should be set before process startup. Each process loads its own model; multiple workers increase memory. 16-way load was throttled; do not bypass admission limits by unbounded queues.

Server binds loopback, one worker; no automatic daemon, container, production deployment or restart is included. For controlled engineering trials put your existing authenticated reverse proxy in front, cap request size/rate/concurrency and protect journals. This task has not verified a public auth/TLS setup, real student-data governance, or any production host's spare RAM. Do not use the model for unattended high-stakes decisions.

Optional review: supply `--review-bundle <authorized-review-runtime>`. The sanitized archive supports guardrails only. Automatic error routing and original-v1 intervals remain unavailable. API still works if the optional module fails, with an explicit unavailable reason and manual review required.

Shadow testing requires rights-approved real data, independent sealed evaluation, supported version provenance, load/failure isolation on the target host, manual review policy, rollback drill, privacy review, and explicit operational authorization. No online deployment has been performed here.
