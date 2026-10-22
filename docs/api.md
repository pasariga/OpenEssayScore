# Python, CLI and HTTP contract

Public package: `openessayscore`; contract `openessayscore/1.0`. Historical model ID remains `wenheng-score-v1-20261022`. Package version, contract version and model version are different identities.

`Scorer(model_store)` loads an operator-managed registry. `Scorer.from_v1_bundle(model_dir, model_store=...)` explicitly copies hash-verified v1 artifacts once. `score(text, writing_prompt=..., authorization_scope='authorized_research', text_verified=True, model_version=None, include_review=False)` returns the same structured result used by CLI/HTTP. For OCR text, callers must actually verify transcription before asserting `text_verified`. Declaring authorization is an assertion, not a rights-checking service.

CLI: `init-v1`, `status`, `score`; `python -m openessayscore --help`. `score` reads a UTF-8 file with a 128KiB file cap and returns JSON. Failures exit 2 with a redacted error code. `status` can return `ready:false` while exit 0 means status was retrieved, not scoring readiness.

HTTP defaults to `127.0.0.1`; `GET /health/live`, `GET /health/ready` (503 if unavailable), `GET /v1/models`, `POST /v1/score`. Strict JSON request fields:

| Field | Meaning |
| --- | --- |
| text | Required string, ≤20000 characters, ≥20 Chinese characters after normalization |
| writing_prompt | Required nonempty context, ≤1000 characters; does not add prompt-relevance scoring |
| authorization_scope | `authorized_research` or `public_research_dataset` |
| text_verified | Strict boolean; must be true |
| model_version | Optional registered immutable version, defaults to active |
| rubric_id | Optional; only `CFMS34-holistic-sum-0to10` supported |
| grade_level | Optional integer1–6 context; does not prove grade validity |
| include_review | Optional boolean false; guardrails only |
| request_id | Optional correlation ID; repeated ID is not idempotency |
| deadline_ms | Optional10…10000, default5000; checked after inference, not hard cancellation |

Response: `prediction.overall_score`, `prediction.dimension_scores=null`, `prediction.confidence=null`; model version and SHA-256; normalized-input hash; review reasons. Input hash is a potentially linkable derivative: protect logs. Optional local journals store results, not raw text; journals are not committed automatically. Default teacher review is required. No probability or interval is inferred for historical v1.

Errors use `error.code` and retryable indicator. Common statuses:400 malformed JSON,413 size,415 content type,422 invalid/insufficient/unverified input,409 wrong rubric,404 unknown version,429 busy,503 missing/corrupt model or resource exhaustion,504 deadline. Raw submitted text is not echoed in diagnostics.

`openessayscore.service.collaboration` is an **offline generic example** with supplied `provider_a/b/c` callables and an illustrative four-trait/100-point rubric. It makes zero provider calls itself. No native10→100 mapping is validated; `map_native_score` raises `MappingUnavailable`, native aggregation weight stays0, existing final result is preserved. Client/HTTP timeout is not a provider fallback assessment. Integrators must validate their own rubric and mappings before aggregation.
