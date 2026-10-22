# Independent CPU deployment: what is actually verified

This toolkit is independent of any website, production database, provider API or cloud vendor. The historical model remains WenHeng-Score v1, native CFMS0–10. Source MIT does not grant rights to the private model or student data; see [licensing](licensing.md).

## Evidence layers

| Layer | Evidence | Boundary |
| --- | --- | --- |
| Historical CPU HTTP prototype | R5: cold~1.30s, serialP95~2ms, 4-wayP95~8.54ms, sampledRSS~230MiB;29 tests | One cloud logical CPU, short tests, no cgroup cap |
| Packaging/recovery | Fresh GitHub clone+private Release download;43 tests, wheel install, Python/CLI/HTTP parity | Authorized private artifact required; no new accuracy evidence |
| Protected local transport | R6:29 new tests; UDS-only, sampledRSS~216.5MiB, cold~1.1s,50 serial requests | No Docker daemon, no512MiB hard cap, no0.25CPU quota enforcement |
| Docker512MiB | Reusable harness provided; **not executed yet** | Current available nonproduction environments cannot run a suitable Docker daemon |

R5/R6 figures are observations, not a universal hardware minimum or production SLA. RSS excludes some container accounting; virtual-address limits are not physical RAM limits. Short observations cannot establish peak capacity or multi-day stability. No model retraining, risk-routing promotion or production deployment accompanied this documentation.

## Practical sizing

Use a separate CPU process/container, one worker, BLAS/OMP1 thread. The two inference artifacts total about1.4MiB, but runtime dependencies and intermediate allocations dominate memory. Inspected dependency files total about275MiB on disk before Python/OS/image layers. Keep space for immutable image/model versions and rollback; do not install into a website's environment. Python3.11 is verified; a host with Python3.10 should use an isolated compatible environment instead of upgrading the application's interpreter.

Plan an initial512MiB container budget and additional host headroom, then test it. This is a **candidate**, not a passed limit. A1GiB host that runs only the scorer differs materially from a host sharing RAM with web, database, OCR and monitoring processes. Limits do not reserve RAM. Count shared-service growth as well as present usage.

An anonymized resource-audit example illustrates the problem: about897MiB available at low load, an existing backend using127MiB with a600MiB limit, a proposed512MiB scorer and256MiB safety margin. Required available RAM is `max(600-127,0)+512+256 ≈1241MiB`, leaving a deficit of about344MiB. This is a conservative concurrent-limit screen, not an exact forecast. CPU/disk may be adequate while RAM peak headroom fails. Do not infer deployment approval from low average CPU or current free RAM alone. No host identity, connection configuration or raw inventory is published here.

## Reusable shadow preparation

`deployment/shadow/` is extracted from the already tested R6 prototype, not a second scoring implementation. It contains:

- Default-disabled, purpose/expiry/withdrawal/guardian/deidentification gates; empty authorization template.
- One dispatcher thread, at most2 unfinished jobs, bounded sampling and hourly limits; timeout/no retry and in-process circuit breaker.
- Protected Unix socket, no TCP, optional no-network container; separate result journal with field allowlist and byte cap.
- Native0–10 output, aggregation weight0, manual review required, automatic risk routing off.
- Independent Compose project and default-plan activation/rollback scripts. Rollback works without the original image configuration and stops only labeled shadow containers.

Run the adapter tests with `python -m pytest -q deployment/shadow/tests`. Artifact-dependent tests explicitly skip without a separately authorized bundle. The actual engine/proxy/limits still need a nonproduction Docker test. No website hook, database exporter or live traffic tap is installed. Authorization booleans must come from a trusted review process, not student requests; automated redaction does not establish anonymity.

## 512MiB hard-limit protocol

Use an existing **nonproduction Linux Docker host** and permitted v1 artifacts. Do not use a production host, privileged nested container, remount shared cgroups, install a daemon into a managed notebook by force, or create a paid instance to make the test pass. Rootless Docker also requires working namespaces/delegated cgroups; see [rootless prerequisites](https://docs.docker.com/engine/security/rootless/).

1. Obtain a reviewed immutable Python3.11 base digest; build the image from the public source. The `.dockerignore` is an allowlist, excluding datasets, credentials and model files from build context. Exact runtime dependencies are preserved; our local source is installed before dependencies, not replaced by a same-named index package.
2. Prepare a private registered model store with `python -m openessayscore init-v1 ...`. Keep original files immutable and do not put them into the image or public CI artifacts.
3. Run the bounded verification tool. It uses no network/GPU, drops capabilities, sets `--memory=512m --memory-swap=512m`, pins oneCPU, and requires both Docker configuration and in-container kernel limit readings to match. Equal memory/swap values disable swap; see [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/).
4. Measure3 cold starts,100 serial requests,1000 repeated requests,4/8-way concurrency, a deliberately touched-page OOM workload inside the verified cap, explicit recreation/recovery and a128MiB negative loading control. Report rejection counts separately; concurrent overload need not have100% success. An exec process can be OOM-killed without `.State.OOMKilled=true` on the main container, so inspect cgroup OOM counters too.
5. Store a new timestamped result; never overwrite historical results. If kernel limit evidence is missing, the run is unsupported/failed, not “512MiB passed”. No simulated metrics or synthetic teacher labels are accepted. Archive code/aggregate evidence and verify recovery before any stable release.

```bash
# Run only after the environment and artifact rights have been confirmed.
docker build -f deployment/cpu/Dockerfile \
  --build-arg PYTHON_BASE_IMAGE='python:3.11-slim-bookworm@sha256:REVIEWED_DIGEST' \
  -t openessayscore:cpu-test .
# Read the resulting local immutable image ID, then pass that sha256 value below.
python tools/docker_hardlimit.py --image sha256:REVIEWED_LOCAL_IMAGE_ID \
  --store /your/authorized-model-store --output /your/results/cpu512-new-run.json \
  --nonproduction-confirmed
```

The tool removes only test containers returned by its own docker-run calls and its new temporary export; it does not prune images or delete source stores/backups. It starts at most one test workload container at a time. Image construction can download open-source dependencies; use a reviewed mirror/offline cache if required. Do not publish the private model, kernel/host identifiers, raw Docker diagnostic inventory or source text when sharing results.

## Current test-environment limitation

Local inspection found no Docker/Podman executable or installed WSL environment. The available managed GPU workspace has no Docker/Podman, no effective SYS_ADMIN capability, and its actual memory/CPU cgroup controllers are mounted read-only. A writable parent tmpfs is not a writable memory controller. No system features, cgroups or unrelated VPS settings were changed. Consequently **real Docker512MiB results remain pending**. The pending status is separate from the already verified CPU interfaces and backup recovery.

Production shadow trials additionally need reviewed data/model purposes, peak capacity, actual hard-limit/OOM/rollback evidence, access control, limited retention and explicit operational approval. User-facing scoring also needs new independent teacher-scored evaluation; existing internal risk results do not justify automated triage.
