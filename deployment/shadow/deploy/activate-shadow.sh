#!/usr/bin/env bash
set -euo pipefail
# Default invocation validates configuration and prints the plan. No production changes.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="$HERE/compose.shadow.yaml"
if [[ ! "${OES_SHADOW_IMAGE:-}" =~ ^[a-zA-Z0-9./:_-]+@sha256:[a-f0-9]{64}$ ]]; then echo 'Digest-pinned image required' >&2; exit 2; fi
if [[ "${OES_MODEL_STORE:-}" != /* || "${OES_RESULTS_DIR:-}" != /* || "${OES_CONTROL_DIR:-}" != /* ]]; then echo 'Absolute independent directories required' >&2; exit 2; fi
mounts=()
for dir in "$OES_MODEL_STORE" "$OES_RESULTS_DIR" "$OES_CONTROL_DIR"; do
  resolved="$(realpath -m -- "$dir")"
  if [[ "$resolved" != /srv/oes-shadow/* ]]; then echo 'All mounts must stay below /srv/oes-shadow' >&2; exit 2; fi
  mounts+=("$resolved")
done
for i in 0 1 2; do
  for j in 0 1 2; do
    if [[ "$i" != "$j" && ( "${mounts[$i]}" == "${mounts[$j]}" || "${mounts[$i]}" == "${mounts[$j]}/"* ) ]]; then
      echo 'Model, control and result mounts may not overlap' >&2; exit 2
    fi
  done
done
docker compose --env-file /dev/null --project-directory "$HERE" -p oes-shadow-r6 -f "$COMPOSE" config --quiet
printf '%s\n' 'Plan: start ONLY the oes-shadow-r6 score container; no TCP, no DB, no production compose.'
if [[ "${1:-}" != '--execute-after-approval' ]]; then exit 0; fi
if [[ "${OES_RESOURCE_GATE_PASSED:-}" != yes || "${OES_AUTHORIZATION_GATE_PASSED:-}" != yes || "${OES_CHANGE_APPROVED:-}" != yes ]]; then
  echo 'Resource, data authorization and explicit change approval gates are required' >&2; exit 3
fi
test -d "$OES_MODEL_STORE"; test -d "$OES_RESULTS_DIR"; test -d "$OES_CONTROL_DIR"
docker compose --env-file /dev/null --project-directory "$HERE" -p oes-shadow-r6 -f "$COMPOSE" up -d --no-build --pull never score
