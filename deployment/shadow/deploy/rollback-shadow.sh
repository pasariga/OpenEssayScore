#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'Plan: disable the independent producer first, then stop ONLY oes-shadow-r6 score; preserve models/results.'
if [[ "${1:-}" != '--execute-after-approval' || "${OES_CHANGE_APPROVED:-}" != yes ]]; then exit 0; fi
# Rollback must still work if the original image/mount environment is unavailable.
listed="$(docker ps --filter label=com.docker.compose.project=oes-shadow-r6 --filter label=com.docker.compose.service=score --format '{{.ID}}')"
ids=()
if [[ -n "$listed" ]]; then mapfile -t ids <<< "$listed"; fi
for id in "${ids[@]}"; do
  if [[ ! "$id" =~ ^[0-9a-f]{12,64}$ ]]; then echo 'Unexpected container ID; refusing stop' >&2; exit 4; fi
done
if (( ${#ids[@]} )); then docker stop -t 5 "${ids[@]}"; fi
# Deliberately no compose down, volume deletion, DB operation or production restart.
