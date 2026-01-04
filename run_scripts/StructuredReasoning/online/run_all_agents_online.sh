#!/usr/bin/env bash
set -euo pipefail

# Run all StructuredReasoning online scripts for all agent methods (excluding capability_eval).
# NOTE: Most per-task scripts are nohup'ed and run in background; this launcher will return quickly.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

ONLINE_DIR="run_scripts/StructuredReasoning/online"

echo "Repo root : ${ROOT_DIR}"
echo "Online dir: ${ONLINE_DIR}"
echo "Launching all agent online scripts (excluding capability_eval) ..."
echo

mapfile -t SCRIPTS < <(
  find "${ONLINE_DIR}" -type f -name "*.sh" \
    -not -path "*/capability_eval/*" \
    | LC_ALL=C sort
)

if [[ "${#SCRIPTS[@]}" -eq 0 ]]; then
  echo "ERROR: No online scripts found under ${ONLINE_DIR}." >&2
  exit 1
fi

for sh in "${SCRIPTS[@]}"; do
  echo "[RUN] bash ${sh} $*"
  bash "${sh}" "$@"
done

echo
echo "Done. Check logs in current directory and results under results/StructuredReasoning_run/."


