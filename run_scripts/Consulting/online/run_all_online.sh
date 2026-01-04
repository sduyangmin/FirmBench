#!/usr/bin/env bash
set -euo pipefail

# Run all Consulting online scripts in this directory.
# NOTE: Scripts typically use nohup and run in background.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

ONLINE_DIR="run_scripts/Consulting/online"

echo "Repo root : ${ROOT_DIR}"
echo "Online dir: ${ONLINE_DIR}"
echo "Launching all Consulting online scripts ..."
echo

mapfile -t SCRIPTS < <(
  find "${ONLINE_DIR}" -maxdepth 1 -type f -name "*.sh" \
    -not -name "run_all_online.sh" \
    | LC_ALL=C sort
)

if [[ "${#SCRIPTS[@]}" -eq 0 ]]; then
  echo "ERROR: No scripts found under ${ONLINE_DIR}." >&2
  exit 1
fi

for sh in "${SCRIPTS[@]}"; do
  echo "[RUN] bash ${sh} $*"
  bash "${sh}" "$@"
done

echo
echo "Done. Check logs in repo root and results under results/Consulting/."


