#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root relative to this script (more portable than hard-coding an absolute path).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-logs/capability_eval}"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/capability_eval_online_${TS}.log"
PID_FILE="${LOG_DIR}/capability_eval_online_${TS}.pid"

# Configurable paths (override via env vars if needed).
RESULTS_ROOT="${RESULTS_ROOT:-results/StructuredReasoning_run}"
OUT_DIR="${OUT_DIR:-results/StructuredReasoning_run/capability_eval_mode}"

# Auto-pick a usable classify_root if not specified.
# We select the newest directory under results/StructuredReasoning_run/llm_reclassify_mode/ that has test/classifications.jsonl.
CLASSIFY_ROOT="${CLASSIFY_ROOT:-}"
if [[ -z "${CLASSIFY_ROOT}" ]]; then
  for d in $(ls -1dt results/StructuredReasoning_run/llm_reclassify_mode/*/ 2>/dev/null || true); do
    d="${d%/}"
    if [[ -f "${d}/test/classifications.jsonl" ]]; then
      CLASSIFY_ROOT="${d}"
      break
    fi
  done
fi
if [[ -z "${CLASSIFY_ROOT}" ]]; then
  echo "ERROR: cannot infer CLASSIFY_ROOT. Please export CLASSIFY_ROOT=<.../llm_reclassify_mode/<name>>" >&2
  exit 1
fi

# Behavior toggles (override via env vars).
ONLY_MODE="${ONLY_MODE:-online}"
ONLY_AGENT="${ONLY_AGENT:-}"
EXPORT_CSV="${EXPORT_CSV:-1}"  # 1: export CSV, 0: no CSV
CSV_AGENTS="${CSV_AGENTS:-cot,self-refine,reflexion,debate,discussion,dc,gepa,ace,amem}"

CMD=(python3 -u utils/capability_eval.py
  --results_root "${RESULTS_ROOT}"
  --classify_root "${CLASSIFY_ROOT}"
  --out_dir "${OUT_DIR}"
  --only_mode "${ONLY_MODE}"
)
if [[ -n "${ONLY_AGENT}" ]]; then
  CMD+=(--only_agent "${ONLY_AGENT}")
fi
if [[ "${EXPORT_CSV}" == "1" ]]; then
  CMD+=(--export_csv --csv_agents "${CSV_AGENTS}")
fi

echo "Repo root     : ${ROOT_DIR}"
echo "Results root  : ${RESULTS_ROOT}"
echo "Classify root : ${CLASSIFY_ROOT}"
echo "Out dir       : ${OUT_DIR}"
echo "Only mode     : ${ONLY_MODE}"
echo "Only agent    : ${ONLY_AGENT:-<all>}"
echo "Export CSV    : ${EXPORT_CSV}"
echo "Log file      : ${LOG_FILE}"

nohup "${CMD[@]}" "$@" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started capability_eval online (pid=$(cat ${PID_FILE})). ${LOG_FILE}"


