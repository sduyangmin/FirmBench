#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_MODULE="StructuredReasoning.run"
BENCHMARK_NAME="StructuredReasoning"
AGENT_METHOD="discussion"
TASK_NAME="SEC-NUM"
MODE="online"
CONFIG_PATH="StructuredReasoning/data/task_config.json"
SAVE_PATH="results/${BENCHMARK_NAME}_run"
LOG_NAME="${BENCHMARK_NAME}_run_${TASK_NAME}_${AGENT_METHOD}_${MODE}.log"

echo "Starting ${TASK_NAME} ${MODE} ${AGENT_METHOD} run. Logs: ${LOG_NAME}"

nohup python -m "${BENCHMARK_MODULE}" \
  --agent_method "${AGENT_METHOD}" \
  --task_name "${TASK_NAME}" \
  --mode "${MODE}" \
  --config_path "${CONFIG_PATH}" \
  --save_path "${SAVE_PATH}" \
  "$@" \
  > "${LOG_NAME}" 2>&1 &


