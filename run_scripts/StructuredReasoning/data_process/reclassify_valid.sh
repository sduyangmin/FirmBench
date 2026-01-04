#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

OUT_NAME="capability_difficulty_score_v1"
LOG_DIR="logs/llm_reclassify"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/reclassify_val_${OUT_NAME}.log"
PID_FILE="${LOG_DIR}/reclassify_val_${OUT_NAME}.pid"

nohup python3 utils/llm_reclassify.py \
  --config_path ./StructuredReasoning/data/task_config.json \
  --split val \
  --output_name "${OUT_NAME}" \
  --api_provider usd_guiji \
  --model USD-guiji/deepseek-v3 \
  --resume \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started reclassify val (pid=$(cat ${PID_FILE})). ${LOG_FILE}"




