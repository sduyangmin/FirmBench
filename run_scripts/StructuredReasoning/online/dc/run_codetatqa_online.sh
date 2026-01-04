#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_MODULE="StructuredReasoning.run"
BENCHMARK_NAME="StructuredReasoning"
AGENT_METHOD="dynamic_cheatsheet"
TASK_NAME="CodeTAT-QA"
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
  --api_provider usd_guiji \
  --generator_model USD-guiji/deepseek-v3 \
  --dc_disable_code_execution \
  --max_tokens 4096 \
  --dc_approach DynamicCheatsheet_RetrievalSynthesis \
  --dc_generator_prompt_path Agents/dynamic_cheatsheet/prompts/generator_prompt.txt \
  --dc_cheatsheet_prompt_path Agents/dynamic_cheatsheet/prompts/cheatsheet_cumulative.txt \
  "$@" \
  > "${LOG_NAME}" 2>&1 &


