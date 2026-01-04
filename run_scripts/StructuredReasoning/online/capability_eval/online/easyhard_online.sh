#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_MODULE="StructuredReasoning.run"
BENCHMARK_NAME="StructuredReasoning"
DOMAINS=("Finance_Reasoning" "Span_extraction" "Knowledge_understand")  # 下划线形式
MODE="online"
CONFIG_PATH="StructuredReasoning/data/task_config.json"
SAVE_PATH="results/${BENCHMARK_NAME}_run"
# 如需调整方法列表，编辑下行
AGENTS=("ace" "cot" "dynamic_cheatsheet" "self_refine" "reflexion" "gepa")

echo "Converting domains ${DOMAINS[*]} ${MODE} for agents: ${AGENTS[*]}"

for DOMAIN in "${DOMAINS[@]}"; do
  for AGENT_METHOD in "${AGENTS[@]}"; do
    LOG_NAME="${BENCHMARK_NAME}_run_${DOMAIN}_${AGENT_METHOD}_${MODE}_easyhard.log"
    echo "Start domain=${DOMAIN}, agent=${AGENT_METHOD}, log: ${LOG_NAME}"
    nohup python -m "${BENCHMARK_MODULE}" \
      --agent_method "${AGENT_METHOD}" \
      --mode "${MODE}" \
      --config_path "${CONFIG_PATH}" \
      --save_path "${SAVE_PATH}" \
      --run_mode "easyhard" \
      --domain "${DOMAIN}" \
      "$@" \
      > "${LOG_NAME}" 2>&1 &
  done
done

