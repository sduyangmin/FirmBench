#!/usr/bin/env bash
set -euo pipefail

# 对齐现有脚本的变量命名方式
BENCHMARK_MODULE="SeriousGame.run_beergame"   # 对应 python -m SeriousGame.run_beergame
BENCHMARK_NAME="SeriousGame"
TASK_NAME="BeerGame"
AGENT_METHOD="self_refine"
MODE="eval_only"                              # 对应 run_beergame.py 的 --mode

# LLM / Agent 相关
API_PROVIDER="usd_guiji"                      # openai / sambanova / together / usd_guiji
GENERATOR_MODEL="USD-guiji/deepseek-v3"       # 对应 --generator_model
MAX_TOKENS=512

# 结果保存
SAVE_DIR="results/${BENCHMARK_NAME}_run"
LOG_NAME="${BENCHMARK_NAME}_run_${TASK_NAME}_${AGENT_METHOD}_${MODE}.log"

# Beer Game MCP server 配置
SERVER_PATH="SeriousGame/beergame_mcp_server.py"   # 相对项目根目录
MCP_TIMEOUT_SEC=60

echo "Starting ${TASK_NAME} ${MODE} ${AGENT_METHOD} run. Logs: ${LOG_NAME}"

nohup python -u -m "${BENCHMARK_MODULE}" \
  --mode "${MODE}" \
  --agent_method "${AGENT_METHOD}" \
  --api_provider "${API_PROVIDER}" \
  --generator_model "${GENERATOR_MODEL}" \
  --max_tokens "${MAX_TOKENS}" \
  --save_dir "${SAVE_DIR}" \
  --server_path "${SERVER_PATH}" \
  --mcp_timeout_sec "${MCP_TIMEOUT_SEC}" \
  "$@" \
  > "${LOG_NAME}" 2>&1 &


