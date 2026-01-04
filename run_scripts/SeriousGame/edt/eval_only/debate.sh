#!/usr/bin/env bash
# Run EDT (Enterprise Digital Twin) serious game evaluation with Debate agent.
#
# NOTE: EDT evaluation is scenario-level and requires a *BPTK template repo root*
# that contains BOTH:
#   - scenarios/interactive.json
#   - simulation_models/ (Python package used by the scenario)
#
# If you cloned the official tutorial repo, a typical value is:
#   <bptk_py_tutorial-master>/model_library/enterprise_digital_twin

set -e

BENCHMARK_MODULE="SeriousGame.run_edt"   # 对应 python -m SeriousGame.run_edt
BENCHMARK_NAME="SeriousGame"
TASK_NAME="EDT"
AGENT_METHOD="debate"
MODE="eval_only"
SAVE_DIR="results"

LOG_NAME="${BENCHMARK_NAME}_run_${TASK_NAME}_${AGENT_METHOD}_${MODE}.log"
echo "Starting ${TASK_NAME} ${MODE} ${AGENT_METHOD} run. Logs: ${LOG_NAME}"

python -m "${BENCHMARK_MODULE}" \
  --mode "${MODE}" \
  --agent_method "${AGENT_METHOD}" \
  --api_provider openai \
  --generator_model deepseek-v3 \
  --max_tokens 4096 \
  --save_dir "${SAVE_DIR}" \
  --bptk_script SeriousGame/run_bptk_server.py \
  --bptk_host 127.0.0.1 \
  --edt_mcp_server_path SeriousGame/edt_mcp_server_local.py \
  --bptk_repo_root ./SeriousGame \
  --episodes 6 \
  "$@" \
  > "${LOG_NAME}" 2>&1 &
