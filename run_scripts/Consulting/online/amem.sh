#!/usr/bin/env bash
set -euo pipefail

#############################################
# AMemAgent on Consulting dataset (online)  #
#############################################

# 项目根目录（相对当前脚本位置向上三级）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# ===== 可按需修改的参数 =====

# 咨询 Case 数据路径（相对于项目根）
DATA_PATH="Consulting/agsm_cases_all.json"

# 结果保存根目录
SAVE_DIR="results"

# API provider & 模型
API_PROVIDER="usd_guiji"
GENERATOR_MODEL="USD-guiji/deepseek-v3"   # 或者你的 GPT-4.1/DeepSeek 模型名

# 评测模式 & 面试轮数
MODE="online"                    # 与 bizbench 保持一致：online / eval_only
MAX_TURNS=12

# 是否启用 LLM Judge 打分（加上该变量则开启）
ENABLE_JUDGE=true

# ============================

JUDGE_FLAG=""
if [ "${ENABLE_JUDGE}" = "true" ]; then
  JUDGE_FLAG="--consulting_judge"
fi

echo "==============================================="
echo " Running AMemAgent on Consulting dataset"
echo "-----------------------------------------------"
echo " PROJECT_ROOT     = ${PROJECT_ROOT}"
echo " DATA_PATH        = ${DATA_PATH}"
echo " SAVE_DIR         = ${SAVE_DIR}"
echo " API_PROVIDER     = ${API_PROVIDER}"
echo " GENERATOR_MODEL  = ${GENERATOR_MODEL}"
echo " MODE             = ${MODE}"
echo " MAX_TURNS        = ${MAX_TURNS}"
echo " ENABLE_JUDGE     = ${ENABLE_JUDGE}"
echo "==============================================="
echo
LOG_FILE="consulting_amem_$(date +%Y%m%d_%H%M%S).log"

nohup python -u -m Consulting.run \
  --data_path "${DATA_PATH}" \
  --api_provider "${API_PROVIDER}" \
  --generator_model "${GENERATOR_MODEL}" \
  --save_dir "${SAVE_DIR}" \
  --mode "${MODE}" \
  --max_turns "${MAX_TURNS}" \
  ${JUDGE_FLAG} \
  >"${LOG_FILE}" 2>&1 &

echo "Started in background with nohup. Log: ${LOG_FILE}"
