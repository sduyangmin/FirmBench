#!/usr/bin/env bash
set -euo pipefail

# 批量为 StructuredReasoning/data 下的所有 *test.jsonl 生成 embedding CSV
# 默认使用本地 SimCSE 模型；如需切换可修改 BACKEND/模型参数。

BACKEND="simcse"
SIM_MODEL="/data0/yangmin/hf_models/sup-simcse-bert-base-uncased"
SIM_DEVICE="auto"
SIM_MAX_LENGTH=256
BATCH_SIZE=32

DATA_DIR="StructuredReasoning/data"
OUT_DIR="StructuredReasoning/data/data_embedding"

mkdir -p "${OUT_DIR}"

for file in "${DATA_DIR}"/*test.jsonl; do
  [ -e "$file" ] || continue
  base=$(basename "$file" .jsonl)
  out="${OUT_DIR}/${base}_embeddings.csv"
  echo "Processing ${file} -> ${out}"
  python tools/generate_embeddings_csv.py \
    --backend "${BACKEND}" \
    --sim_model "${SIM_MODEL}" \
    --sim_device "${SIM_DEVICE}" \
    --sim_max_length "${SIM_MAX_LENGTH}" \
    --input_jsonl "${file}" \
    --output_csv "${out}" \
    --batch_size "${BATCH_SIZE}"
done

echo "All embeddings generated into ${OUT_DIR}"




