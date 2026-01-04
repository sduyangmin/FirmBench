#!/usr/bin/env python3
"""
离线生成输入语料的 embedding CSV，用于 DynamicCheatsheet 检索模式预处理。

输入：JSONL，每行包含 question 字段，可选 context 字段。
输出：CSV，列为 input, tokens, embedding_json（embedding 序列的 JSON 字符串）。

用法示例（OpenAI 后端）：
python tools/generate_embeddings_csv.py \
  --backend openai \
  --input_jsonl StructuredReasoning/data/finer_test.jsonl \
  --output_csv outputs/finer_embeddings.csv \
  --model text-embedding-3-large \
  --batch_size 32

用法示例（本地/自托管 HF 模型，推荐轻量 BAAI/bge-small-en-v1.5）：
python tools/generate_embeddings_csv.py \
  --backend hf \
  --hf_model BAAI/bge-small-en-v1.5 \
  --input_jsonl StructuredReasoning/data/finer_test.jsonl \
  --output_csv outputs/finer_embeddings.csv \
  --batch_size 32
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict

import tiktoken
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

# 可选：本地/自托管 HuggingFace embedding 模型（sentence-transformers）
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def read_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_input_text(item: Dict, question_field: str, context_field: str) -> str:
    question = item.get(question_field, "") or ""
    context = item.get(context_field, "") or ""
    if context:
        return f"{question}\n\nContext:\n{context}"
    return question


def count_tokens(text: str, enc) -> int:
    try:
        return len(enc.encode(text))
    except Exception:
        return 0


def generate_embeddings_openai(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [e.embedding for e in resp.data]


def generate_embeddings_hf(model: "SentenceTransformer", texts: List[str], batch_size: int) -> List[List[float]]:
    # SentenceTransformer 会自行分批；convert_to_numpy=True 更快
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return embs.tolist()


@torch.no_grad()
def generate_embeddings_simcse(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[List[float]]:
    model.eval()
    vecs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, return_dict=True)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            e = out.pooler_output
        else:
            e = out.last_hidden_state[:, 0, :]  # CLS
        e = torch.nn.functional.normalize(e, p=2, dim=1)
        vecs.append(e.cpu())
        print(f"Processed {min(start + len(batch), len(texts))}/{len(texts)} (simcse)", end="\r")
    return torch.cat(vecs, dim=0).tolist()


def main():
    parser = argparse.ArgumentParser(description="Generate embedding CSV for DynamicCheatsheet retrieval modes.")
    parser.add_argument("--backend", choices=["openai", "hf", "simcse"], default="openai", help="选择 embedding 后端：openai/hf/simcse。")
    parser.add_argument("--input_jsonl", required=True, help="输入 JSONL 路径，含 question/context 字段。")
    parser.add_argument("--output_csv", required=True, help="输出 CSV 路径。")
    parser.add_argument("--model", default="text-embedding-3-large", help="OpenAI embedding 模型名称。")
    parser.add_argument("--hf_model", default="BAAI/bge-small-en-v1.5", help="HuggingFace embedding 模型名称或本地路径。")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小。")
    parser.add_argument("--sim_model", default="/data0/yangmin/hf_models/sup-simcse-bert-base-uncased", help="SimCSE 本地/远程模型路径。")
    parser.add_argument("--sim_device", default="auto", choices=["auto", "cpu", "cuda"], help="SimCSE 推理设备。")
    parser.add_argument("--sim_max_length", type=int, default=256, help="SimCSE tokenizer 最大长度。")
    parser.add_argument("--question_field", default="question", help="问题字段名。")
    parser.add_argument("--context_field", default="context", help="上下文字段名。")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(input_path)
    texts = [build_input_text(it, args.question_field, args.context_field) for it in items]

    enc = tiktoken.encoding_for_model(args.model) if args.model else tiktoken.get_encoding("cl100k_base")
    token_counts = [count_tokens(t, enc) for t in texts]

    embeddings: List[List[float]] = []
    if args.backend == "openai":
        client = OpenAI()
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            embs = generate_embeddings_openai(client, batch, args.model)
            embeddings.extend(embs)
            print(f"Processed {min(start + len(batch), len(texts))}/{len(texts)} (openai)")
    elif args.backend == "hf":
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers 未安装，请先 pip install sentence-transformers")
        hf_model = SentenceTransformer(args.hf_model)
        embeddings = generate_embeddings_hf(hf_model, texts, args.batch_size)
        print(f"Processed {len(texts)}/{len(texts)} (hf)")
    else:  # simcse backend with transformers
        device = (
            torch.device("cpu")
            if args.sim_device == "cpu"
            else torch.device("cuda" if torch.cuda.is_available() and args.sim_device != "cpu" else "cpu")
        )
        tok = AutoTokenizer.from_pretrained(args.sim_model)
        sim_model = AutoModel.from_pretrained(args.sim_model).to(device)
        embeddings = generate_embeddings_simcse(
            tok, sim_model, texts, device, args.batch_size, args.sim_max_length
        )
        print(f"\nProcessed {len(texts)}/{len(texts)} (simcse on {device})")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "tokens", "embedding_json"])
        for text, tok, emb in zip(texts, token_counts, embeddings):
            writer.writerow([text, tok, json.dumps(emb)])

    print(f"Saved embeddings to: {output_path}")


if __name__ == "__main__":
    main()

