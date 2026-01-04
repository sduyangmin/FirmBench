#!/usr/bin/env python3
"""
LLM-based reclassification for StructuredReasoning samples.

This script uses a user-provided prompt template to ask an LLM to classify each
sample into:
  - Capability: one of 4 categories
  - DifficultyScore: 0-10 (continuous)

It does NOT solve the question. It only labels it.

Outputs:
  - classifications.jsonl: one line per sample with original sample + parsed labels + raw response
  - summary.json: aggregate counts

Supports resume: if classifications.jsonl exists, already-seen (task, split, index) pairs are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

# Ensure project root is on sys.path so `import utils.*` works when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#
# NOTE: We intentionally do NOT import `utils.llm` / `openai` at module import time.
# This allows `--dry_run` to work even if `openai` is not installed in the runtime.
# LLM-related imports are done lazily inside `main()` when needed.


DOMAIN_MAP: Dict[str, List[str]] = {
    "Finance Reasoning": ["FinCode", "CodeFinQA", "CodeTAT-QA", "formula"],
    "Span extraction": ["ConvFinQA", "SEC-NUM", "TAT-QA"],
    "Knowledge understand": ["finer", "FinKnow", "FormulaEval"],
}

ALL_TASKS: List[str] = sorted({t for tasks in DOMAIN_MAP.values() for t in tasks})


PROMPT_TEMPLATE = """You are an expert in question quality review and capability evaluation.

Do NOT solve the question. Your task is to analyze what the question is testing.

TaskName: {task_name}

### Capability Categories
Select ONLY ONE primary capability from the following four categories:

1. Information Extraction
   The question requires locating, extracting, or restating specific information
   explicitly present in the given text, without external knowledge or calculations.

2. Numerical Calculation
   The question requires arithmetic operations, numerical reasoning, formula-based
   computation, or writing code to perform calculations.

3. Domain Knowledge
   The question primarily tests specialized knowledge itself, such as:
   - Finance, accounting, XBRL, or US GAAP concepts and tags
   - Financial formulas or accounting standards
   - CFA exam questions focused on knowledge recall or understanding
   - MMLU questions in business ethics, microeconomics, or professional accounting
     that assess knowledge rather than decision-making

4. Complex Reasoning
   The question requires judgment or decision-making under constraints, such as:
   - Scenario-based CFA exam questions
   - MMLU questions involving business ethics, microeconomics, or professional
     accounting that ask what should be done in a given situation

### Decision Rules (IMPORTANT)
Use these rules to avoid confusion between Information Extraction and Numerical Calculation:

- Information Extraction ONLY IF:
  - The final answer can be directly copied from the provided text/table/verbatim span, AND
  - NO arithmetic/computation is required.

- Numerical Calculation IF ANY computation is needed, even if very simple, including but not limited to:
  - subtraction / difference / net change (e.g., "2015 - 2014")
  - addition / sum / total across years or rows
  - division / ratio / percentage / percent change / growth rate
  - max/min/average over a set of numbers
  - any formula-based computation, or any need to write/execute code

Examples:
- If the context contains two values and the question asks "net change" / "difference" / "by what percentage",
  this is Numerical Calculation (even though the values are extracted from the context).
- If the question asks "What is X in 2017?" and X is explicitly stated as a single value in the table,
  this is Information Extraction.

### Your Task
Analyze the following question and provide:
1. The primary capability category (choose exactly one)
2. A difficulty score from 0 to 10 (0 = trivial, 10 = extremely hard)
3. A brief justification explaining both the capability classification and
   the difficulty level

### Output Format (strictly follow):
Capability: <Information Extraction | Numerical Calculation | Domain Knowledge | Complex Reasoning>
DifficultyScore: <0-10>
Reasoning: <Up to 4 sentences>

### Scoring Guidance (IMPORTANT)
Be critical and use the full 0-10 range as much as possible. Avoid clustering scores in 0-5.
Assign higher scores (6-10) to genuinely challenging questions (long/complex context, multi-step computation, tricky domain knowledge, ambiguity, or decision-making).
Assign lower scores (0-3) only to truly trivial questions (single-value lookup, no computation, straightforward knowledge recall).

### Special Difficulty Rules for Label-Selection / Classification Tasks (IMPORTANT)
Some tasks look short but are hard because the model must choose the correct label from a large, confusing label space
(e.g., US-GAAP XBRL tag selection, fine-grained schema mapping).

- If the input contains a long list of candidate labels/tags (dozens to hundreds), difficulty should be HIGH (often 7-10),
  even if the sentence is short, because many labels are semantically overlapping.
- If there are multiple independent sub-questions in one sample (e.g., "answer the following 4 questions"), difficulty increases.
- If the correct label is not explicitly named and requires semantic interpretation to disambiguate between similar labels, difficulty increases.
- Reserve 0-3 only for cases where the label set is tiny and the mapping is obvious/unambiguous.

### Question to Analyze:
{question_block}
"""

_CAP_RE = re.compile(
    r"^\s*Capability:\s*(Information Extraction|Numerical Calculation|Domain Knowledge|Complex Reasoning)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SCORE_RE = re.compile(
    r"^\s*DifficultyScore:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/10)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_REASON_RE = re.compile(r"^\s*Reasoning:\s*(.+)$", re.IGNORECASE | re.MULTILINE | re.DOTALL)


def _domain_slug(domain: str) -> str:
    return domain.replace(" ", "_")


def _load_task_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_task_samples(task_name: str, split: str, config: Dict) -> Iterable[Tuple[int, Dict]]:
    key = f"{split}_data"
    if task_name not in config or key not in config[task_name]:
        raise ValueError(f"config 中找不到任务 {task_name} 的 {key} 字段")
    data_path = config[task_name][key]
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def _truncate_text(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _build_question_block(sample: Dict, max_context_chars: int) -> str:
    """
    Put the original sample fields into a compact block.
    We include question + (optional) context/options/program to help the model
    decide the capability category.
    """
    q = sample.get("question", "")
    ctx = sample.get("context", None)
    opts = sample.get("options", None)
    prog = sample.get("program", None)
    task = sample.get("task", None)

    # Light-weight metadata to help difficulty scoring (especially for classification tasks like `finer`)
    ctx_str = str(ctx) if ctx is not None else ""
    # Estimate number of candidate labels/tags if present in the context (heuristic).
    # `finer` often contains: "Here is a list of US GAAP tags options: ,tag1,tag2,..."
    cand_est = None
    if ctx_str:
        marker = "tags options:"
        low = ctx_str.lower()
        if marker in low:
            tail = ctx_str[low.find(marker) + len(marker) :]
            # take a prefix slice to avoid scanning the whole thing
            prefix = tail[: min(len(tail), 6000)]
            # rough: count commas before the instruction part
            prefix_low = prefix.lower()
            cut = prefix_low.find("answer the following")
            if cut != -1:
                prefix = prefix[:cut]
            cand_est = max(prefix.count(","), 0)
    # Estimate number of subquestions (e.g., "1. ... 2. ...")
    subq_est = 0
    if ctx_str:
        for tok in ["\n1.", "\n2.", "\n3.", "\n4.", "\n5.", "\n6."]:
            if tok in ctx_str:
                subq_est += 1

    payload = {
        "task": task,
        "meta": {
            "context_chars": len(ctx_str) if ctx_str else 0,
            "options_count": len(opts) if isinstance(opts, list) else (len(opts) if isinstance(opts, dict) else (None if opts is None else 1)),
            "candidate_labels_estimate": cand_est,
            "subquestions_estimate": subq_est if subq_est else None,
        },
        "question": q,
        "context": _truncate_text(ctx_str, max_context_chars) if ctx is not None else None,
        "options": opts,
        "program": _truncate_text(str(prog), 1200) if prog else None,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_response(text: str) -> Dict[str, Optional[str]]:
    cap_m = _CAP_RE.search(text or "")
    score_m = _SCORE_RE.search(text or "")
    reason_m = _REASON_RE.search(text or "")

    capability = cap_m.group(1).strip() if cap_m else None
    difficulty_score: Optional[float] = None
    if score_m:
        try:
            difficulty_score = float(score_m.group(1))
        except Exception:
            difficulty_score = None
    reasoning = reason_m.group(1).strip() if reason_m else None

    # Normalize
    if capability:
        capability = {
            "information extraction": "Information Extraction",
            "numerical calculation": "Numerical Calculation",
            "domain knowledge": "Domain Knowledge",
            "complex reasoning": "Complex Reasoning",
        }.get(capability.lower(), capability)
    if difficulty_score is not None:
        # clamp into [0, 10]
        if difficulty_score < 0:
            difficulty_score = 0.0
        if difficulty_score > 10:
            difficulty_score = 10.0

    return {"capability": capability, "difficulty_score": difficulty_score, "reasoning": reasoning}


def _load_seen_keys(output_jsonl: str) -> set:
    seen = set()
    if not os.path.exists(output_jsonl):
        return seen
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj.get("uid")
                if key:
                    seen.add(key)
            except Exception:
                continue
    return seen


def _write_jsonl(path: str, obj: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="LLM-based capability+difficulty reclassification")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./StructuredReasoning/data/task_config.json",
        help="任务配置 JSON（包含每个任务的 test_data 路径）",
    )
    # NOTE: domain is intentionally not supported anymore.
    # We always flatten all tasks and classify per-sample capability+difficulty.
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="要标注的数据划分。all 表示依次处理 train/val/test 并分别保存。",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        default="usd_guiji",
        choices=["sambanova", "together", "openai", "usd_guiji"],
        help="LLM API provider",
    )
    parser.add_argument("--model", type=str, default="USD-guiji/deepseek-v3", help="LLM model name")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_context_chars", type=int, default=3500)
    parser.add_argument("--max_samples", type=int, default=0, help=">0 时限制总样本数（用于试跑）")
    parser.add_argument("--output_root", type=str, default="results/StructuredReasoning_run/llm_reclassify_mode")
    parser.add_argument(
        "--output_name",
        type=str,
        default="default",
        help="稳定输出目录名（用于缓存复用）。建议填模型简称/实验名，例如 deepseekv3_prompt_v1",
    )
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已存在 uid 的样本")
    parser.add_argument("--dry_run", action="store_true", help="不调用LLM，仅输出 prompt 预览")
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="逗号分隔的 task 列表（例如 finer,FinKnow）。为空表示跑 ALL_TASKS。",
    )
    args = parser.parse_args()

    # Always flatten tasks (optionally filtered)
    if args.tasks.strip():
        requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in requested if t not in ALL_TASKS]
        if unknown:
            raise ValueError(f"未知 tasks: {unknown}. 可选: {ALL_TASKS}")
        tasks_to_run = requested
    else:
        tasks_to_run = ALL_TASKS

    def _make_out_dir(split: str) -> str:
        # Stable path for caching
        out_dir = os.path.join(args.output_root, args.output_name, split)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    cfg = _load_task_config(args.config_path)

    # init client (lazy import; allow --dry_run without installing openai)
    gen_client = None
    if not args.dry_run:
        from utils.llm import timed_llm_call  # noqa: E402
        from utils.tools import initialize_clients  # noqa: E402
        gen_client, _, _ = initialize_clients(args.api_provider)

    counts = {
        "domain": {},
        "capability": {},
        "difficulty": {},
        "domain_x_difficulty": {},
        "capability_x_difficulty": {},
    }

    overall_processed = 0
    for split in splits:
        out_dir = _make_out_dir(split)
        out_jsonl = os.path.join(out_dir, "classifications.jsonl")
        out_summary = os.path.join(out_dir, "summary.json")
        seen = _load_seen_keys(out_jsonl) if args.resume else set()

        counts = {
            "capability": {},
            "difficulty_score": {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "hist_0_10": {str(i): 0 for i in range(11)},
                "unparsed": 0,
            },
        }

        processed = 0
        for task in tasks_to_run:
            for idx, sample in _iter_task_samples(task, split, cfg):
                uid = f"{task}:{split}:{idx}"
                if uid in seen:
                    continue

                # attach task if missing
                if "task" not in sample:
                    sample["task"] = task

                q_block = _build_question_block(sample, args.max_context_chars)
                prompt = PROMPT_TEMPLATE.format(task_name=task, question_block=q_block)

                if args.dry_run:
                    _write_jsonl(
                        out_jsonl,
                        {"uid": uid, "task": task, "split": split, "index": idx, "prompt": prompt},
                    )
                    processed += 1
                    overall_processed += 1
                    if args.max_samples and overall_processed >= args.max_samples:
                        break
                    continue

                resp, call_info = timed_llm_call(
                    client=gen_client,
                    api_provider=args.api_provider,
                    model=args.model,
                    prompt=prompt,
                    role="classifier",
                    call_id=f"classify_{uid}",
                    max_tokens=args.max_tokens,
                    log_dir=out_dir,
                    temperature=args.temperature,
                )

                parsed = _parse_response(resp)

                rec = {
                    "uid": uid,
                    "task": task,
                    "split": split,
                    "index": idx,
                    "sample": sample,
                    "llm_response": resp,
                    "parsed": parsed,
                }
                _write_jsonl(out_jsonl, rec)
                seen.add(uid)

                cap = parsed.get("capability") or "UNPARSED"
                counts["capability"][cap] = counts["capability"].get(cap, 0) + 1

                score = parsed.get("difficulty_score")
                ds = counts["difficulty_score"]
                if score is None:
                    ds["unparsed"] += 1
                else:
                    ds["count"] += 1
                    ds["min"] = score if ds["min"] is None else min(ds["min"], score)
                    ds["max"] = score if ds["max"] is None else max(ds["max"], score)
                    # integer histogram by rounding to nearest int
                    b = int(round(float(score)))
                    if b < 0:
                        b = 0
                    if b > 10:
                        b = 10
                    ds["hist_0_10"][str(b)] += 1

                processed += 1
                overall_processed += 1
                if args.max_samples and overall_processed >= args.max_samples:
                    break

            if args.max_samples and overall_processed >= args.max_samples:
                break
        # write summary for this split
        # finalize mean
        ds = counts["difficulty_score"]
        if ds["count"] > 0:
            # mean from histogram approximation: weighted by bin centers (ints)
            total = 0.0
            for k, v in ds["hist_0_10"].items():
                total += float(k) * float(v)
            ds["mean"] = total / float(ds["count"])

        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "config_path": args.config_path,
                    "split": split,
                    "tasks": tasks_to_run,
                    "model": args.model,
                    "api_provider": args.api_provider,
                    "processed": processed,
                    "counts": counts,
                    "output_jsonl": out_jsonl,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"[LLM-Reclassify] split={split} done. processed={processed}")
        print(f"[LLM-Reclassify] output: {out_dir}")
        if args.max_samples and overall_processed >= args.max_samples:
            break


if __name__ == "__main__":
    main()


