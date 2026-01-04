#!/usr/bin/env python3
"""
Sample 50 examples per StructuredReasoning task from TRAIN split and (optionally) run llm_reclassify on them.

This script is intended for the "extra test set" experiment to address leakage concerns:
- Experience file (experience.latest.json) stays fixed (learned from the original benchmark runs).
- We create a *new* small test set sampled from train splits, then evaluate AOA on it.
- We also run capability+difficulty reclassification on this sampled set, so we can report
  capability×difficulty tables for AOA on this extra set.

Outputs:
  <out_data_dir>/
    sampled_jsonl/<Task>__sample_<k>_seed<seed>.jsonl
    task_config_aoa_sampled_test_<k>.json

  If --run_reclassify:
    <output_root>/<output_name>/test/classifications.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_TASKS = "FinCode,CodeFinQA,CodeTAT-QA,SEC-NUM,TAT-QA,ConvFinQA,FinKnow,FormulaEval,finer,formula"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_task_config(path: Path) -> Dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(base_config_path: Path, p: str) -> Path:
    p = str(p)
    if p.startswith("./"):
        # task_config.json uses repo-root relative paths like "./StructuredReasoning/data/xxx.jsonl"
        return (_REPO_ROOT / Path(p[2:])).resolve()
    return Path(p).resolve()


def build_sampled_test_config(
    base_config_path: Path,
    tasks: List[str],
    k: int,
    seed: int,
    out_data_dir: Path,
) -> Path:
    cfg = _load_task_config(base_config_path)
    rng = random.Random(seed)

    sampled_dir = out_data_dir / "sampled_jsonl"
    sampled_dir.mkdir(parents=True, exist_ok=True)

    out_cfg: Dict[str, dict] = {}
    skipped: List[str] = []

    for task in tasks:
        if task not in cfg:
            skipped.append(task)
            continue
        train_p = str(cfg[task].get("train_data") or "")
        if not train_p:
            skipped.append(task)
            continue
        train_path = _resolve_path(base_config_path, train_p)
        rows = _read_jsonl(train_path)
        if not rows:
            skipped.append(task)
            continue
        n = min(k, len(rows))
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        picked = [rows[i] for i in idxs[:n]]

        out_jsonl = sampled_dir / f"{task}__sample_{n}_seed{seed}.jsonl"
        _write_jsonl(out_jsonl, picked)

        out_cfg[task] = dict(cfg[task])
        # IMPORTANT: map sampled-train -> test_data so downstream pipelines can reuse `split=test`
        out_cfg[task]["test_data"] = str(out_jsonl)

    out_cfg["_meta"] = {
        "generated_by": "Agents/aoa/sample_testset_and_reclassify.py",
        "base_config": str(base_config_path),
        "seed": seed,
        "k": k,
        "tasks_requested": tasks,
        "tasks_written": sorted([t for t in out_cfg.keys() if not t.startswith("_")]),
        "tasks_skipped": sorted(skipped),
        "note": "test_data is remapped to sampled train jsonl for extra-test evaluation.",
    }

    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_config = out_data_dir / f"task_config_aoa_sampled_test_{k}.json"
    out_config.write_text(json.dumps(out_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_config


def run_reclassify(
    config_path: Path,
    output_root: Path,
    output_name: str,
    api_provider: str,
    model: str,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        "utils/llm_reclassify.py",
        "--config_path",
        str(config_path),
        "--split",
        "test",
        "--output_root",
        str(output_root),
        "--output_name",
        output_name,
        "--api_provider",
        api_provider,
        "--model",
        model,
    ]
    print("[AOA][sample+reclassify] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_config",
        type=str,
        default="StructuredReasoning/data/task_config.json",
        help="Base StructuredReasoning task_config.json",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_TASKS,
        help="Comma-separated tasks to sample from train_data",
    )
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_data_dir",
        type=str,
        default="StructuredReasoning/data/aoa_sampled_test",
        help="Where to store sampled jsonl + derived config",
    )
    ap.add_argument("--run_reclassify", action="store_true", help="If set, run utils/llm_reclassify.py after sampling")
    ap.add_argument(
        "--reclassify_output_root",
        type=str,
        default="results/StructuredReasoning_run/llm_reclassify_mode",
    )
    ap.add_argument("--reclassify_output_name", type=str, default="aoa_sampled_test_labels")
    ap.add_argument("--api_provider", type=str, default="usd_guiji")
    ap.add_argument("--model", type=str, default="USD-guiji/deepseek-v3")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_config = Path(args.base_config)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_data_dir = Path(args.out_data_dir)

    out_config = build_sampled_test_config(
        base_config_path=base_config,
        tasks=tasks,
        k=args.k,
        seed=args.seed,
        out_data_dir=out_data_dir,
    )
    print(f"[AOA][sample+reclassify] Wrote sampled config: {out_config.resolve()}")

    if args.run_reclassify:
        run_reclassify(
            config_path=out_config,
            output_root=Path(args.reclassify_output_root),
            output_name=args.reclassify_output_name,
            api_provider=args.api_provider,
            model=args.model,
        )
        print("[AOA][sample+reclassify] Reclassify finished.")
    else:
        print(
            "[AOA][sample+reclassify] Reclassify not run. To run later:\n"
            f"  python3 -u utils/llm_reclassify.py --config_path {out_config} --split test "
            f"--output_root {args.reclassify_output_root} --output_name {args.reclassify_output_name} "
            f"--api_provider {args.api_provider} --model {args.model}"
        )


if __name__ == "__main__":
    main()


