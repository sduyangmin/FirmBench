#!/usr/bin/env python3
"""
Prepare a leakage-safe "router training set" from train splits.

What it does:
1) For each task, sample N examples from the original train_data jsonl.
2) Write sampled jsonl files under an output directory.
3) Generate a derived task_config JSON where **test_data** points to the sampled file
   (so we can reuse existing StructuredReasoning runner in --mode eval_only).

Rationale:
- We want the router experience (findings / rules) to come from train-only samples.
- Existing pipelines (agents run, llm_reclassify, capability_eval) all assume "test_data".
  By mapping sampled-train -> test_data in a new config, we can reuse everything.

Notes:
- Some tasks in `StructuredReasoning/data/task_config.json` have train_data == test_data
  (i.e., no real train split). By default we SKIP these tasks to avoid leakage.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_task_config(path: Path) -> Dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_probably_no_train(train_path: str, test_path: str) -> bool:
    # Heuristic: same file or train path contains "_test"
    if not train_path or not test_path:
        return True
    if train_path == test_path:
        return True
    low = train_path.lower()
    return "_test" in low and "_train" not in low


def prepare_train50_config(
    base_config_path: Path,
    out_dir: Path,
    tasks: List[str],
    k: int,
    seed: int,
    allow_no_train_tasks: bool,
) -> Tuple[Path, Dict[str, dict]]:
    """
    Returns: (new_config_path, new_config_dict)
    """
    cfg = _load_task_config(base_config_path)

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "sampled_jsonl"
    samples_dir.mkdir(parents=True, exist_ok=True)

    new_cfg: Dict[str, dict] = {}
    skipped: List[str] = []

    for task in tasks:
        if task not in cfg:
            skipped.append(task)
            continue
        train_p = str(cfg[task].get("train_data") or "")
        test_p = str(cfg[task].get("test_data") or "")
        if _is_probably_no_train(train_p, test_p) and not allow_no_train_tasks:
            skipped.append(task)
            continue

        train_path = (_REPO_ROOT / Path(train_p[2:])).resolve() if train_p.startswith("./") else Path(train_p).resolve()
        rows = _read_jsonl(train_path)
        if not rows:
            skipped.append(task)
            continue

        n = min(k, len(rows))
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        picked = [rows[i] for i in idxs[:n]]

        out_jsonl = samples_dir / f"{task}__train_sample_{n}.jsonl"
        _write_jsonl(out_jsonl, picked)

        # New config: keep original train/val (unused), but set test_data to sampled-train.
        new_cfg[task] = dict(cfg[task])
        new_cfg[task]["test_data"] = str(out_jsonl)

    meta = {
        "_generated_by": "Agents/aoa/trainset_prep.py",
        "_base_config": str(base_config_path),
        "_seed": seed,
        "_k_per_task": k,
        "_tasks_requested": tasks,
        "_tasks_written": sorted(list(new_cfg.keys())),
        "_tasks_skipped": sorted(skipped),
        "_note": "test_data is remapped to sampled train jsonl for eval_only usage.",
    }
    wrapper = {"_meta": meta, "tasks": new_cfg}
    out_config = out_dir / f"task_config_aoa_train_sample_{k}.json"
    out_config.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_config, wrapper


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_config",
        type=str,
        default="StructuredReasoning/data/task_config.json",
        help="Base task_config.json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="StructuredReasoning/data/aoa_train_sample",
        help="Output dir for sampled jsonl + derived config",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default="FinCode,CodeFinQA,CodeTAT-QA,SEC-NUM,TAT-QA,ConvFinQA,FinKnow,FormulaEval,finer,formula",
        help="Comma-separated task names",
    )
    ap.add_argument("--k", type=int, default=50, help="Samples per task")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--allow_no_train_tasks",
        action="store_true",
        help="If set, include tasks whose train_data seems to be actually test_data.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_config, wrapper = prepare_train50_config(
        base_config_path=Path(args.base_config),
        out_dir=Path(args.out_dir),
        tasks=tasks,
        k=args.k,
        seed=args.seed,
        allow_no_train_tasks=args.allow_no_train_tasks,
    )
    print(f"[AOA][trainset_prep] Wrote config: {out_config.resolve()}")
    meta = wrapper.get("_meta", {})
    print(f"[AOA][trainset_prep] tasks_written={len(meta.get('_tasks_written', []))} tasks_skipped={len(meta.get('_tasks_skipped', []))}")


if __name__ == "__main__":
    main()


