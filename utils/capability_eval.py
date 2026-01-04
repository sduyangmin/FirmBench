#!/usr/bin/env python3
"""
Capability-based difficulty bucketing + evaluation aggregation for StructuredReasoning_run.

Inputs:
  - LLM reclassification cache:
      results/StructuredReasoning_run/llm_reclassify_mode/<output_name>/<split>/classifications.jsonl
    Each line contains:
      { uid: "<Task>:<split>:<index>", task, split, index, parsed: {capability, difficulty_score, ...}, ... }

  - Evaluation results:
      results/StructuredReasoning_run/<Task>/<agent_method>/<mode>/<timestamp>/test_results.json

This script aligns per-sample correctness (by error indices in test_results.json) with per-sample
capability + difficulty score, then aggregates accuracy by:
  - capability
  - capability + difficulty_bucket (capability-specific rules)

Bucketing rules (per user):
  - Information Extraction: do NOT split by difficulty (report only overall under this capability).
  - Complex Reasoning: only split score==6 as middle and score==7 as hard (fallback: score<7 -> middle, score>=7 -> hard).
  - Numerical Calculation / Domain Knowledge:
        1-3  => easy
        4-6  => middle
        >=7  => hard   (note: user wrote \">7\"; we treat 7 as hard to avoid an unassigned bucket)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Ensure repo root on sys.path when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


try:
    # Reuse robust parsing logic for multiple result formats (ACE/CoT/GEPA/etc.)
    from utils.easyhard import _load_test_results, _build_error_index_set  # type: ignore
except Exception:
    # Fallback: keep script runnable even if imports change; we re-implement minimal parts.
    import math

    def _load_test_results(test_results_path: str) -> Tuple[float, int, List[dict]]:
        with open(test_results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        errors: List[Dict[str, Any]] = []
        total = 0
        accuracy = 0.0

        if "test_results" in payload and isinstance(payload["test_results"], dict):
            tr = payload["test_results"]
            total = tr.get("total", total)
            accuracy = tr.get("accuracy", accuracy) or accuracy
            if "window_results" in tr and isinstance(tr["window_results"], list):
                win_errs: List[Dict[str, Any]] = []
                for w in tr["window_results"]:
                    start = w.get("start", 0)
                    for err in w.get("errors", []) or []:
                        new_err = dict(err)
                        if "index" in new_err and "global_index" not in new_err:
                            new_err["global_index"] = start + int(new_err["index"])
                        win_errs.append(new_err)
                if win_errs:
                    errors = win_errs
            if not errors and "errors" in tr:
                errors = tr.get("errors", []) or []

        if not errors and "test_error_log" in payload and isinstance(payload["test_error_log"], dict):
            errors = payload["test_error_log"].get("errors", []) or []
            accuracy = payload["test_error_log"].get("accuracy", accuracy)
        if not errors and "error_log" in payload and isinstance(payload["error_log"], dict):
            errors = payload["error_log"].get("errors", []) or []
            accuracy = payload["error_log"].get("accuracy", accuracy)
        if not errors and "errors" in payload and isinstance(payload["errors"], list):
            errors = payload["errors"]

        if "test_accuracy" in payload:
            accuracy = payload.get("test_accuracy", accuracy)
        if "total" in payload and (not total or total == 0):
            total = payload.get("total", total) or total
        total = int(total) if total else 0
        return accuracy or 0.0, total, errors

    def _build_error_index_set(errors: List[dict]) -> Tuple[set, Dict[int, dict]]:
        idx_set = set()
        idx_to_error: Dict[int, dict] = {}
        for err in errors:
            key = None
            if "global_index" in err:
                key = int(err["global_index"])
            elif "index" in err:
                key = int(err["index"])
            if key is not None:
                idx_set.add(key)
                idx_to_error[key] = err
        return idx_set, idx_to_error


CAP_CANON = {
    "information extraction": "Information Extraction",
    "numerical calculation": "Numerical Calculation",
    "domain knowledge": "Domain Knowledge",
    "complex reasoning": "Complex Reasoning",
}


@dataclass(frozen=True)
class ParsedLabel:
    capability: str
    difficulty_score: float
    reasoning: Optional[str] = None


CAPS = [
    "Information Extraction",
    "Numerical Calculation",
    "Domain Knowledge",
    "Complex Reasoning",
]

DIFFS = ["easy", "middle", "hard"]


def _canon_capability(cap: str) -> str:
    k = str(cap or "").strip()
    k_low = k.lower()
    return CAP_CANON.get(k_low, k)


def _capability_difficulty_bucket(capability: str, score: float) -> Optional[str]:
    """
    Returns bucket in {easy, middle, hard} or None if this capability is not bucketed.
    """
    cap = _canon_capability(capability)
    if cap == "Information Extraction":
        return None
    if cap == "Complex Reasoning":
        # Not bucketed: too few samples and often non-monotonic; report overall only.
        return None

    # Numerical Calculation / Domain Knowledge
    if score <= 3.0:
        return "easy"
    if score <= 6.0:
        return "middle"
    # NOTE: user wrote \">7\"; we treat 7 as hard to avoid an unassigned score bucket.
    return "hard"


def _latest_run_dir(task_root: Path, agent_method: str, mode: str) -> Optional[Path]:
    candidate_root = task_root / agent_method / mode
    if not candidate_root.exists():
        return None
    subdirs = [p for p in candidate_root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


def _discover_runs(results_root: Path) -> List[Tuple[str, str, str, Path]]:
    """
    Returns list of (task, agent_method, mode, run_dir) for latest run per task/agent/mode.
    """
    skip_tasks = {"easyhard_mode", "llm_reclassify_mode", "consulting_interview"}
    runs: List[Tuple[str, str, str, Path]] = []

    for task_dir in results_root.iterdir():
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        if task in skip_tasks:
            continue

        # structure: task/agent/mode/timestamp/test_results.json
        for agent_dir in task_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            agent = agent_dir.name
            for mode_dir in agent_dir.iterdir():
                if not mode_dir.is_dir():
                    continue
                mode = mode_dir.name
                latest = _latest_run_dir(task_dir, agent, mode)
                if latest is None:
                    continue
                tr = latest / "test_results.json"
                if tr.exists():
                    runs.append((task, agent, mode, latest))
    # de-dup (task,agent,mode) keeping latest only
    best: Dict[Tuple[str, str, str], Path] = {}
    for task, agent, mode, run_dir in runs:
        k = (task, agent, mode)
        if k not in best or run_dir.stat().st_mtime > best[k].stat().st_mtime:
            best[k] = run_dir
    return [(t, a, m, d) for (t, a, m), d in best.items()]


def _load_labels(classify_root: Path, split: str) -> Dict[str, List[ParsedLabel]]:
    """
    Returns mapping: task -> labels_by_index (list index == sample index).
    """
    jsonl_path = classify_root / split / "classifications.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing classifications jsonl: {jsonl_path}")

    tmp: Dict[str, Dict[int, ParsedLabel]] = defaultdict(dict)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task = obj.get("task") or ""
            idx = obj.get("index")
            parsed = obj.get("parsed") or {}
            cap = parsed.get("capability")
            score = parsed.get("difficulty_score")
            if task is None or idx is None or cap is None or score is None:
                continue
            try:
                idx_i = int(idx)
                score_f = float(score)
            except Exception:
                continue
            cap_c = _canon_capability(str(cap))
            tmp[str(task)][idx_i] = ParsedLabel(
                capability=cap_c,
                difficulty_score=score_f,
                reasoning=parsed.get("reasoning"),
            )

    labels_by_task: Dict[str, List[ParsedLabel]] = {}
    for task, idx_map in tmp.items():
        if not idx_map:
            continue
        max_idx = max(idx_map.keys())
        labels = [None] * (max_idx + 1)  # type: ignore[list-item]
        for i, lab in idx_map.items():
            if 0 <= i < len(labels):
                labels[i] = lab
        # fill holes with a safe default to keep alignment robust
        # (shouldn't happen if classifications is complete)
        fallback = ParsedLabel(capability="__MISSING__", difficulty_score=float("nan"), reasoning=None)
        labels_filled: List[ParsedLabel] = [lab if lab is not None else fallback for lab in labels]  # type: ignore[arg-type]
        labels_by_task[task] = labels_filled

    return labels_by_task


def _acc(correct: int, total: int) -> float:
    return (correct / total) if total else 0.0


def _parse_task_overrides(s: str) -> Dict[str, str]:
    """
    Parse overrides like: "finer=Information Extraction,SEC-NUM=Information Extraction"
    """
    out: Dict[str, str] = {}
    s = (s or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        task, cap = p.split("=", 1)
        task = task.strip()
        cap = cap.strip()
        if task and cap:
            out[task] = cap
    return out


def _export_csv_table(
    out_root: Path,
    mode: str,
    timestamp: str,
    agent_cols: List[str],
    agent_col_to_dir: Dict[str, str],
    out_csv_path: Path,
) -> None:
    """
    Export a single CSV table with rows: capability x difficulty_bucket, columns: agents.
    Reads summaries from: out_root/<agent_dir>/<mode>/<timestamp>/summary.json
    """
    import csv

    def load_summary(p: Path) -> dict:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def cell_value(summary: dict, cap: str, diff: str) -> str:
        by_cap = summary.get("by_capability", {}) or {}
        by_cap_bucket = summary.get("by_capability_bucket", {}) or {}

        cap_acc = None
        if cap in by_cap and isinstance(by_cap[cap], dict):
            cap_acc = by_cap[cap].get("accuracy", None)

        if cap in ("Information Extraction", "Complex Reasoning"):
            # Not bucketed: repeat to fill easy/middle/hard rows (matches user's table shape)
            return "" if cap_acc is None else f"{float(cap_acc):.4f}"

        # Numerical Calculation / Domain Knowledge
        cap_b = by_cap_bucket.get(cap, {}) or {}
        if diff in cap_b and isinstance(cap_b[diff], dict) and cap_b[diff].get("total", 0) > 0:
            return f"{float(cap_b[diff]['accuracy']):.4f}"
        return ""

    summaries: Dict[str, dict] = {}
    for col in agent_cols:
        agent_dir = agent_col_to_dir.get(col)
        if not agent_dir:
            continue
        summary_path = out_root / agent_dir / mode / timestamp / "summary.json"
        if summary_path.exists():
            summaries[col] = load_summary(summary_path)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capability", "difficulty", *agent_cols])
        for cap in CAPS:
            for diff in DIFFS:
                row = [cap, diff]
                for col in agent_cols:
                    summ = summaries.get(col)
                    row.append("" if summ is None else cell_value(summ, cap, diff))
                w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="results/StructuredReasoning_run")
    ap.add_argument(
        "--classify_root",
        type=str,
        default="results/StructuredReasoning_run/llm_reclassify_mode/capability_difficulty_score_v1_merged_finer_try50",
        help="Root that contains <split>/classifications.jsonl",
    )
    ap.add_argument(
        "--label_split",
        type=str,
        default="",
        choices=["", "train", "val", "test"],
        help=(
            "Which split of LLM reclassification cache to use. "
            "If empty, it is inferred from evaluation mode: online/eval_only->test, offline->train."
        ),
    )
    ap.add_argument("--out_dir", type=str, default="results/StructuredReasoning_run/capability_eval_mode")
    ap.add_argument(
        "--only_mode",
        type=str,
        default="online",
        help="If set, only evaluate this mode (default: online). Use empty string to include all modes.",
    )
    ap.add_argument("--only_agent", type=str, default="", help="If set, only evaluate this agent method (e.g. cot)")
    ap.add_argument(
        "--export_csv",
        action="store_true",
        help="If set, also export a CSV table (capability x difficulty_bucket by agent) into the same output root.",
    )
    ap.add_argument(
        "--csv_agents",
        type=str,
        default="cot,self-refine,reflexion,debate,discussion,dc,gepa,ace,amem,aoa",
        help="Comma-separated agent columns for CSV export: cot,self-refine,reflexion,debate,discussion,dc,gepa,ace,amem,aoa",
    )
    ap.add_argument(
        "--csv_out",
        type=str,
        default="",
        help="Optional explicit CSV output path. If empty, writes to <out_dir>/_tables/<mode>/<timestamp>/capability_eval.csv",
    )
    ap.add_argument(
        "--task_capability_override",
        type=str,
        default="",
        help=(
            "Optional task->capability overrides, e.g. "
            "\"finer=Information Extraction\". Multiple via comma."
        ),
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    classify_root = Path(args.classify_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Decide which split of *labels* to use.
    #
    # IMPORTANT: We align against indices used in each run's `test_results.json`, which are
    # indices of the evaluation set (almost always the test set) even for `offline` mode.
    # Therefore default label split is `test` for all modes unless explicitly overridden.
    def infer_label_split(_mode: str) -> str:
        return args.label_split or "test"

    labels_cache: Dict[str, Dict[str, List[ParsedLabel]]] = {}
    def get_labels(split: str) -> Dict[str, List[ParsedLabel]]:
        if split not in labels_cache:
            labels_cache[split] = _load_labels(classify_root, split)
        return labels_cache[split]

    task_cap_override = _parse_task_overrides(args.task_capability_override)

    runs = _discover_runs(results_root)
    if args.only_mode:
        runs = [r for r in runs if r[2] == args.only_mode]
    if args.only_agent:
        runs = [r for r in runs if r[1] == args.only_agent]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Aggregate by agent/mode across tasks
    agg: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for task, agent, mode, run_dir in sorted(runs, key=lambda x: (x[1], x[2], x[0])):
        label_split = infer_label_split(mode)
        labels_by_task = get_labels(label_split)
        if task not in labels_by_task:
            # Skip tasks without LLM labels in this split
            continue
        labels = labels_by_task[task]
        # Apply optional task-level capability override.
        if task in task_cap_override:
            override_cap = task_cap_override[task]
            labels = [
                ParsedLabel(capability=_canon_capability(override_cap), difficulty_score=lab.difficulty_score, reasoning=lab.reasoning)
                for lab in labels
            ]

        test_results_path = run_dir / "test_results.json"
        _, _, errors = _load_test_results(str(test_results_path))
        err_set, _ = _build_error_index_set(errors)

        total = len(labels)
        correct_total = 0

        # stats structures
        by_cap: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0})
        by_cap_bucket: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
            lambda: defaultdict(lambda: {"total": 0, "correct": 0})
        )

        per_sample_path = None
        # Only dump per-sample for agent/mode aggregate directory once (append across tasks).
        key = (agent, mode)
        if key not in agg:
            agg[key] = {
                "agent_method": agent,
                "mode": mode,
                "label_split": label_split,
                "tasks": {},
                "by_capability": defaultdict(lambda: {"total": 0, "correct": 0}),
                "by_capability_bucket": defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0})),
            }
            # Create dir + per-sample jsonl
            out_dir = out_root / agent / mode / timestamp
            out_dir.mkdir(parents=True, exist_ok=True)
            per_sample_path = out_dir / "per_sample.jsonl"
            agg[key]["_out_dir"] = str(out_dir)
            agg[key]["_per_sample_path"] = str(per_sample_path)

        per_sample_path = Path(agg[key]["_per_sample_path"])

        # Evaluate per sample
        with per_sample_path.open("a", encoding="utf-8") as wf:
            for idx, lab in enumerate(labels):
                if not isinstance(lab.difficulty_score, float) or lab.difficulty_score != lab.difficulty_score:
                    # NaN / missing label
                    continue
                is_correct = idx not in err_set
                correct_total += 1 if is_correct else 0

                cap = lab.capability
                by_cap[cap]["total"] += 1
                by_cap[cap]["correct"] += 1 if is_correct else 0

                bucket = _capability_difficulty_bucket(cap, lab.difficulty_score)
                if bucket is not None:
                    by_cap_bucket[cap][bucket]["total"] += 1
                    by_cap_bucket[cap][bucket]["correct"] += 1 if is_correct else 0

                wf.write(
                    json.dumps(
                        {
                            "task": task,
                            "agent_method": agent,
                            "mode": mode,
                            "index": idx,
                            "capability": cap,
                            "difficulty_score": lab.difficulty_score,
                            "difficulty_bucket": bucket,
                            "correct": bool(is_correct),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # finalize per-run task stats
        task_payload = {
            "task": task,
            "run_dir": str(run_dir),
            "label_split": label_split,
            "total_samples": sum(v["total"] for v in by_cap.values()),
            "overall_accuracy": _acc(correct_total, total),
            "by_capability": {},
            "by_capability_bucket": {},
        }
        for cap, s in by_cap.items():
            task_payload["by_capability"][cap] = {
                **s,
                "accuracy": _acc(s["correct"], s["total"]),
            }
        for cap, buckets in by_cap_bucket.items():
            task_payload["by_capability_bucket"][cap] = {}
            for b, s in buckets.items():
                task_payload["by_capability_bucket"][cap][b] = {
                    **s,
                    "accuracy": _acc(s["correct"], s["total"]),
                }

        agg[key]["tasks"][task] = task_payload

        # merge into global agg (agent/mode)
        for cap, s in by_cap.items():
            agg[key]["by_capability"][cap]["total"] += s["total"]
            agg[key]["by_capability"][cap]["correct"] += s["correct"]
        for cap, buckets in by_cap_bucket.items():
            for b, s in buckets.items():
                agg[key]["by_capability_bucket"][cap][b]["total"] += s["total"]
                agg[key]["by_capability_bucket"][cap][b]["correct"] += s["correct"]

    # write summary per agent/mode
    written = 0
    for (agent, mode), payload in agg.items():
        out_dir = Path(payload["_out_dir"])

        # compute accuracies
        by_cap_out: Dict[str, Any] = {}
        for cap, s in payload["by_capability"].items():
            by_cap_out[cap] = {**s, "accuracy": _acc(s["correct"], s["total"])}

        by_cap_bucket_out: Dict[str, Any] = {}
        for cap, buckets in payload["by_capability_bucket"].items():
            by_cap_bucket_out[cap] = {}
            for b, s in buckets.items():
                by_cap_bucket_out[cap][b] = {**s, "accuracy": _acc(s["correct"], s["total"])}

        total_all = sum(v["total"] for v in payload["by_capability"].values())
        correct_all = sum(v["correct"] for v in payload["by_capability"].values())

        summary = {
            "agent_method": agent,
            "mode": mode,
            "label_split": payload["label_split"],
            "total_samples": total_all,
            "overall_accuracy": _acc(correct_all, total_all),
            "by_capability": by_cap_out,
            "by_capability_bucket": by_cap_bucket_out,
            "tasks": payload["tasks"],
            "bucketing_rules": {
                "Information Extraction": "no bucketing (report overall only)",
                "Complex Reasoning": "no bucketing (report overall only)",
                "Numerical Calculation": {"1-3": "easy", "4-6": "middle", ">=7": "hard"},
                "Domain Knowledge": {"1-3": "easy", "4-6": "middle", ">=7": "hard"},
                "note": "User spec said '>7' for hard; implementation treats score==7 as hard to avoid an unassigned score.",
            },
        }

        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        written += 1

    # Optional: export CSV for a single mode (this matches user's use-case: online).
    if args.export_csv:
        mode = args.only_mode or "online"
        agent_cols = [a.strip() for a in args.csv_agents.split(",") if a.strip()]
        agent_col_to_dir = {
            "cot": "cot",
            "self-refine": "self_refine",
            "reflexion": "reflexion",
            "debate": "debate",
            "discussion": "discussion",
            "dc": "dynamic_cheatsheet",
            "gepa": "gepa",
            "ace": "ace",
            "amem": "amem",
            "aoa": "aoa",
        }
        out_csv_path = (
            Path(args.csv_out)
            if args.csv_out
            else (out_root / "_tables" / mode / timestamp / "capability_eval.csv")
        )
        _export_csv_table(
            out_root=out_root,
            mode=mode,
            timestamp=timestamp,
            agent_cols=agent_cols,
            agent_col_to_dir=agent_col_to_dir,
            out_csv_path=out_csv_path,
        )
        print(f"Wrote CSV: {out_csv_path.resolve()}")

    override = args.label_split or "default(test)"
    mode_hint = args.only_mode if args.only_mode else "ALL"
    print(
        f"Evaluated mode={mode_hint} under {results_root} using label_split={override}. "
        f"Wrote {written} summaries under: {out_root.resolve()}"
    )


if __name__ == "__main__":
    main()


