#!/usr/bin/env python3
"""
Trace-based experience extraction for AOA.

Instead of reading an aggregated capability_eval.csv, we:
1) Discover each (task, agent_method) run directory under results/StructuredReasoning_run.
2) Parse test_results.json (see utils/easyhard.py parsing logic) to get error indices.
3) Sample up to K examples per (task, agent) covering capability/difficulty and correct/incorrect.
4) For each sampled example, load the agent trace from detailed_llm_logs/*_{call_id}_*.json (if available),
   and ask an LLM to produce one experience bullet.
5) After collecting all bullets, ask the LLM to synthesize meta routing policy.
6) Save to experience.latest.json (or a specified output path).

This is designed to reduce "table-max" bias and make experience grounded in actual reasoning traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure repo root on sys.path when running as a script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.llm import timed_llm_call  # noqa: E402
from utils.playbook_utils import extract_json_from_text  # noqa: E402
from utils.tools import initialize_clients  # noqa: E402
from utils.easyhard import _load_test_results, _build_error_index_set  # noqa: E402
from StructuredReasoning.data_processor import DataProcessor  # noqa: E402


def _canon_agent_name(s: str) -> str:
    s = (s or "").strip()
    alias = {"self-refine": "self_refine", "dc": "dynamic_cheatsheet"}
    return alias.get(s, s)


def _repo_relative_path(p: str) -> Path:
    p = str(p)
    if p.startswith("./"):
        return (_ROOT / Path(p[2:])).resolve()
    return Path(p).resolve()

def _parse_csv_set(s: str) -> set[str]:
    s = (s or "").strip()
    if not s:
        return set()
    return {x.strip().lower() for x in s.split(",") if x.strip()}

def _truncate_text(s: Any, max_chars: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."

def _compact_bullets_for_meta(
    bullets: List[Dict[str, Any]],
    *,
    bullet_max_chars: int,
    diagnosis_max_chars: int,
    keep_routing_hint: bool,
    keep_confidence: bool,
) -> List[Dict[str, Any]]:
    """
    Reduce token usage for meta synthesis by keeping only high-signal fields and truncating long text.
    This is important when meta_max_bullets is large (context length limits).
    """
    out: List[Dict[str, Any]] = []
    for b in bullets:
        if not isinstance(b, dict):
            continue
        tags = _bullet_tags(b)
        rec: Dict[str, Any] = {
            "bullet": _truncate_text(b.get("bullet"), bullet_max_chars),
            "tags": {
                "agent_method": tags.get("agent_method") or tags.get("agent"),
                "task_name": tags.get("task_name"),
                "capability": tags.get("capability"),
                "difficulty_bucket": tags.get("difficulty_bucket"),
            },
            "outcome": b.get("outcome"),
            "diagnosis": _truncate_text(b.get("diagnosis"), diagnosis_max_chars),
        }
        if keep_routing_hint and "routing_hint" in b:
            rec["routing_hint"] = b.get("routing_hint")
        if keep_confidence and "confidence" in b:
            rec["confidence"] = b.get("confidence")
        out.append(rec)
    return out

def _now_ts() -> str:
    # Keep consistent with other scripts in this repo (local time).
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _maybe_timestamp_run_dir(out_path: Path, log_dir: Path) -> Tuple[Path, Path, Optional[Path]]:
    """
    If out_path is directly under .../aoa_mode/experience/, create a timestamped run dir:
      .../aoa_mode/experience/<ts>/
    and relocate:
      out_path -> <run_dir>/<basename(out_path)>
      log_dir  -> <run_dir>/trace_logs   (if log_dir was also under .../aoa_mode/experience/)

    Returns: (new_out_path, new_log_dir, run_dir_or_none)
    """
    try:
        base = out_path.parent
        if base.name != "experience":
            return out_path, log_dir, None
        run_dir = base / _now_ts()
        new_out = run_dir / out_path.name

        # Move log_dir only if it lives under the same base experience dir (default behavior).
        try:
            log_dir.relative_to(base)
            new_log = run_dir / "trace_logs"
        except Exception:
            new_log = log_dir

        return new_out, new_log, run_dir
    except Exception:
        return out_path, log_dir, None


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _job_key(task: str, agent: str, index: int) -> str:
    return f"{task}::{_canon_agent_name(agent)}::{int(index)}"

def _safe_str(x: Any, default: str = "NA") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default

def _stable_hash_int(s: str) -> int:
    """
    Python's built-in hash() is salted per-process (PYTHONHASHSEED), so it is not stable across runs.
    We use md5 for a small, stable, deterministic integer hash.
    """
    s = str(s or "")
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _bullet_sort_key(b: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    """
    Provide a deterministic ordering for bullets to make meta sampling reproducible even when
    bullets are generated/loaded in a non-deterministic order (e.g., parallel extraction).
    """
    tags = _bullet_tags(b)
    return (
        _safe_str(tags.get("task_name")),
        _safe_str(tags.get("agent_method") or tags.get("agent")),
        _safe_str(tags.get("capability")),
        _safe_str(tags.get("difficulty_bucket")),
        _safe_str(b.get("outcome")),
        _safe_str(b.get("bullet")),
    )

def _meta_group_key(b: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Group bullets for meta synthesis sampling.
    We intentionally include agent + (capability,difficulty,outcome) to keep meta input diverse.
    """
    tags = b.get("tags") or {}
    if not isinstance(tags, dict):
        tags = {}
    agent = _safe_str(tags.get("agent_method") or tags.get("agent") or tags.get("agent_name"))
    cap = _safe_str(tags.get("capability"))
    diff = _safe_str(tags.get("difficulty_bucket"))
    outcome = _safe_str(b.get("outcome") or tags.get("outcome"))
    return agent, cap, diff, outcome

def _bullet_tags(b: Dict[str, Any]) -> Dict[str, Any]:
    tags = b.get("tags") or {}
    return tags if isinstance(tags, dict) else {}

def _stratified_round_robin_pick(
    idxs: List[int],
    *,
    bullets: List[Dict[str, Any]],
    key_fn,
    rng: random.Random,
    k: int,
) -> List[int]:
    """
    Pick up to k indices from idxs using round-robin across groups defined by key_fn(bullet).
    Deterministic given rng state.
    """
    if k <= 0 or not idxs:
        return []
    # group -> indices (shuffled)
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i in idxs:
        try:
            groups[key_fn(bullets[i])].append(i)
        except Exception:
            groups[("NA",)].append(i)
    for _, g in groups.items():
        rng.shuffle(g)
    keys = list(groups.keys())
    rng.shuffle(keys)

    picked: List[int] = []
    progressed = True
    while len(picked) < k and progressed and keys:
        progressed = False
        for kk in keys:
            if len(picked) >= k:
                break
            g = groups.get(kk) or []
            if not g:
                continue
            picked.append(g.pop())
            progressed = True
    return picked

def _sample_meta_bullets(
    bullets: List[Dict[str, Any]],
    *,
    n: int,
    seed: int,
    strategy: str,
) -> List[Dict[str, Any]]:
    """
    Select bullets passed into meta synthesizer.
    - head: first N bullets (old behavior; can be order-biased)
    - shuffle: global shuffle then take N
    - stratified: round-robin across (agent,cap,diff,outcome) groups
    - capability_stratified: allocate ~equal quota per capability, then round-robin within each capability across (agent,diff,outcome)
    """
    if n <= 0:
        return []
    # Make meta sampling robust to input order (e.g., from parallel extraction / jsonl appends).
    bullets = sorted(bullets, key=_bullet_sort_key)
    if len(bullets) <= n:
        return bullets
    strategy = (strategy or "stratified").strip().lower()
    rng = random.Random(int(seed))

    if strategy == "head":
        return bullets[:n]
    if strategy == "shuffle":
        idxs = list(range(len(bullets)))
        rng.shuffle(idxs)
        return [bullets[i] for i in idxs[:n]]

    idxs_all = list(range(len(bullets)))

    if strategy in {"capability_stratified", "cap_stratified", "capability"}:
        # 1) bucket by capability
        cap_to_idxs: Dict[str, List[int]] = defaultdict(list)
        for i in idxs_all:
            tags = _bullet_tags(bullets[i])
            cap = _safe_str(tags.get("capability"))
            cap_to_idxs[cap].append(i)
        caps = sorted(cap_to_idxs.keys())
        rng.shuffle(caps)
        if not caps:
            return [bullets[i] for i in idxs_all[:n]]

        base = n // len(caps)
        rem = n % len(caps)
        picked: List[int] = []
        for j, cap in enumerate(caps):
            quota = base + (1 if j < rem else 0)
            cap_idxs = cap_to_idxs.get(cap) or []
            # within cap: round-robin across (agent,diff,outcome)
            picked.extend(
                _stratified_round_robin_pick(
                    cap_idxs,
                    bullets=bullets,
                    rng=rng,
                    k=quota,
                    key_fn=lambda bb: (
                        _safe_str(_bullet_tags(bb).get("agent_method") or _bullet_tags(bb).get("agent")),
                        _safe_str(_bullet_tags(bb).get("difficulty_bucket")),
                        _safe_str(bb.get("outcome")),
                    ),
                )
            )
        # If some caps couldn't fill quota, top-up from remaining pool with global stratified.
        if len(picked) < n:
            picked_set = set(picked)
            remaining = [i for i in idxs_all if i not in picked_set]
            need = n - len(picked)
            picked.extend(
                _stratified_round_robin_pick(
                    remaining,
                    bullets=bullets,
                    rng=rng,
                    k=need,
                    key_fn=_meta_group_key,
                )
            )
        return [bullets[i] for i in picked[:n]]

    # stratified (default): round-robin across (agent,cap,diff,outcome)
    picked = _stratified_round_robin_pick(
        idxs_all,
        bullets=bullets,
        rng=rng,
        k=n,
        key_fn=_meta_group_key,
    )
    return [bullets[i] for i in picked]


def _difficulty_bucket(score: Optional[float]) -> str:
    if score is None:
        return "NA"
    try:
        s = float(score)
    except Exception:
        return "NA"
    if s <= 3.0:
        return "easy"
    if s <= 6.0:
        return "middle"
    return "hard"


@dataclass(frozen=True)
class Label:
    capability: str
    difficulty_score: Optional[float]


def _load_labels(classifications_jsonl: Path) -> Dict[str, Dict[int, Label]]:
    """
    Returns: task -> (index -> Label)
    """
    out: Dict[str, Dict[int, Label]] = defaultdict(dict)
    if not classifications_jsonl.exists():
        return out
    with classifications_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task = str(obj.get("task") or "")
            idx = obj.get("index")
            parsed = obj.get("parsed") or {}
            if idx is None or not task:
                continue
            try:
                i = int(idx)
            except Exception:
                continue
            out[task][i] = Label(
                capability=str(parsed.get("capability") or ""),
                difficulty_score=parsed.get("difficulty_score"),
            )
    return out


def _discover_run_dir(results_root: Path, task: str, agent: str, mode: str, timestamp: str) -> Optional[Path]:
    rd = results_root / task / agent / mode / timestamp
    if rd.exists():
        return rd
    return None


def _latest_timestamp(results_root: Path, task: str, agent: str, mode: str) -> Optional[str]:
    base = results_root / task / agent / mode
    if not base.exists():
        return None
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0].name


def _load_task_samples(task_config_path: Path, task: str) -> List[Dict[str, Any]]:
    cfg = json.loads(task_config_path.read_text(encoding="utf-8"))
    # support wrapper style configs
    if "tasks" in cfg and isinstance(cfg["tasks"], dict):
        cfg = cfg["tasks"]
    if task not in cfg:
        return []
    test_path = _repo_relative_path(str(cfg[task].get("test_data") or ""))
    if not test_path.exists():
        return []
    raw: List[dict] = []
    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    proc = DataProcessor(task_name=task)
    return proc.process_task_data(raw)


def _find_trace_log(detailed_dir: Path, call_id: str) -> Optional[Path]:
    """
    timed_llm_call logs as: <role>_<call_id>_<timestamp>.json
    For evaluation, call_id is typically 'test_eval_<i>'.
    """
    if not detailed_dir.exists():
        return None
    # simple scan (directory is usually not huge for sampled runs)
    for p in detailed_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if f"_{call_id}_" in name and name.endswith(".json"):
            return p
    return None


def _sample_indices(
    task: str,
    agent: str,
    total: int,
    error_idx_set: set,
    labels: Dict[int, Label],
    k: int,
    seed: int,
) -> List[int]:
    """
    Stratified sampling across (capability, difficulty_bucket, correct/incorrect).
    """
    rng = random.Random(seed + (_stable_hash_int(task) % 100000) + (_stable_hash_int(agent) % 100000))
    groups: Dict[Tuple[str, str, bool], List[int]] = defaultdict(list)
    for i in range(total):
        lab = labels.get(i)
        cap = lab.capability if lab and lab.capability else "UNKNOWN"
        bucket = _difficulty_bucket(lab.difficulty_score if lab else None)
        is_correct = i not in error_idx_set
        groups[(cap, bucket, is_correct)].append(i)
    for g in groups.values():
        rng.shuffle(g)

    picked: List[int] = []
    # round-robin across groups
    keys = list(groups.keys())
    rng.shuffle(keys)
    while len(picked) < min(k, total):
        progressed = False
        for key in keys:
            if len(picked) >= min(k, total):
                break
            lst = groups[key]
            if lst:
                picked.append(lst.pop())
                progressed = True
        if not progressed:
            break
    picked = sorted(set(picked))
    return picked


def _call_llm_json(
    client,
    api_provider: str,
    model: str,
    system_prompt: str,
    payload: Dict[str, Any],
    call_id: str,
    max_tokens: int,
    temperature: float,
    log_dir: Optional[str],
) -> Dict[str, Any]:
    user_input = json.dumps(payload, ensure_ascii=False)
    raw, _info = timed_llm_call(
        client=client,
        api_provider=api_provider,
        model=model,
        prompt=user_input,
        role="aoa_experience",
        call_id=call_id,
        max_tokens=max_tokens,
        log_dir=log_dir,
        use_json_mode=True,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    obj = extract_json_from_text(raw) or {}
    return obj if isinstance(obj, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="results/StructuredReasoning_run")
    ap.add_argument("--mode", type=str, default="online", choices=["online", "eval_only", "offline"])
    ap.add_argument("--tasks", type=str, default="")
    ap.add_argument(
        "--exclude_tasks",
        type=str,
        default="",
        help="Comma-separated task names to exclude (case-insensitive).",
    )
    ap.add_argument(
        "--agents",
        type=str,
        default="cot,ace,amem,self_refine,reflexion,gepa,debate,discussion,dynamic_cheatsheet",
    )
    ap.add_argument("--timestamp", type=str, default="", help="If empty, use latest per (task,agent).")
    ap.add_argument("--task_config", type=str, default="StructuredReasoning/data/task_config.json")
    ap.add_argument(
        "--classifications_jsonl",
        type=str,
        default="results/StructuredReasoning_run/llm_reclassify_mode/capability_difficulty_score_v1_merged_finer_try50/test/classifications.jsonl",
    )
    ap.add_argument("--k_per_task_agent", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallelism for per-sample trace->bullet extraction. Use small numbers to avoid rate limits.",
    )

    ap.add_argument("--api_provider", type=str, default="usd_guiji")
    ap.add_argument("--model", type=str, default="USD-guiji/deepseek-v3")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument(
        "--out",
        type=str,
        default="results/StructuredReasoning_run/aoa_mode/experience/experience.latest.json",
    )
    ap.add_argument("--log_dir", type=str, default="results/StructuredReasoning_run/aoa_mode/experience/trace_logs")
    ap.add_argument("--meta_max_bullets", type=int, default=600, help="Max bullets passed into meta synthesizer.")
    ap.add_argument(
        "--meta_compact",
        action="store_true",
        help="If set, compact/truncate bullets before meta synthesis to reduce token usage.",
    )
    ap.add_argument(
        "--meta_bullet_max_chars",
        type=int,
        default=280,
        help="Max characters kept for each bullet text when --meta_compact is set.",
    )
    ap.add_argument(
        "--meta_diagnosis_max_chars",
        type=int,
        default=180,
        help="Max characters kept for each diagnosis when --meta_compact is set.",
    )
    ap.add_argument(
        "--meta_keep_routing_hint",
        action="store_true",
        help="When --meta_compact is set, keep routing_hint field (may increase tokens).",
    )
    ap.add_argument(
        "--meta_keep_confidence",
        action="store_true",
        help="When --meta_compact is set, keep confidence field (small token increase).",
    )
    ap.add_argument(
        "--meta_target_rules",
        type=int,
        default=10,
        help="Target number of routing_policy.rules for meta synthesizer to generate (soft target).",
    )
    ap.add_argument(
        "--meta_min_rules",
        type=int,
        default=12,
        help="Minimum number of routing_policy.rules requested (soft target).",
    )
    ap.add_argument(
        "--meta_max_rules",
        type=int,
        default=40,
        help="Maximum number of routing_policy.rules requested (soft target).",
    )
    ap.add_argument(
        "--meta_sampling",
        type=str,
        default="stratified",
        choices=["capability_stratified", "stratified", "shuffle", "head"],
        help="How to select bullets for meta synthesis (default: stratified).",
    )
    ap.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Write incremental jsonl + progress.json every N extracted bullets (0 disables).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If set, resume from existing checkpoints under the output run directory (skip processed jobs).",
    )
    ap.add_argument(
        "--meta_only",
        action="store_true",
        help=(
            "If set, skip trace->bullet extraction entirely and ONLY redo meta synthesis "
            "from existing bullets.jsonl under the output directory. "
            "Useful after editing Agents/aoa/prompts/meta_experience.md."
        ),
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    task_config_path = _repo_relative_path(args.task_config)
    labels_by_task = _load_labels(_repo_relative_path(args.classifications_jsonl))

    # tasks list
    cfg = json.loads(task_config_path.read_text(encoding="utf-8"))
    if "tasks" in cfg and isinstance(cfg["tasks"], dict):
        cfg_tasks = sorted(cfg["tasks"].keys())
    else:
        cfg_tasks = sorted(cfg.keys())
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else cfg_tasks
    exclude_tasks = _parse_csv_set(args.exclude_tasks)
    if exclude_tasks:
        tasks = [t for t in tasks if t.strip().lower() not in exclude_tasks]
    agents = [_canon_agent_name(a) for a in args.agents.split(",") if a.strip()]

    # LLM client + prompts
    client, _, _ = initialize_clients(args.api_provider)
    trace_prompt = (Path(__file__).parent / "prompts" / "trace_experience.md").read_text(encoding="utf-8")
    meta_prompt = (Path(__file__).parent / "prompts" / "meta_experience.md").read_text(encoding="utf-8")

    # Output layout: if user writes directly under .../aoa_mode/experience/,
    # automatically create a timestamped run directory and put everything inside it.
    out_path = _repo_relative_path(args.out)
    log_dir_path = _repo_relative_path(args.log_dir)
    out_path, log_dir_path, out_run_dir = _maybe_timestamp_run_dir(out_path, log_dir_path)
    os.makedirs(str(log_dir_path), exist_ok=True)
    if out_run_dir is not None:
        print(f"[AOA][trace_experience] Using run_dir: {out_run_dir.resolve()}")
    log_dir = str(log_dir_path)

    # Incremental outputs (ACE-like playbook accumulation)
    # - bullets.jsonl: one extracted bullet per line
    # - sampled_records.jsonl: trace metadata per bullet
    # - progress.json: lightweight progress snapshots
    ckpt_dir = out_run_dir if out_run_dir is not None else out_path.parent
    bullets_jsonl = ckpt_dir / "bullets.jsonl"
    records_jsonl = ckpt_dir / "sampled_records.jsonl"
    progress_json = ckpt_dir / "progress.json"

    bullets: List[Dict[str, Any]] = []
    sampled_records: List[Dict[str, Any]] = []

    # Build jobs first, then execute in parallel (avoids nested blocking LLM calls).
    jobs: List[Dict[str, Any]] = []
    processed: set[str] = set()
    if args.resume:
        # Prefer sampled_records.jsonl as ground truth of finished jobs.
        for rec in _load_jsonl(records_jsonl):
            t = str(rec.get("task") or "")
            a = _canon_agent_name(str(rec.get("agent") or ""))
            i = rec.get("index")
            if t and a and i is not None:
                try:
                    processed.add(_job_key(t, a, int(i)))
                except Exception:
                    continue
        if processed:
            print(f"[AOA][trace_experience] Resume: loaded {len(processed)} processed jobs from {records_jsonl}")
        # Load existing bullets/records into memory for meta synthesis and final output.
        bullets = _load_jsonl(bullets_jsonl)
        sampled_records = _load_jsonl(records_jsonl)

    if args.meta_only:
        # Load bullets/records from disk and skip all trace extraction.
        if not bullets:
            bullets = _load_jsonl(bullets_jsonl)
        if not sampled_records:
            sampled_records = _load_jsonl(records_jsonl)
        if not bullets:
            raise FileNotFoundError(
                f"--meta_only 需要已存在的 bullets.jsonl，但未找到或为空: {bullets_jsonl}. "
                f"请先跑一次完整 experience_from_traces，或使用 --resume 指向已有 run_dir。"
            )
        print(f"[AOA][trace_experience] meta_only: loaded bullets={len(bullets)} from {bullets_jsonl}")
    else:
        # For each task/agent, sample and extract bullets
        for task in tasks:
            samples = _load_task_samples(task_config_path, task)
            if not samples:
                continue
            for agent in agents:
                ts = args.timestamp.strip() or _latest_timestamp(results_root, task, agent, args.mode)
                if not ts:
                    continue
                src_run_dir = _discover_run_dir(results_root, task, agent, args.mode, ts)
                if src_run_dir is None:
                    continue
                tr_path = src_run_dir / "test_results.json"
                if not tr_path.exists():
                    continue
                _, total, errors = _load_test_results(str(tr_path))
                # Align total to sample length if result file is weird
                total = len(samples) if len(samples) else total
                err_set, _ = _build_error_index_set(errors)
                label_map = labels_by_task.get(task, {})

                picked = _sample_indices(
                    task,
                    agent,
                    total=total,
                    error_idx_set=err_set,
                    labels=label_map,
                    k=args.k_per_task_agent,
                    seed=args.seed,
                )
                detailed_dir = src_run_dir / "detailed_llm_logs"

                for i in picked:
                    if args.resume and (_job_key(task, agent, int(i)) in processed):
                        continue
                    jobs.append(
                        {
                            "task": task,
                            "agent": agent,
                            "run_timestamp": ts,
                            "index": i,
                            "samples": samples,
                            "label_map": label_map,
                            "err_set": err_set,
                            "errors": errors,
                            "detailed_dir": detailed_dir,
                        }
                    )

    def run_one_job(job: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        task = job["task"]
        agent = job["agent"]
        ts = job["run_timestamp"]
        i = int(job["index"])
        samples = job["samples"]
        label_map = job["label_map"]
        err_set = job["err_set"]
        errors = job["errors"]
        detailed_dir: Path = job["detailed_dir"]

        sample = samples[i] if i < len(samples) else {}
        lab = label_map.get(i)
        cap = lab.capability if lab and lab.capability else "UNKNOWN"
        dscore = lab.difficulty_score if lab else None
        dbucket = _difficulty_bucket(dscore)
        is_correct = i not in err_set

        call_id = f"test_eval_{i}"
        trace_path = _find_trace_log(detailed_dir, call_id)
        model_response = ""
        target = sample.get("target")

        if trace_path and trace_path.exists():
            try:
                j = json.loads(trace_path.read_text(encoding="utf-8"))
                model_response = str(j.get("response") or "")
            except Exception:
                model_response = ""

        if not model_response and not is_correct:
            for e in errors:
                idx = e.get("global_index", e.get("index"))
                if idx == i:
                    model_response = str(e.get("prediction") or "")
                    break

        try:
            parsed = json.loads(model_response)
            final_answer = str(parsed.get("final_answer") or "")
        except Exception:
            final_answer = (model_response or "").strip()

        payload = {
            "task_name": task,
            "agent_method": agent,
            "capability": cap,
            "difficulty_score": dscore,
            "difficulty_bucket": dbucket,
            "is_correct": bool(is_correct),
            "question": sample.get("question", ""),
            "context": sample.get("context", ""),
            "model_response": model_response,
            "final_answer": final_answer,
            "target": target,
        }
        obj = _call_llm_json(
            client=client,
            api_provider=args.api_provider,
            model=args.model,
            system_prompt=trace_prompt,
            payload=payload,
            call_id=f"trace_{task}_{agent}_{i}",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            log_dir=log_dir,
        )
        if not obj:
            return None, None
        rec = {
            "task": task,
            "agent": agent,
            "run_timestamp": ts,
            "index": i,
            "capability": cap,
            "difficulty_bucket": dbucket,
            "is_correct": bool(is_correct),
            "trace_log": str(trace_path) if trace_path else "",
        }
        return obj, rec

    if jobs:
        workers = max(1, int(args.workers))
        print(f"[AOA][trace_experience] Extracting {len(jobs)} bullets with {workers} workers...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_one_job, j) for j in jobs]
            completed = 0
            for fut in as_completed(futs):
                obj, rec = fut.result()
                if obj:
                    bullets.append(obj)
                    _append_jsonl(bullets_jsonl, obj)
                if rec:
                    sampled_records.append(rec)
                    _append_jsonl(records_jsonl, rec)
                completed += 1
                if args.checkpoint_every and int(args.checkpoint_every) > 0:
                    if completed % int(args.checkpoint_every) == 0:
                        snap = {
                            "updated_at": datetime.now().isoformat(),
                            "completed_futures": completed,
                            "new_jobs_this_run": len(jobs),
                            "total_bullets_in_memory": len(bullets),
                            "total_records_in_memory": len(sampled_records),
                            "bullets_jsonl": str(bullets_jsonl),
                            "records_jsonl": str(records_jsonl),
                        }
                        progress_json.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"[AOA][trace_experience] checkpoint: {completed}/{len(jobs)} futures done; bullets={len(bullets)}")

    # Meta synthesis
    # Keep meta input bounded, but avoid order bias via stratified/shuffle sampling.
    meta_in = _sample_meta_bullets(
        bullets,
        n=max(0, int(args.meta_max_bullets)),
        seed=int(args.seed),
        strategy=str(args.meta_sampling),
    )
    if args.meta_compact:
        meta_in = _compact_bullets_for_meta(
            meta_in,
            bullet_max_chars=int(args.meta_bullet_max_chars),
            diagnosis_max_chars=int(args.meta_diagnosis_max_chars),
            keep_routing_hint=bool(args.meta_keep_routing_hint),
            keep_confidence=bool(args.meta_keep_confidence),
        )
    try:
        from collections import Counter

        c_agent = Counter()
        c_cap = Counter()
        c_diff = Counter()
        c_out = Counter()
        for b in meta_in:
            tags = _bullet_tags(b)
            c_agent[_safe_str(tags.get("agent_method") or tags.get("agent"))] += 1
            c_cap[_safe_str(tags.get("capability"))] += 1
            c_diff[_safe_str(tags.get("difficulty_bucket"))] += 1
            c_out[_safe_str(b.get("outcome"))] += 1
        if c_agent:
            agent_top = ", ".join([f"{k}:{v}" for k, v in c_agent.most_common(10)])
            cap_top = ", ".join([f"{k}:{v}" for k, v in c_cap.most_common(10)])
            diff_top = ", ".join([f"{k}:{v}" for k, v in c_diff.most_common(10)])
            out_top = ", ".join([f"{k}:{v}" for k, v in c_out.most_common(10)])
            print(
                f"[AOA][meta_synth] meta_sampling={args.meta_sampling} meta_in={len(meta_in)} "
                f"agent_mix(top10)={agent_top} | cap_mix={cap_top} | diff_mix={diff_top} | outcome_mix={out_top}"
            )
    except Exception:
        pass
    meta_payload = {
        "bullets": meta_in,
        "note": "Synthesize routing playbook based on these experience bullets. Prefer robust rules.",
        "requested_rules": {
            "target": int(args.meta_target_rules),
            "min": int(args.meta_min_rules),
            "max": int(args.meta_max_rules),
        },
    }
    meta_obj = _call_llm_json(
        client=client,
        api_provider=args.api_provider,
        model=args.model,
        system_prompt=meta_prompt,
        payload=meta_payload,
        call_id="meta_synth",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_dir=log_dir,
    )

    out = {
        "meta": {
            "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "source": "trace_based",
            "results_root": str(results_root),
            "mode": args.mode,
            "tasks": tasks,
            "agents": agents,
            "k_per_task_agent": args.k_per_task_agent,
            "seed": args.seed,
            "classifications_jsonl": str(_repo_relative_path(args.classifications_jsonl)),
            "checkpoints": {
                "bullets_jsonl": str(bullets_jsonl),
                "sampled_records_jsonl": str(records_jsonl),
                "progress_json": str(progress_json),
                "resume": bool(args.resume),
                "checkpoint_every": int(args.checkpoint_every),
            },
        },
        "sampled_records": sampled_records,
        "experience_bullets": bullets,
        "meta_experience": meta_obj,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[AOA][trace_experience] Wrote: {out_path.resolve()} (bullets={len(bullets)})")


if __name__ == "__main__":
    main()


