#!/usr/bin/env python3
"""
Materialize AOA as a "virtual agent" inside capability_eval_mode.

We DO sample-level routing:
  for each (task, index), choose an underlying agent (either rule-based from experience.json,
  or LLM router that reads sample content), then inherit correctness from that agent's per_sample.jsonl.

Then aggregate accuracy by capability and difficulty bucket, and write:
  results/StructuredReasoning_run/capability_eval_mode/aoa/<mode>/<timestamp>/
    - per_sample.jsonl
    - summary.json

This makes AOA show up as a new agent column in capability_eval tables.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.llm import timed_llm_call
from utils.playbook_utils import extract_json_from_text
from utils.tools import initialize_clients


Key = Tuple[str, int]  # (task, index)


def _canon_agent_name(s: str) -> str:
    s = (s or "").strip()
    alias = {
        "self-refine": "self_refine",
        "dc": "dynamic_cheatsheet",
    }
    return alias.get(s, s)


def _canon_bucket(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() in {"na", "none", "null"}:
        return None
    return s


@dataclass(frozen=True)
class SampleInfo:
    task: str
    index: int
    capability: str
    difficulty_score: Optional[float]

def _extract_features(question: str, context: str) -> Dict[str, Any]:
    """
    Cheap, deterministic features for routing.
    """
    import re

    q = question or ""
    c = context or ""
    text = f"{q}\n{c}"
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    has_code = ("```" in text) or ("def " in text) or ("import " in text) or ("<code>" in text.lower())
    has_table = "|" in c and ("\n|" in c or "| " in c)
    table_rows = 0
    if has_table:
        table_rows = sum(1 for line in c.splitlines() if "|" in line)
    return {
        "question_chars": len(q),
        "context_chars": len(c),
        "num_numbers": len(nums),
        "has_code": bool(has_code),
        "has_table": bool(has_table),
        "table_row_estimate": table_rows,
    }

def _get_routing_policy_from_experience(exp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support two formats:
    - table-based: top-level {"routing_policy": {...}}
    - trace-based: {"meta_experience": {"routing_policy": {...}}}
    """
    if isinstance(exp.get("routing_policy"), dict):
        return exp["routing_policy"]  # type: ignore[return-value]
    if isinstance(exp.get("meta_experience"), dict) and isinstance(exp["meta_experience"].get("routing_policy"), dict):
        return exp["meta_experience"]["routing_policy"]  # type: ignore[return-value]
    return {}

def _get_meta_findings_from_experience(exp: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(exp.get("meta_findings"), list):
        return [x for x in exp.get("meta_findings") if isinstance(x, dict)]  # type: ignore[list-item]
    if isinstance(exp.get("meta_experience"), dict) and isinstance(exp["meta_experience"].get("meta_findings"), list):
        return [x for x in exp["meta_experience"].get("meta_findings") if isinstance(x, dict)]  # type: ignore[list-item]
    return []

def _compact_text(s: Any, max_chars: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."

def _retrieve_bullets_for_sample(
    exp: Dict[str, Any],
    *,
    task_name: str,
    capability: str,
    difficulty_bucket: Optional[str],
    max_bullets: int,
    bullet_max_chars: int,
    diagnosis_max_chars: int,
) -> List[Dict[str, Any]]:
    """
    Lightweight retrieval: score bullets by tag overlap with (task, capability, difficulty_bucket).
    Returns compacted bullets for router consumption.
    """
    if max_bullets <= 0:
        return []
    bullets = exp.get("experience_bullets")
    if not isinstance(bullets, list):
        return []

    def score(b: Dict[str, Any]) -> int:
        tags = b.get("tags") or {}
        if not isinstance(tags, dict):
            tags = {}
        s = 0
        if str(tags.get("task_name") or "").strip() == str(task_name or "").strip():
            s += 3
        if str(tags.get("capability") or "").strip() == str(capability or "").strip():
            s += 2
        db = str(tags.get("difficulty_bucket") or "").strip()
        if difficulty_bucket is not None and db == str(difficulty_bucket):
            s += 1
        return s

    scored: List[tuple[int, int, Dict[str, Any]]] = []
    for i, b in enumerate(bullets):
        if not isinstance(b, dict):
            continue
        scored.append((score(b), i, b))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    out: List[Dict[str, Any]] = []
    for s, _i, b in scored:
        if len(out) >= max_bullets:
            break
        tags = b.get("tags") or {}
        if not isinstance(tags, dict):
            tags = {}
        out.append(
            {
                "score": int(s),
                "bullet": _compact_text(b.get("bullet"), bullet_max_chars),
                "diagnosis": _compact_text(b.get("diagnosis"), diagnosis_max_chars),
                "outcome": b.get("outcome"),
                "tags": {
                    "agent_method": tags.get("agent_method"),
                    "task_name": tags.get("task_name"),
                    "capability": tags.get("capability"),
                    "difficulty_bucket": tags.get("difficulty_bucket"),
                },
            }
        )
    return out

def _compact_experience_for_router(
    exp: Dict[str, Any],
    *,
    task_name: str,
    capability: str,
    difficulty_bucket: Optional[str],
    max_bullets: int,
    bullet_max_chars: int,
    diagnosis_max_chars: int,
) -> Dict[str, Any]:
    """
    Return a smaller experience_json object for router LLM:
    - routing_policy
    - meta_findings
    - retrieved_bullets (top-K compact)
    """
    policy = _get_routing_policy_from_experience(exp)
    findings = _get_meta_findings_from_experience(exp)
    retrieved = _retrieve_bullets_for_sample(
        exp,
        task_name=task_name,
        capability=capability,
        difficulty_bucket=difficulty_bucket,
        max_bullets=max_bullets,
        bullet_max_chars=bullet_max_chars,
        diagnosis_max_chars=diagnosis_max_chars,
    )
    out: Dict[str, Any] = {
        "routing_policy": policy,
        "meta_findings": findings,
    }
    if retrieved:
        out["retrieved_bullets"] = retrieved
    return out


def _difficulty_bucket_for_eval(capability: str, score: Optional[float]) -> Optional[str]:
    """
    Bucketing used for capability_eval tables.
    - IE: None (not bucketed; exporter can repeat overall)
    - CR: None (not bucketed; report overall only, consistent with utils/capability_eval.py)
    - NC/DK: easy/middle/hard by score
    """
    cap = (capability or "").strip()
    if cap == "Information Extraction":
        return None
    if cap == "Complex Reasoning":
        return None
    if score is None:
        return None
    # Numerical Calculation / Domain Knowledge
    if score <= 3.0:
        return "easy"
    if score <= 6.0:
        return "middle"
    return "hard"


def _read_per_sample(path: Path) -> Dict[Key, Tuple[SampleInfo, bool, Optional[str]]]:
    """
    Returns:
      key -> (SampleInfo, correct, difficulty_bucket_from_file)
    """
    out: Dict[Key, Tuple[SampleInfo, bool, Optional[str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task = str(obj.get("task") or "")
            idx = int(obj.get("index"))
            cap = str(obj.get("capability") or "")
            score_raw = obj.get("difficulty_score", None)
            score = None
            if score_raw is not None:
                try:
                    score = float(score_raw)
                except Exception:
                    score = None
            correct = bool(obj.get("correct"))
            diff_bucket = _canon_bucket(obj.get("difficulty_bucket", None))
            key = (task, idx)
            out[key] = (SampleInfo(task=task, index=idx, capability=cap, difficulty_score=score), correct, diff_bucket)
    return out

def _load_task_config(config_path: Path) -> Dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_repo_relative(p: str) -> Path:
    """
    StructuredReasoning configs commonly use "./StructuredReasoning/data/xxx.jsonl" relative to repo root.
    """
    repo_root = Path(__file__).resolve().parents[2]
    p = str(p)
    if p.startswith("./"):
        return (repo_root / Path(p[2:])).resolve()
    return Path(p).resolve()


def _load_test_samples_for_tasks(task_config_path: Path, tasks: Sequence[str]) -> Dict[str, List[dict]]:
    cfg = _load_task_config(task_config_path)
    # Allow wrapper format produced by AOA scripts: {"_meta":..., "tasks": {...}}
    if "tasks" in cfg and isinstance(cfg["tasks"], dict):
        cfg = cfg["tasks"]
    out: Dict[str, List[dict]] = {}
    for task in tasks:
        if task not in cfg:
            continue
        test_p = cfg[task].get("test_data")
        if not test_p:
            continue
        test_path = _resolve_repo_relative(str(test_p))
        rows: List[dict] = []
        with test_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        out[task] = rows
    return out


def _align_keys(per_agent: Dict[str, Dict[Key, Tuple[SampleInfo, bool, Optional[str]]]]) -> List[Key]:
    keys = None
    for m in per_agent.values():
        k = set(m.keys())
        keys = k if keys is None else keys.intersection(k)
    return sorted(keys) if keys else []


def _latest_common_timestamp(cap_eval_root: Path, agents: Sequence[str], mode: str) -> str:
    candidates_by_agent: List[set[str]] = []
    for a in agents:
        base = cap_eval_root / a / mode
        if not base.exists():
            raise FileNotFoundError(f"cap_eval_root 下不存在: {base}")
        ts = {p.name for p in base.iterdir() if p.is_dir()}
        candidates_by_agent.append(ts)
    common = set.intersection(*candidates_by_agent) if candidates_by_agent else set()
    if not common:
        raise ValueError(f"找不到所有 agents 的公共 timestamp: agents={agents}, mode={mode}, root={cap_eval_root}")
    return sorted(common, reverse=True)[0]


def _rule_specificity(when: Dict[str, Any]) -> int:
    score = 0
    for k in ("task_name", "capability", "difficulty_bucket"):
        v = when.get(k, None)
        if v is None:
            continue
        if isinstance(v, str) and v.strip().upper() == "ALL":
            continue
        if isinstance(v, str) and not v.strip():
            continue
        score += 1
    # If feature_conditions is provided and non-empty, treat as more specific.
    fc = when.get("feature_conditions", None)
    if isinstance(fc, dict) and any(v is not None for v in fc.values()):
        score += 1
    return score


def _matches(when: Dict[str, Any], info: SampleInfo, bucket: Optional[str]) -> bool:
    task = when.get("task_name", None)
    if isinstance(task, str) and task.strip() and task.strip().upper() != "ALL":
        if task.strip() != info.task:
            return False
    cap = when.get("capability", None)
    if isinstance(cap, str) and cap.strip() and cap.strip().upper() != "ALL":
        if cap.strip() != info.capability:
            return False
    diff = when.get("difficulty_bucket", None)
    if isinstance(diff, str) and diff.strip():
        if diff.strip().upper() == "ALL":
            return True
        if _canon_bucket(diff) != bucket:
            return False
    return True


def _feature_match(feature_conditions: Any, features: Optional[Dict[str, Any]]) -> bool:
    """
    feature_conditions: JSON dict (see prompts/meta_experience.md).
    If feature_conditions is not a dict / empty: no constraint.
    If dict has any constraint but features is None: cannot match.
    """
    if not feature_conditions or not isinstance(feature_conditions, dict):
        return True
    if not any(v is not None for v in feature_conditions.values()):
        return True
    if not features:
        return False

    def _want_bool(v: Any) -> Optional[bool]:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"", "omit", "none", "null"}:
                return None
            if s in {"true", "yes", "1"}:
                return True
            if s in {"false", "no", "0"}:
                return False
        return None

    def _want_num(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"", "omit", "none", "null"}:
                return None
            try:
                return float(s)
            except Exception:
                return None
        return None

    # boolean constraints
    for bk in ("has_table", "has_code"):
        if bk in feature_conditions:
            want = _want_bool(feature_conditions.get(bk))
            if want is None:
                continue
            if bool(features.get(bk)) != want:
                return False

    # numeric range constraints
    ranges = [
        ("num_numbers", "num_numbers_min", "num_numbers_max"),
        ("context_chars", "context_chars_min", "context_chars_max"),
        ("question_chars", "question_chars_min", "question_chars_max"),
        ("table_row_estimate", "table_row_estimate_min", "table_row_estimate_max"),
    ]
    for base, kmin, kmax in ranges:
        if (kmin not in feature_conditions) and (kmax not in feature_conditions):
            continue
        try:
            fv = float(features.get(base, 0.0))
        except Exception:
            continue
        mn = _want_num(feature_conditions.get(kmin))
        mx = _want_num(feature_conditions.get(kmax))
        if mn is not None and fv < mn:
            return False
        if mx is not None and fv > mx:
            return False

    return True


def _select_agent(
    info: SampleInfo,
    bucket: Optional[str],
    policy: Dict[str, Any],
    agents_available: set[str],
    low_conf_fallback: bool = True,
    features: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    default_agent = _canon_agent_name(str(policy.get("default_agent") or "")) or "ace"
    rules = policy.get("rules", []) or []

    applicable: List[Tuple[int, int, Dict[str, Any]]] = []
    for i, r in enumerate(rules):
        when = r.get("when", {}) or {}
        if not isinstance(when, dict):
            continue
        if not _matches(when, info, bucket):
            continue
        # optional feature constraints
        fc = when.get("feature_conditions", None)
        if not _feature_match(fc, features):
            continue
        applicable.append((_rule_specificity(when), i, r))

    if not applicable:
        return default_agent, "default(no_match)"

    applicable.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    best = applicable[0][2]
    choose = _canon_agent_name(str(best.get("choose") or default_agent))
    conf = str(best.get("confidence") or "").lower()

    if low_conf_fallback and conf == "low":
        return default_agent, f"default(low_confidence_match:{best.get('when')})"
    if choose not in agents_available:
        return default_agent, f"default(agent_missing:{best.get('choose')})"
    return choose, f"rule_match:{best.get('when')}"

def _select_agent_llm(
    *,
    router_client,
    api_provider: str,
    router_model: str,
    router_prompt: str,
    router_temperature: float,
    router_max_tokens: int,
    experience_json: Dict[str, Any],
    candidates: List[str],
    task: str,
    index: int,
    sample: Dict[str, Any],
    capability: str,
    difficulty_bucket: Optional[str],
    features: Dict[str, Any],
    log_dir: Optional[str],
    compact_experience: bool = False,
    experience_max_bullets: int = 0,
    experience_bullet_max_chars: int = 160,
    experience_diagnosis_max_chars: int = 80,
) -> Tuple[str, Dict[str, Any]]:
    """
    Ask an LLM to choose the best agent. This call is only for *routing*.
    Underlying agent outputs are NOT recomputed; we reuse existing per-sample correctness.
    """
    question = sample.get("question", "")
    context = sample.get("context", "") or ""
    exp_for_router = experience_json
    if compact_experience:
        exp_for_router = _compact_experience_for_router(
            experience_json,
            task_name=task,
            capability=capability,
            difficulty_bucket=difficulty_bucket,
            max_bullets=max(0, int(experience_max_bullets)),
            bullet_max_chars=int(experience_bullet_max_chars),
            diagnosis_max_chars=int(experience_diagnosis_max_chars),
        )
    payload = {
        "experience_json": exp_for_router,
        "sample": {
            "task_name": task,
            "index": index,
            "capability": capability,
            "difficulty_bucket": difficulty_bucket or "NA",
            "features": features,
            "question": question,
            "context": context,
        },
        "candidates": candidates,
    }
    user_input = json.dumps(payload, ensure_ascii=False)
    raw, info = timed_llm_call(
        client=router_client,
        api_provider=api_provider,
        model=router_model,
        prompt=user_input,
        role="aoa_router",
        call_id=f"aoa_route_{task}_{index}",
        max_tokens=router_max_tokens,
        log_dir=log_dir,
        use_json_mode=True,
        temperature=router_temperature,
        messages=[
            {"role": "system", "content": router_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    obj = extract_json_from_text(raw) or {}
    chosen = _canon_agent_name(str(obj.get("chosen_agent") or obj.get("agent") or obj.get("choice") or ""))
    if not chosen:
        chosen = _canon_agent_name(str(experience_json.get("routing_policy", {}).get("default_agent", "ace")))
    return chosen, {"router_output": obj, "router_call_info": info, "router_raw": raw}


def _acc(c: int, t: int) -> float:
    return (c / t) if t else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experience_json",
        type=str,
        required=True,
        help="Path to experience.json generated by experience_extractor.",
    )
    ap.add_argument(
        "--cap_eval_root",
        type=str,
        default="results/StructuredReasoning_run/capability_eval_mode",
        help="Root dir that contains <agent>/<mode>/<timestamp>/per_sample.jsonl",
    )
    ap.add_argument("--mode", type=str, default="online", choices=["online", "offline", "eval_only"])
    ap.add_argument(
        "--agents",
        type=str,
        default="cot,self_refine,reflexion,debate,discussion,dynamic_cheatsheet,gepa,ace,amem",
        help="Comma-separated agent folder names under cap_eval_root.",
    )
    ap.add_argument("--timestamp", type=str, default="", help="If empty, auto-pick latest common timestamp.")
    ap.add_argument(
        "--out_cap_eval_root",
        type=str,
        default="results/StructuredReasoning_run/capability_eval_mode",
        help="Write AOA outputs under this root as <out_cap_eval_root>/aoa/<mode>/<timestamp>/",
    )
    ap.add_argument("--aoa_agent_dir", type=str, default="aoa", help="Folder name for the virtual agent.")
    ap.add_argument(
        "--no_low_conf_fallback",
        action="store_true",
        help="If set, do NOT fallback to default_agent when matched rule confidence=low.",
    )
    ap.add_argument(
        "--override_default_agent",
        type=str,
        default="",
        help="If set, override routing_policy.default_agent at runtime (for ablation).",
    )

    # LLM router mode (true agent-of-agents routing, but still reusing existing agent results)
    ap.add_argument(
        "--router_llm",
        action="store_true",
        help="If set, call an LLM router for EACH sample to choose agent (reuses existing agent outputs).",
    )
    ap.add_argument(
        "--router_experience_compact",
        action="store_true",
        help="If set, send a compact experience_json to router (routing_policy + meta_findings + retrieved_bullets).",
    )
    ap.add_argument(
        "--router_experience_max_bullets",
        type=int,
        default=120,
        help="When --router_experience_compact, retrieve at most K bullets for router.",
    )
    ap.add_argument(
        "--router_experience_bullet_max_chars",
        type=int,
        default=160,
        help="When --router_experience_compact, truncate each bullet text to this many chars.",
    )
    ap.add_argument(
        "--router_experience_diagnosis_max_chars",
        type=int,
        default=80,
        help="When --router_experience_compact, truncate each diagnosis to this many chars.",
    )
    ap.add_argument(
        "--task_config",
        type=str,
        default="StructuredReasoning/data/task_config.json",
        help="Task config to load sample question/context for LLM routing.",
    )
    ap.add_argument("--router_api_provider", type=str, default="usd_guiji")
    ap.add_argument("--router_model", type=str, default="USD-guiji/deepseek-v3")
    ap.add_argument("--router_temperature", type=float, default=0.0)
    ap.add_argument("--router_max_tokens", type=int, default=512)
    ap.add_argument(
        "--router_workers",
        type=int,
        default=8,
        help="LLM router parallelism (ThreadPoolExecutor). Use small numbers to avoid rate limits.",
    )
    ap.add_argument(
        "--router_prompt_path",
        type=str,
        default="Agents/aoa/prompts/router.md",
        help="Router system prompt path.",
    )
    ap.add_argument(
        "--router_cache_jsonl",
        type=str,
        default="",
        help="Optional cache jsonl to resume router decisions (stored under AOA output dir if empty).",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap on number of aligned samples to route (0 means all).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    exp_path = Path(args.experience_json)
    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    if not isinstance(exp, dict):
        raise ValueError("experience_json 格式不正确（应为 JSON object）。")
    # Support two formats:
    # 1) table-based extractor: top-level {"routing_policy": {...}, ...}
    # 2) trace-based extractor: {"meta_experience": {"routing_policy": {...}, ...}, ...}
    if isinstance(exp.get("routing_policy"), dict):
        policy = exp["routing_policy"]
    elif isinstance(exp.get("meta_experience"), dict) and isinstance(exp["meta_experience"].get("routing_policy"), dict):
        policy = exp["meta_experience"]["routing_policy"]
    else:
        raise ValueError("experience_json 缺少 routing_policy（顶层或 meta_experience.routing_policy），或格式不正确。")
    if args.override_default_agent:
        policy = dict(policy)
        policy["default_agent"] = args.override_default_agent.strip()

    cap_eval_root = Path(args.cap_eval_root)
    mode = args.mode
    agents = [_canon_agent_name(x) for x in args.agents.split(",") if x.strip()]
    if len(agents) < 2:
        raise ValueError("AOA 至少需要 2 个 agents。")

    timestamp = args.timestamp.strip() or _latest_common_timestamp(cap_eval_root, agents, mode)

    per_agent: Dict[str, Dict[Key, Tuple[SampleInfo, bool, Optional[str]]]] = {}
    for a in agents:
        p = cap_eval_root / a / mode / timestamp / "per_sample.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"缺少 per_sample.jsonl: {p}")
        per_agent[a] = _read_per_sample(p)

    keys = _align_keys(per_agent)
    if not keys:
        raise ValueError("无法对齐 keys：可能各 agent 的 per_sample 覆盖不一致。")
    if args.max_samples and args.max_samples > 0:
        keys = keys[: int(args.max_samples)]

    # Build convenience maps
    key_to_info: Dict[Key, SampleInfo] = {k: per_agent[agents[0]][k][0] for k in keys}
    key_to_bucket: Dict[Key, Optional[str]] = {}
    for k in keys:
        info, _correct, bucket_from_file = per_agent[agents[0]][k]
        key_to_bucket[k] = bucket_from_file if bucket_from_file is not None else _difficulty_bucket_for_eval(
            info.capability, info.difficulty_score
        )

    per_agent_correct: Dict[str, Dict[Key, bool]] = {a: {k: per_agent[a][k][1] for k in keys} for a in agents}
    agents_available = set(agents)

    # Keep outputs separated under the same aligned timestamp to avoid mixing with other AOA products.
    out_root = Path(args.out_cap_eval_root) / args.aoa_agent_dir / mode / timestamp / "as_agent_capability_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    # Detect whether rule-based routing needs sample-level features (feature_conditions)
    need_feature_rules = False
    try:
        for r in (policy.get("rules") or []):
            when = (r or {}).get("when", {}) or {}
            fc = when.get("feature_conditions", None) if isinstance(when, dict) else None
            if isinstance(fc, dict) and any(v is not None for v in fc.values()):
                need_feature_rules = True
                break
    except Exception:
        need_feature_rules = False

    # LLM router setup (optional)
    router_client = None
    router_prompt = ""
    samples_by_task: Dict[str, List[dict]] = {}
    router_cache_path = ""
    router_cache: Dict[str, Dict[str, Any]] = {}
    cache_lock: Optional[Lock] = None
    cache_f = None
    if args.router_llm:
        router_client, _, _ = initialize_clients(args.router_api_provider)
        router_prompt = Path(args.router_prompt_path).read_text(encoding="utf-8")
        samples_by_task = _load_test_samples_for_tasks(Path(args.task_config), sorted({t for (t, _) in keys}))
        router_cache_path = args.router_cache_jsonl or str(out_root / "router_cache.jsonl")
        os.makedirs(str(out_root / "router_llm_logs"), exist_ok=True)
        # load cache if exists
        if router_cache_path and Path(router_cache_path).exists():
            with Path(router_cache_path).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    k = obj.get("key")
                    if isinstance(k, str):
                        router_cache[k] = obj
        cache_lock = Lock()
        # open cache file for incremental appends (thread-safe via lock)
        cache_f = open(router_cache_path, "a", encoding="utf-8") if router_cache_path else None
    elif need_feature_rules:
        # Rule-based routing with feature_conditions: we need sample text to compute features.
        # This is cheap and deterministic (no LLM calls).
        try:
            samples_by_task = _load_test_samples_for_tasks(Path(args.task_config), sorted({t for (t, _) in keys}))
        except Exception:
            samples_by_task = {}

    selected = Counter()
    used_rule = Counter()
    correct_total = 0

    # Aggregates
    by_cap: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_cap_bucket: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "correct": 0})
    )

    # Write per-sample
    per_sample_path = out_root / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as wf:
        # Pre-compute router decisions (possibly parallel) to keep final writing deterministic.
        router_decisions: Dict[str, Tuple[str, str, Dict[str, Any]]] = {}

        def route_one_key(k: Key) -> Tuple[str, str, str, Dict[str, Any]]:
            """
            Returns: (cache_key, chosen_agent, rule_desc, router_meta)
            """
            info = key_to_info[k]
            bucket = key_to_bucket[k]
            cache_key = f"{info.task}::{info.index}"

            if not args.router_llm:
                feats = None
                if need_feature_rules:
                    task_samples = samples_by_task.get(info.task) or []
                    if 0 <= info.index < len(task_samples):
                        s = task_samples[info.index]
                        feats = _extract_features(s.get("question", ""), s.get("context", "") or "")
                chosen, rule_desc = _select_agent(
                    info=info,
                    bucket=bucket,
                    policy=policy,
                    agents_available=agents_available,
                    low_conf_fallback=not args.no_low_conf_fallback,
                    features=feats,
                )
                return cache_key, chosen, rule_desc, {"features": feats} if feats is not None else {}

            cached = router_cache.get(cache_key)
            if cached and "chosen_agent" in cached:
                chosen = _canon_agent_name(str(cached.get("chosen_agent") or ""))
                rule_desc = "router_cache"
                router_meta = cached.get("router_meta") or {}
                return cache_key, chosen, rule_desc, router_meta

            task_samples = samples_by_task.get(info.task) or []
            if info.index < 0 or info.index >= len(task_samples):
                chosen, rule_desc = _select_agent(
                    info=info,
                    bucket=bucket,
                    policy=policy,
                    agents_available=agents_available,
                    low_conf_fallback=not args.no_low_conf_fallback,
                )
                return cache_key, chosen, f"{rule_desc}|fallback(no_sample)", {}

            sample = task_samples[info.index]
            feats = _extract_features(sample.get("question", ""), sample.get("context", "") or "")
            chosen, meta = _select_agent_llm(
                router_client=router_client,
                api_provider=args.router_api_provider,
                router_model=args.router_model,
                router_prompt=router_prompt,
                router_temperature=args.router_temperature,
                router_max_tokens=args.router_max_tokens,
                experience_json=exp,
                # IMPORTANT: keep candidates order stable for reproducibility
                candidates=list(agents),
                task=info.task,
                index=info.index,
                sample=sample,
                capability=info.capability,
                difficulty_bucket=bucket,
                features=feats,
                log_dir=str(out_root / "router_llm_logs"),
                compact_experience=bool(args.router_experience_compact),
                experience_max_bullets=int(args.router_experience_max_bullets),
                experience_bullet_max_chars=int(args.router_experience_bullet_max_chars),
                experience_diagnosis_max_chars=int(args.router_experience_diagnosis_max_chars),
            )
            if chosen not in agents_available:
                chosen = _canon_agent_name(str(policy.get("default_agent") or "ace"))

            router_meta = {
                "used_rule": meta.get("router_output", {}).get("used_rule") if isinstance(meta.get("router_output"), dict) else None,
                "confidence": meta.get("router_output", {}).get("confidence") if isinstance(meta.get("router_output"), dict) else None,
            }

            # append cache line (thread-safe)
            if cache_f is not None and cache_lock is not None:
                rec = {"key": cache_key, "chosen_agent": chosen, "router_meta": router_meta}
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                with cache_lock:
                    cache_f.write(line)
                    cache_f.flush()
                    router_cache[cache_key] = rec

            return cache_key, chosen, "router_llm", router_meta

        if args.router_llm:
            # Parallel router calls (dominant cost)
            workers = max(1, int(args.router_workers))
            print(f"[AOA][router_llm] Routing {len(keys)} samples with {workers} workers...")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(route_one_key, k) for k in keys]
                for fut in as_completed(futs):
                    cache_key, chosen, rule_desc, router_meta = fut.result()
                    router_decisions[cache_key] = (chosen, rule_desc, router_meta)
        else:
            # Rule-based routing (fast)
            for k in keys:
                cache_key, chosen, rule_desc, router_meta = route_one_key(k)
                router_decisions[cache_key] = (chosen, rule_desc, router_meta)

        for k in keys:
            info = key_to_info[k]
            bucket = key_to_bucket[k]

            cache_key = f"{info.task}::{info.index}"
            chosen, rule_desc, router_meta = router_decisions.get(cache_key, ("ace", "default(missing_decision)", {}))
            selected[chosen] += 1
            used_rule[rule_desc] += 1
            is_correct = bool(per_agent_correct[chosen].get(k, False))
            correct_total += 1 if is_correct else 0

            by_cap[info.capability]["total"] += 1
            by_cap[info.capability]["correct"] += 1 if is_correct else 0

            if bucket is not None:
                by_cap_bucket[info.capability][bucket]["total"] += 1
                by_cap_bucket[info.capability][bucket]["correct"] += 1 if is_correct else 0

            wf.write(
                json.dumps(
                    {
                        "task": info.task,
                        "agent_method": "aoa",
                        "mode": mode,
                        "index": info.index,
                        "capability": info.capability,
                        "difficulty_score": info.difficulty_score,
                        "difficulty_bucket": bucket,
                        "chosen_agent": chosen,
                        "router": rule_desc,
                        "router_meta": router_meta or None,
                        "correct": bool(is_correct),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if cache_f is not None:
            cache_f.close()

    # Build summary.json compatible with capability_eval expectations
    by_cap_out: Dict[str, Any] = {}
    for cap, s in by_cap.items():
        by_cap_out[cap] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": _acc(s["correct"], s["total"]),
        }

    by_cap_bucket_out: Dict[str, Any] = {}
    for cap, buckets in by_cap_bucket.items():
        by_cap_bucket_out[cap] = {}
        for b, s in buckets.items():
            by_cap_bucket_out[cap][b] = {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": _acc(s["correct"], s["total"]),
            }

    total_all = len(keys)
    summary = {
        "agent_method": "aoa",
        "mode": mode,
        "total_samples": total_all,
        "overall_accuracy": _acc(correct_total, total_all),
        "by_capability": by_cap_out,
        "by_capability_bucket": by_cap_bucket_out,
        # AOA-specific metadata for analysis
        "aoa": {
            "experience_json": str(exp_path),
            "source_cap_eval_root": str(cap_eval_root),
            "base_agents": agents,
            "timestamp_aligned_from": timestamp,
            "output_subdir": "as_agent_capability_eval",
            "low_conf_fallback": (not args.no_low_conf_fallback),
            "router_llm": bool(args.router_llm),
            "router_model": args.router_model if args.router_llm else None,
            "router_api_provider": args.router_api_provider if args.router_llm else None,
            "router_cache_jsonl": router_cache_path if args.router_llm else None,
            "selected_agent_counts": dict(selected),
            "top_used_rule_descriptions": used_rule.most_common(30),
        },
    }

    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[AOA][as_agent] Wrote: {(out_root / 'summary.json').resolve()}")
    print(f"[AOA][as_agent] overall_acc={summary['overall_accuracy']:.4f} (N={total_all})")


if __name__ == "__main__":
    main()


