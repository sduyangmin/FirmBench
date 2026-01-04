#!/usr/bin/env python3
"""
Evaluate an LLM-generated routing policy (experience.json) on existing per-sample results.

Goal:
  用 LLM 总结出来的 routing_policy（来自 experience_extractor）
  在不重跑任何 agent 的前提下，做确定性路由评测，输出可写论文的数字。

Inputs:
  - experience_json: results/.../aoa_mode/experience/experience.latest.json
  - per-sample: results/StructuredReasoning_run/capability_eval_mode/<agent>/<mode>/<timestamp>/per_sample.jsonl

Output:
  - results/.../aoa_mode/rule_eval/<mode>/<timestamp>/rule_eval_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    difficulty_bucket: Optional[str]

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


def _extract_features(question: str, context: str) -> Dict[str, Any]:
    """
    Cheap, deterministic features for routing.
    Keep consistent with as_agent_capability_eval.py.
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


def _read_per_sample(path: Path) -> Dict[Key, Tuple[SampleInfo, bool]]:
    out: Dict[Key, Tuple[SampleInfo, bool]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task = str(obj.get("task") or "")
            idx = int(obj.get("index"))
            cap = str(obj.get("capability") or "")
            diff = _canon_bucket(obj.get("difficulty_bucket", None))
            correct = bool(obj.get("correct"))
            key = (task, idx)
            out[key] = (SampleInfo(task=task, index=idx, capability=cap, difficulty_bucket=diff), correct)
    return out


def _align_keys(per_agent: Dict[str, Dict[Key, Tuple[SampleInfo, bool]]]) -> List[Key]:
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


def _acc(c: int, t: int) -> float:
    return (c / t) if t else 0.0


def _rule_specificity(when: Dict[str, Any]) -> int:
    """
    More specific rules should win.
    We score based on which keys are constrained (not ALL / not empty).
    """
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


def _feature_match(feature_conditions: Any, features: Optional[Dict[str, Any]]) -> bool:
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

    for bk in ("has_table", "has_code"):
        if bk in feature_conditions:
            want = _want_bool(feature_conditions.get(bk))
            if want is None:
                continue
            if bool(features.get(bk)) != want:
                return False

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


def _matches(when: Dict[str, Any], info: SampleInfo) -> bool:
    # task_name optional
    task = when.get("task_name", None)
    if isinstance(task, str) and task.strip() and task.strip().upper() != "ALL":
        if task.strip() != info.task:
            return False

    cap = when.get("capability", None)
    if isinstance(cap, str) and cap.strip() and cap.strip().upper() != "ALL":
        if cap.strip() != info.capability:
            return False

    diff = when.get("difficulty_bucket", None)
    if isinstance(diff, str) and diff.strip() and diff.strip().upper() != "ALL":
        if _canon_bucket(diff) != info.difficulty_bucket:
            return False

    return True


def _select_agent(
    info: SampleInfo,
    policy: Dict[str, Any],
    agents_available: set[str],
    low_conf_fallback: bool = True,
    features: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Returns: (chosen_agent, used_rule_desc)
    """
    default_agent = _canon_agent_name(str(policy.get("default_agent") or ""))
    if not default_agent:
        # 若 experience 没给，保守用 ace
        default_agent = "ace"

    rules = policy.get("rules", []) or []
    applicable: List[Tuple[int, int, Dict[str, Any]]] = []
    for i, r in enumerate(rules):
        when = r.get("when", {}) or {}
        if not isinstance(when, dict):
            continue
        if not _matches(when, info):
            continue
        fc = when.get("feature_conditions", None)
        if not _feature_match(fc, features):
            continue
        applicable.append((_rule_specificity(when), i, r))

    if not applicable:
        chosen = default_agent
        return chosen, "default(no_match)"

    # highest specificity first; stable by original order
    applicable.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    best = applicable[0][2]
    choose = _canon_agent_name(str(best.get("choose") or default_agent))
    conf = str(best.get("confidence") or "").lower()

    if low_conf_fallback and conf == "low":
        choose = default_agent
        return choose, f"default(low_confidence_match:{best.get('when')})"

    # ensure agent exists in per-sample set
    if choose not in agents_available:
        choose = default_agent
        return choose, f"default(agent_missing:{best.get('choose')})"

    return choose, f"rule_match:{best.get('when')}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experience_json",
        type=str,
        required=True,
        help="Path to experience.latest.json generated by experience_extractor.",
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
        "--out_dir",
        type=str,
        default="results/StructuredReasoning_run/capability_eval_mode/aoa",
        help=(
            "Output root dir. For consistency with capability_eval_mode, summary is written to "
            "<out_dir>/<mode>/<timestamp>/rule_eval/rule_eval_summary.json"
        ),
    )
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
    ap.add_argument(
        "--task_config",
        type=str,
        default="StructuredReasoning/data/task_config.json",
        help="Optional: load sample question/context to enable feature_conditions in routing_policy.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    exp_path = Path(args.experience_json)
    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    if not isinstance(exp, dict):
        raise ValueError("experience_json 格式不正确（应为 JSON object）。")
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
        raise ValueError("rule_eval 至少需要 2 个 agents。")

    timestamp = args.timestamp.strip() or _latest_common_timestamp(cap_eval_root, agents, mode)

    per_agent: Dict[str, Dict[Key, Tuple[SampleInfo, bool]]] = {}
    for a in agents:
        p = cap_eval_root / a / mode / timestamp / "per_sample.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"缺少 per_sample.jsonl: {p}")
        per_agent[a] = _read_per_sample(p)

    keys = _align_keys(per_agent)
    if not keys:
        raise ValueError("无法对齐 keys：可能各 agent 的 per_sample 覆盖不一致。")

    key_to_info: Dict[Key, SampleInfo] = {k: per_agent[agents[0]][k][0] for k in keys}
    per_agent_correct: Dict[str, Dict[Key, bool]] = {a: {k: per_agent[a][k][1] for k in keys} for a in agents}

    # Load samples only if routing_policy contains feature_conditions
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

    samples_by_task: Dict[str, List[dict]] = {}
    if need_feature_rules:
        try:
            samples_by_task = _load_test_samples_for_tasks(Path(args.task_config), sorted({t for (t, _) in keys}))
        except Exception:
            samples_by_task = {}

    selected = Counter()
    used_rule = Counter()
    correct = 0

    agents_available = set(agents)
    for k in keys:
        info = key_to_info[k]
        feats = None
        if need_feature_rules:
            task_samples = samples_by_task.get(info.task) or []
            if 0 <= info.index < len(task_samples):
                s = task_samples[info.index]
                feats = _extract_features(s.get("question", ""), s.get("context", "") or "")
        chosen, rule_desc = _select_agent(
            info=info,
            policy=policy,
            agents_available=agents_available,
            low_conf_fallback=not args.no_low_conf_fallback,
            features=feats,
        )
        selected[chosen] += 1
        used_rule[rule_desc] += 1
        correct += 1 if per_agent_correct[chosen].get(k, False) else 0

    routed_acc = _acc(correct, len(keys))
    out_root = Path(args.out_dir) / mode / timestamp / "rule_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "experience_json": str(exp_path),
        "cap_eval_root": str(cap_eval_root),
        "mode": mode,
        "timestamp": timestamp,
        "agents": agents,
        "aligned_samples": len(keys),
        "routing_policy": {
            "default_agent": policy.get("default_agent"),
            "tie_breaker": policy.get("tie_breaker"),
            "min_margin": policy.get("min_margin"),
            "num_rules": len(policy.get("rules") or []),
            "low_conf_fallback": (not args.no_low_conf_fallback),
        },
        "routed": {
            "accuracy": routed_acc,
            "selected_agent_counts": dict(selected),
        },
        "diagnostics": {
            "top_used_rule_descriptions": used_rule.most_common(30),
        },
    }

    (out_root / "rule_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[AOA][rule_eval] Wrote: {(out_root / 'rule_eval_summary.json').resolve()}")
    print(f"[AOA][rule_eval] routed_acc={routed_acc:.4f} (N={len(keys)})")


if __name__ == "__main__":
    main()


