#!/usr/bin/env python3
"""
AOA: Offline routing evaluation (no re-run).

数据来源（由 `utils/capability_eval.py` 生成）：
  results/StructuredReasoning_run/capability_eval_mode/<agent>/<mode>/<timestamp>/per_sample.jsonl

每行示例：
  {"task":"CodeFinQA","index":0,"capability":"Numerical Calculation","difficulty_bucket":"easy","correct":true,...}

我们将不同 agent 的 per-sample 结果按 (task,index) 对齐，评估多种路由策略：
- best_single: 选择单一 agent（整体准确率最高）
- oracle_any: 每个样本只要“存在某个 agent 做对”就计为对（路由上界）
- task_best: 每个 task 选择一个 agent
- cap_best: 每个 capability 选择一个 agent
- capdiff_best: 每个 (capability, difficulty_bucket) 选择一个 agent

注意：这些策略都是在同一批 per-sample 上“拟合并评估”，属于乐观估计；
用于论文时建议明确标注为 oracle / in-sample router，或进一步做 hold-out 评估。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Key = Tuple[str, int]  # (task, index)


def _canon_agent_name(s: str) -> str:
    s = (s or "").strip()
    # 兼容 CSV 展示名
    alias = {
        "self-refine": "self_refine",
        "dc": "dynamic_cheatsheet",
    }
    return alias.get(s, s)


def _latest_common_timestamp(cap_eval_root: Path, agents: Sequence[str], mode: str) -> str:
    """
    找到所有 agent 都存在的最新 timestamp（按字典序倒序，匹配 YYYYMMDD_HHMMSS）。
    """
    candidates_by_agent: List[set[str]] = []
    for a in agents:
        base = cap_eval_root / a / mode
        if not base.exists():
            raise FileNotFoundError(f"cap_eval_root 下不存在: {base}")
        ts = {p.name for p in base.iterdir() if p.is_dir()}
        candidates_by_agent.append(ts)
    common = set.intersection(*candidates_by_agent) if candidates_by_agent else set()
    if not common:
        raise ValueError(
            f"找不到所有 agents 的公共 timestamp: agents={agents}, mode={mode}, root={cap_eval_root}"
        )
    return sorted(common, reverse=True)[0]


@dataclass(frozen=True)
class SampleInfo:
    task: str
    index: int
    capability: str
    difficulty_bucket: Optional[str]


def _read_per_sample(path: Path) -> Dict[Key, Tuple[SampleInfo, bool]]:
    """
    Returns:
      key -> (SampleInfo, correct)
    """
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
            diff = obj.get("difficulty_bucket", None)
            diff = None if diff in ("", "None", None) else str(diff)
            correct = bool(obj.get("correct"))
            key = (task, idx)
            out[key] = (SampleInfo(task=task, index=idx, capability=cap, difficulty_bucket=diff), correct)
    return out


def _align_keys(per_agent: Dict[str, Dict[Key, Tuple[SampleInfo, bool]]]) -> List[Key]:
    keys = None
    for _, m in per_agent.items():
        k = set(m.keys())
        keys = k if keys is None else keys.intersection(k)
    if not keys:
        return []
    return sorted(keys)


def _acc(n_correct: int, n_total: int) -> float:
    return (n_correct / n_total) if n_total else 0.0


def _build_task_best(
    keys: Sequence[Key],
    per_agent_correct: Dict[str, Dict[Key, bool]],
    key_to_info: Dict[Key, SampleInfo],
    agents: Sequence[str],
) -> Dict[str, str]:
    # task -> best agent
    task_keys: Dict[str, List[Key]] = defaultdict(list)
    for k in keys:
        task_keys[key_to_info[k].task].append(k)

    mapping: Dict[str, str] = {}
    for task, tkeys in task_keys.items():
        best_a = None
        best_acc = -1.0
        for a in agents:
            c = sum(1 for k in tkeys if per_agent_correct[a].get(k, False))
            acc = _acc(c, len(tkeys))
            if acc > best_acc:
                best_acc = acc
                best_a = a
        mapping[task] = str(best_a)
    return mapping


def _build_cap_best(
    keys: Sequence[Key],
    per_agent_correct: Dict[str, Dict[Key, bool]],
    key_to_info: Dict[Key, SampleInfo],
    agents: Sequence[str],
) -> Dict[str, str]:
    # capability -> best agent
    cap_keys: Dict[str, List[Key]] = defaultdict(list)
    for k in keys:
        cap_keys[key_to_info[k].capability].append(k)

    mapping: Dict[str, str] = {}
    for cap, ckeys in cap_keys.items():
        best_a = None
        best_acc = -1.0
        for a in agents:
            c = sum(1 for k in ckeys if per_agent_correct[a].get(k, False))
            acc = _acc(c, len(ckeys))
            if acc > best_acc:
                best_acc = acc
                best_a = a
        mapping[cap] = str(best_a)
    return mapping


def _bucket_key(info: SampleInfo) -> Tuple[str, Optional[str]]:
    # IE / CR 在 capability_eval 里可能为 None（不分桶），保持 None 即可
    return (info.capability, info.difficulty_bucket)


def _build_capdiff_best(
    keys: Sequence[Key],
    per_agent_correct: Dict[str, Dict[Key, bool]],
    key_to_info: Dict[Key, SampleInfo],
    agents: Sequence[str],
) -> Dict[str, str]:
    # (cap, diff) -> best agent
    bucket_keys: Dict[Tuple[str, Optional[str]], List[Key]] = defaultdict(list)
    for k in keys:
        bucket_keys[_bucket_key(key_to_info[k])].append(k)

    mapping: Dict[str, str] = {}
    for (cap, diff), bkeys in bucket_keys.items():
        best_a = None
        best_acc = -1.0
        for a in agents:
            c = sum(1 for k in bkeys if per_agent_correct[a].get(k, False))
            acc = _acc(c, len(bkeys))
            if acc > best_acc:
                best_acc = acc
                best_a = a
        mapping[f"{cap}::{diff or 'NA'}"] = str(best_a)
    return mapping


def _eval_policy_select(
    keys: Sequence[Key],
    per_agent_correct: Dict[str, Dict[Key, bool]],
    key_to_info: Dict[Key, SampleInfo],
    policy_name: str,
    mapping: Dict[str, str],
) -> Tuple[float, Counter]:
    """
    Evaluate a routing policy that selects an agent per key based on mapping.
    Returns: (accuracy, selected_agent_counter)
    """
    selected = Counter()
    correct = 0
    for k in keys:
        info = key_to_info[k]
        if policy_name == "task_best":
            a = mapping[info.task]
        elif policy_name == "cap_best":
            a = mapping[info.capability]
        elif policy_name == "capdiff_best":
            mk = f"{info.capability}::{info.difficulty_bucket or 'NA'}"
            a = mapping[mk]
        else:
            raise ValueError(f"未知 policy_name: {policy_name}")
        selected[a] += 1
        correct += 1 if per_agent_correct[a].get(k, False) else 0
    return _acc(correct, len(keys)), selected


def _best_single(keys: Sequence[Key], per_agent_correct: Dict[str, Dict[Key, bool]], agents: Sequence[str]) -> Tuple[str, float]:
    best_a = ""
    best_acc = -1.0
    for a in agents:
        c = sum(1 for k in keys if per_agent_correct[a].get(k, False))
        acc = _acc(c, len(keys))
        if acc > best_acc:
            best_acc = acc
            best_a = a
    return best_a, best_acc


def _oracle_any(keys: Sequence[Key], per_agent_correct: Dict[str, Dict[Key, bool]], agents: Sequence[str]) -> float:
    c = 0
    for k in keys:
        if any(per_agent_correct[a].get(k, False) for a in agents):
            c += 1
    return _acc(c, len(keys))


def _findings_from_mapping(
    policy_name: str,
    mapping: Dict[str, str],
    selected_counts: Counter,
    baseline_best_single: Tuple[str, float],
    oracle_acc: float,
    policy_acc: float,
) -> str:
    """
    生成可直接塞进论文/附录的“路由发现”文字（无需再调用 LLM）。
    """
    best_agent, best_acc = baseline_best_single
    lines: List[str] = []
    lines.append(f"AOA policy = {policy_name}")
    lines.append(f"- best-single agent = {best_agent}, acc={best_acc:.4f}")
    lines.append(f"- oracle-any upper bound = {oracle_acc:.4f}")
    lines.append(f"- routed acc = {policy_acc:.4f}")
    lines.append("")
    lines.append("Routing table (subset):")
    # 控制长度：按“被选次数最多的 agent”相关条目优先展示
    hot_agents = [a for a, _ in selected_counts.most_common(3)]
    shown = 0
    for k, v in sorted(mapping.items()):
        if v in hot_agents:
            lines.append(f"- {k} -> {v}")
            shown += 1
        if shown >= 30:
            break
    if shown == 0:
        for k, v in list(sorted(mapping.items()))[:20]:
            lines.append(f"- {k} -> {v}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
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
    ap.add_argument(
        "--timestamp",
        type=str,
        default="",
        help="If empty, auto-pick latest common timestamp across agents.",
    )
    ap.add_argument(
        "--policy",
        type=str,
        default="capdiff_best",
        choices=["task_best", "cap_best", "capdiff_best"],
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results/StructuredReasoning_run/capability_eval_mode/aoa",
        help=(
            "Output root dir. For consistency with capability_eval_mode, results are saved to "
            "<out_dir>/<mode>/<timestamp>/offline_eval/<policy>/ (contains aoa_summary.json + findings.txt)."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cap_eval_root = Path(args.cap_eval_root)
    mode = args.mode
    agents = [_canon_agent_name(x) for x in args.agents.split(",") if x.strip()]
    if len(agents) < 2:
        raise ValueError("AOA 至少需要 2 个 agents 才有意义。")

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

    # 从任意一个 agent 拿 sample meta（capability/difficulty 应一致）
    any_agent = agents[0]
    key_to_info: Dict[Key, SampleInfo] = {k: per_agent[any_agent][k][0] for k in keys}

    per_agent_correct: Dict[str, Dict[Key, bool]] = {
        a: {k: per_agent[a][k][1] for k in keys} for a in agents
    }

    # baselines
    best_single_agent, best_single_acc = _best_single(keys, per_agent_correct, agents)
    oracle_acc = _oracle_any(keys, per_agent_correct, agents)

    # build mapping + evaluate
    mapping: Dict[str, str]
    if args.policy == "task_best":
        mapping = _build_task_best(keys, per_agent_correct, key_to_info, agents)
    elif args.policy == "cap_best":
        mapping = _build_cap_best(keys, per_agent_correct, key_to_info, agents)
    else:
        mapping = _build_capdiff_best(keys, per_agent_correct, key_to_info, agents)

    routed_acc, selected_counts = _eval_policy_select(
        keys=keys,
        per_agent_correct=per_agent_correct,
        key_to_info=key_to_info,
        policy_name=args.policy,
        mapping=mapping,
    )

    out_root = Path(args.out_dir) / mode / timestamp / "offline_eval" / args.policy
    out_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "aoa": {
            "policy": args.policy,
            "agents": agents,
            "cap_eval_root": str(cap_eval_root),
            "mode": mode,
            "timestamp": timestamp,
            "aligned_samples": len(keys),
        },
        "baselines": {
            "best_single": {"agent": best_single_agent, "accuracy": best_single_acc},
            "oracle_any": {"accuracy": oracle_acc},
        },
        "routed": {
            "accuracy": routed_acc,
            "selected_agent_counts": dict(selected_counts),
            "mapping": mapping,
        },
        "findings_text": _findings_from_mapping(
            policy_name=args.policy,
            mapping=mapping,
            selected_counts=selected_counts,
            baseline_best_single=(best_single_agent, best_single_acc),
            oracle_acc=oracle_acc,
            policy_acc=routed_acc,
        ),
    }

    with (out_root / "aoa_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (out_root / "findings.txt").open("w", encoding="utf-8") as f:
        f.write(payload["findings_text"])

    print(f"[AOA] Wrote: {(out_root / 'aoa_summary.json').resolve()}")
    print(
        f"[AOA] best_single={best_single_agent}({best_single_acc:.4f}) "
        f"routed={routed_acc:.4f} oracle_any={oracle_acc:.4f} "
        f"(N={len(keys)})"
    )


if __name__ == "__main__":
    main()



