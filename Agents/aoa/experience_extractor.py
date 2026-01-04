from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List

from utils.llm import timed_llm_call
from utils.playbook_utils import extract_json_from_text
from utils.tools import initialize_clients


@dataclass(frozen=True)
class ExperienceExtractorConfig:
    api_provider: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _canon_agent_name(s: str) -> str:
    s = (s or "").strip()
    alias = {
        "self-refine": "self_refine",
        "dc": "dynamic_cheatsheet",
    }
    return alias.get(s, s)

def _infer_agents_from_cap_eval_csv_text(csv_text: str) -> List[str]:
    """
    capability_eval 的 CSV 第一行是表头：
      capability,difficulty,<agent_col_1>,<agent_col_2>,...
    这里把 agent 列名抽出来，并做简单归一化，作为 meta.agents 的 ground truth。
    """
    first = (csv_text or "").splitlines()[0].strip() if (csv_text or "").strip() else ""
    if not first:
        return []
    cols = [c.strip() for c in first.split(",") if c.strip()]
    if len(cols) < 3:
        return []
    agent_cols = cols[2:]
    # 保持与目录名一致（兼容 self-refine/dc）
    return [_canon_agent_name(a) for a in agent_cols]


def _build_input(results_table: str, notes: str, source: str) -> str:
    return "\n".join(
        [
            "You must output JSON only.",
            "",
            f"source: {source}",
            "",
            "results_table:",
            results_table.strip(),
            "",
            "notes:",
            notes.strip(),
            "",
        ]
    )


def extract_experience(
    cap_eval_csv_path: Path,
    out_path: Path,
    cfg: ExperienceExtractorConfig,
    prompt_path: Optional[Path] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """
    让 LLM 读取 capability_eval.csv，并生成可执行的经验 JSON（findings + routing_policy）。
    """
    prompt_path = prompt_path or (Path(__file__).parent / "prompts" / "experience_extractor.md")
    system_prompt = _read_text(prompt_path)
    results_table = _read_text(cap_eval_csv_path)
    agents_in_table = _infer_agents_from_cap_eval_csv_text(results_table)
    user_input = _build_input(
        results_table=results_table,
        notes=notes
        or (
            "- Information Extraction: 不分难度桶（easy/middle/hard 行重复同一数值）。\n"
            "- Complex Reasoning: easy 桶可能为空/样本稀少，需保守回退。\n"
            "- 你需要输出可执行 routing_policy（含 default/tie_break/min_margin）。\n"
        ),
        source=str(cap_eval_csv_path),
    )

    client, _, _ = initialize_clients(cfg.api_provider)
    raw, _info = timed_llm_call(
        client=client,
        api_provider=cfg.api_provider,
        model=cfg.model,
        prompt=user_input,  # 用于日志与异常记录
        role="aoa_extractor",
        call_id="aoa_experience_extract",
        max_tokens=cfg.max_tokens,
        log_dir=None,
        use_json_mode=True,
        temperature=cfg.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )

    exp = extract_json_from_text(raw)
    if exp is None:
        raise ValueError(
            "Experience Extractor 输出无法解析为 JSON（可能包含 markdown 包裹或混入解释文本）。"
            f"原始输出前 800 字:\n{raw[:800]}"
        )

    # 统一注入/覆盖 meta（避免模型写死时间或遗漏字段）
    if isinstance(exp, dict):
        exp.setdefault("meta", {})
        if isinstance(exp["meta"], dict):
            exp["meta"]["generated_at"] = _now_iso()
            exp["meta"]["source"] = str(cap_eval_csv_path)
            # 以输入表格表头为准，避免模型漏写 debate/discussion 等列
            exp["meta"]["agents"] = agents_in_table or exp["meta"].get("agents", [])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(exp, ensure_ascii=False, indent=2), encoding="utf-8")
    return exp


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cap_eval_csv",
        type=str,
        default="",
        help="Path to capability_eval.csv (table-based mode only). In --from_traces mode this is ignored.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="results/StructuredReasoning_run/aoa_mode/experience/experience.json",
        help="Output path for experience JSON.",
    )
    ap.add_argument("--api_provider", type=str, default="usd_guiji")
    ap.add_argument("--model", type=str, default="USD-guiji/deepseek-v3")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional prompt path. Default: Agents/aoa/prompts/experience_extractor.md",
    )
    ap.add_argument("--notes", type=str, default="", help="Optional extra notes injected to user input.")
    ap.add_argument(
        "--override_default_agent",
        type=str,
        default="",
        help="If set, override routing_policy.default_agent in the produced experience JSON (for stability/ablation).",
    )
    ap.add_argument(
        "--from_traces",
        action="store_true",
        help="If set, build experience from per-sample reasoning traces under results/StructuredReasoning_run (trace-based).",
    )
    ap.add_argument(
        "--trace_results_root",
        type=str,
        default="results/StructuredReasoning_run",
        help="Trace-based: results root that contains <Task>/<agent>/<mode>/<timestamp>/test_results.json and detailed_llm_logs.",
    )
    ap.add_argument("--trace_mode", type=str, default="online", choices=["online", "eval_only", "offline"])
    ap.add_argument("--trace_tasks", type=str, default="", help="Trace-based: comma-separated tasks (empty means all in task_config).")
    ap.add_argument(
        "--trace_agents",
        type=str,
        default="cot,ace,amem,self_refine,reflexion,gepa,debate,discussion,dynamic_cheatsheet",
        help="Trace-based: comma-separated agent methods to scan.",
    )
    ap.add_argument("--trace_timestamp", type=str, default="", help="Trace-based: if empty, use latest per (task,agent).")
    ap.add_argument(
        "--trace_task_config",
        type=str,
        default="StructuredReasoning/data/task_config.json",
        help="Trace-based: task_config.json for loading the original test samples.",
    )
    ap.add_argument(
        "--trace_classifications_jsonl",
        type=str,
        default="results/StructuredReasoning_run/llm_reclassify_mode/capability_difficulty_score_v1_merged_finer_try50/test/classifications.jsonl",
        help="Trace-based: classifications.jsonl for capability/difficulty labels.",
    )
    ap.add_argument("--trace_k_per_task_agent", type=int, default=50)
    ap.add_argument("--trace_seed", type=int, default=42)
    ap.add_argument("--trace_log_dir", type=str, default="results/StructuredReasoning_run/aoa_mode/experience/trace_logs")
    ap.add_argument("--trace_meta_max_bullets", type=int, default=600)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Trace-based experience mode (preferred for paper-quality experience).
    if args.from_traces:
        from Agents.aoa.experience_from_traces import main as trace_main  # lazy import

        # Re-invoke the trace-based CLI by reconstructing sys.argv for that module.
        import sys as _sys

        _sys.argv = [
            _sys.argv[0],
            "--results_root",
            args.trace_results_root,
            "--mode",
            args.trace_mode,
            "--tasks",
            args.trace_tasks,
            "--agents",
            args.trace_agents,
            "--timestamp",
            args.trace_timestamp,
            "--task_config",
            args.trace_task_config,
            "--classifications_jsonl",
            args.trace_classifications_jsonl,
            "--k_per_task_agent",
            str(args.trace_k_per_task_agent),
            "--seed",
            str(args.trace_seed),
            "--api_provider",
            args.api_provider,
            "--model",
            args.model,
            "--max_tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--out",
            args.out,
            "--log_dir",
            args.trace_log_dir,
            "--meta_max_bullets",
            str(args.trace_meta_max_bullets),
        ]
        trace_main()
        return

    # Table-based mode requires cap_eval_csv.
    if not args.cap_eval_csv:
        raise SystemExit("ERROR: --cap_eval_csv is required unless --from_traces is set.")

    exp = extract_experience(
        cap_eval_csv_path=Path(args.cap_eval_csv),
        out_path=Path(args.out),
        cfg=ExperienceExtractorConfig(
            api_provider=args.api_provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
        prompt_path=Path(args.prompt) if args.prompt else None,
        notes=args.notes,
    )

    # Optional override: force default_agent to a fixed value (e.g., 'ace') to avoid LLM bias.
    if args.override_default_agent and isinstance(exp, dict):
        rp = exp.get("routing_policy")
        if isinstance(rp, dict):
            rp["default_agent"] = args.override_default_agent.strip()
            exp["routing_policy"] = rp
            Path(args.out).write_text(json.dumps(exp, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[AOA] override_default_agent={rp['default_agent']}")

    print(f"[AOA] Wrote experience JSON: {Path(args.out).resolve()}")
    if isinstance(exp, dict):
        pol = (exp.get("routing_policy") or {}) if isinstance(exp.get("routing_policy"), dict) else {}
        print(f"[AOA] default_agent={pol.get('default_agent')}, rules={len(pol.get('rules') or [])}")


if __name__ == "__main__":
    main()


