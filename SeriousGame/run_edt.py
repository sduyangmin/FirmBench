#!/usr/bin/env python3
"""Runner script for SeriousGame/EDT evaluation.

EDT is evaluated in a *scenario-level* regime:

- The agent receives a template scenario (e.g. scenarios/interactive.json).
- It outputs a minimal decision schema once per episode.
- The evaluator materializes a new scenario file, starts a BPTK server for that
  scenario, then runs the episode **without** mid-episode interventions.

This entrypoint mirrors the structure of SeriousGame/run_beergame.py:
- load scenarios -> test_samples
- build a config dict
- instantiate agent
- call agent.run(...)

The episode rollout and BPTK/MCP orchestration lives in utils.seriousgame_tools.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from .edt_data_processor import EDTDataProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SeriousGame - EDT Runner (scenario-level)")

    # common
    p.add_argument("--mode", type=str, default="eval_only", choices=["online", "eval_only"])
    p.add_argument(
        "--agent_method",
        type=str,
        default="amem",
        choices=["amem", "cot", "ace", "mem0", "debate", "discussion", "gepa", "reflexion", "self-refine", "dc"],
        help="Agent method",
    )
    p.add_argument(
        "--api_provider",
        type=str,
        default="openai",
        choices=["openai", "deepseek", "together", "sambanova"],
        help="LLM API provider",
    )
    p.add_argument("--generator_model", type=str, default="gpt-5", help="Generator model name")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--save_dir", type=str, default="results")

    # EDT scenarios
    p.add_argument(
        "--scenario_path",
        type=str,
        default="",
        help="Optional scenarios.json/jsonl. If empty, defaults to one episode based on interactive.",
    )
    p.add_argument("--episodes", type=int, default=0, help="If >0, repeat scenarios to reach this count")

    # BPTK template repo (must contain ./scenarios and ./simulation_models)
    p.add_argument(
        "--bptk_repo_root",
        type=str,
        default=os.getenv("BPTK_REPO_ROOT", ""),
        help=(
            "Path to the BPTK repo root that contains EDT's ./scenarios and ./simulation_models. "
            "For the tutorial repo this is typically: model_library/enterprise_digital_twin"
        ),
    )
    p.add_argument(
        "--bptk_script",
        type=str,
        default="SeriousGame/run_bptk_server.py",
        help="Path to run_bptk_server.py (relative to FinAgent repo root).",
    )
    p.add_argument("--bptk_host", type=str, default="127.0.0.1", help="Host bind for the per-episode BPTK server")
    p.add_argument(
        "--bptk_bearer_token",
        type=str,
        default=os.getenv("BPTK_BEARER_TOKEN", ""),
        help="Optional bearer token for BptkServer.",
    )
    p.add_argument(
        "--bptk_start_timeout_sec",
        type=float,
        default=60.0,
        help="Max seconds to wait for BptkServer /scenarios to become available.",
    )

    # EDT MCP server
    p.add_argument(
        "--edt_mcp_server_path",
        type=str,
        default="SeriousGame/edt_mcp_server_local.py",
        help="Path to edt_mcp_server_local.py (relative to repo root)",
    )
    p.add_argument("--mcp_timeout_sec", type=float, default=90.0)

    # tool mapping (optional)
    p.add_argument("--toolmap_json", type=str, default="", help="Optional JSON string for tool name mapping")

    # scenario generation debug
    p.add_argument(
        "--keep_temp_bptk_repo",
        action="store_true",
        help="Keep temporary generated BPTK repo roots (for debugging).",
    )

    p.add_argument(
        "--reflector_model",
        type=str,
        default=None,
        help="Reflector model name (used by ACE/GEPA). Defaults to generator_model if not set.",
    )
    p.add_argument(
        "--curator_model",
        type=str,
        default=None,
        help="ACE only: curator model name; defaults to generator_model if not set.",
    )
    p.add_argument(
        "--initial_playbook_path",
        type=str,
        default=None,
        help="ACE only: optional path to an initial playbook file.",
    )
    p.add_argument(
        "--use_bulletpoint_analyzer",
        action="store_true",
        help="ACE only: enable bulletpoint analyzer.",
    )
    p.add_argument(
        "--bulletpoint_analyzer_threshold",
        type=float,
        default=0.90,
        help="ACE only: confidence threshold for bulletpoint analyzer.",
    )

    # Debate (optional)
    p.add_argument("--debate_rounds", type=int, default=1)
    p.add_argument("--debate_pro_temperature", type=float, default=0.0)
    p.add_argument("--debate_con_temperature", type=float, default=0.2)
    p.add_argument("--debate_judge_temperature", type=float, default=0.0)

    # Discussion (optional)
    p.add_argument("--discussion_num_experts", type=int, default=3)
    p.add_argument("--discussion_rounds", type=int, default=1)
    p.add_argument("--discussion_expert_temperature", type=float, default=0.2)
    p.add_argument("--discussion_moderator_temperature", type=float, default=0.0)

    return p.parse_args()


def _repo_root() -> str:
    """Infer FinAgent repo root from this file's location."""
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(here))


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print("SeriousGame Runner - EDT (scenario-level)")
    print(f"{'='*60}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Agent Method: {args.agent_method.upper()}")
    print(f"Generator Model: {args.generator_model}")
    print(f"Scenarios file: {args.scenario_path or '(default: interactive)'}")
    print(f"BPTK template repo root: {args.bptk_repo_root or '(NOT SET)'}")
    print(f"BPTK server script: {args.bptk_script}")
    print(f"EDT MCP server: {args.edt_mcp_server_path}")
    print(f"{'='*60}\n")

    raw = EDTDataProcessor.load_scenarios(args.scenario_path if args.scenario_path else None)
    test_samples = EDTDataProcessor.process_task_data(raw)

    if args.episodes and args.episodes > 0:
        base = list(test_samples)
        out = []
        i = 0
        while len(out) < args.episodes:
            s = dict(base[i % len(base)])
            s["scenario_id"] = f"{s['scenario_id']}_rep{len(out):04d}"
            out.append(s)
            i += 1
        test_samples = out

    repo_root = _repo_root()

    # resolve scripts to absolute paths (consistent with run_beergame)
    bptk_script = args.bptk_script if os.path.isabs(args.bptk_script) else os.path.join(repo_root, args.bptk_script)
    edt_mcp_script = (
        args.edt_mcp_server_path
        if os.path.isabs(args.edt_mcp_server_path)
        else os.path.join(repo_root, args.edt_mcp_server_path)
    )

    toolmap = json.loads(args.toolmap_json) if args.toolmap_json else {
        "new_episode": "init-edt-env",
        "step": "step-edt-env",
        "metrics": "get-edt-metrics",  # unused in our current evaluator; kept for parity
        "trace": "get-edt-state",
        "close": "close-edt-env",
    }

    config = {
        "task_name": "SeriousGame/EDT",
        "save_dir": args.save_dir,
        "agent_method": args.agent_method,
        "json_mode": False,
        "test_workers": 1,
        "edt": {
            # MCP server command is stable; base_url is injected per episode by the evaluator
            "mcp": {
                "command": os.environ.get("PYTHON", os.sys.executable),
                "args": [edt_mcp_script],
                "timeout_sec": args.mcp_timeout_sec,
                "max_retry_attempts": 2,
                "retry_backoff_seconds_base": 1.0,
                "cache_tools_list": False,
            },
            "toolmap": toolmap,
            "mcp_episode_config_key": "config",
            "max_steps": 96,
            "keep_temp_bptk_repo": bool(args.keep_temp_bptk_repo),
            "bptk": {
                "template_repo_root": args.bptk_repo_root,
                "script": bptk_script,
                "host": args.bptk_host,
                "bearer_token": args.bptk_bearer_token,
                "start_timeout_sec": args.bptk_start_timeout_sec,
            },
            "base_scenario": {
                "scenario_manager": "smEDT",
                "scenario_file": "interactive.json",
                "scenario_key": "interactive",
            },
        },
    }

    if args.agent_method == "amem":
        from Agents.amem import AMemAgent

        agent = AMemAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "cot":
        from Agents.cot import ChainOfThoughtAgent

        agent = ChainOfThoughtAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "ace":
        from Agents.ace.agent import ACE

        reflector_model = args.reflector_model or args.generator_model
        curator_model = args.curator_model or args.generator_model

        initial_playbook = None
        if args.initial_playbook_path and os.path.exists(args.initial_playbook_path):
            with open(args.initial_playbook_path, "r", encoding="utf-8") as f:
                initial_playbook = f.read()
            print(f"[ACE] Loaded initial playbook from {args.initial_playbook_path}")
        elif args.initial_playbook_path:
            print(
                f"[ACE][WARN] initial_playbook_path '{args.initial_playbook_path}' "
                "not found. Using empty playbook."
            )

        agent = ACE(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=reflector_model,
            curator_model=curator_model,
            max_tokens=args.max_tokens,
            initial_playbook=initial_playbook,
            use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
            bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
            agent_method="ace",
        )
    elif args.agent_method == "mem0":
        from Agents.mem0.agent import Mem0Agent

        agent = Mem0Agent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "debate":
        from Agents.debate import DebateAgent

        agent = DebateAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
            rounds=args.debate_rounds,
            pro_temperature=args.debate_pro_temperature,
            con_temperature=args.debate_con_temperature,
            judge_temperature=args.debate_judge_temperature,
        )
    elif args.agent_method == "discussion":
        from Agents.discussion import DiscussionAgent

        agent = DiscussionAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
            num_experts=args.discussion_num_experts,
            rounds=args.discussion_rounds,
            expert_temperature=args.discussion_expert_temperature,
            moderator_temperature=args.discussion_moderator_temperature,
        )
    elif args.agent_method == "gepa":
        from Agents.gepa import GEPAAgent

        reflector_model = args.reflector_model or args.generator_model
        agent = GEPAAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=reflector_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "reflexion":
        from Agents.reflexion import ReflexionAgent
        config["edt"]["online"] = {
            "utilization_threshold": 0.70,
            "reflect_on_success": False,
        }
        agent = ReflexionAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "self-refine":
        from Agents.self_refine import SelfRefineAgent
        agent = SelfRefineAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "dc":
        from Agents.dynamic_cheatsheet import DynamicCheatsheetAgent
        if "edt" not in config or not isinstance(config["edt"], dict):
            config["edt"] = {}
        config["edt"].update({
                "cheatsheet_window_size": 1,
                "initial_cheatsheet_path": None,
        })
        agent = DynamicCheatsheetAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    else:
        raise ValueError(f"Unsupported agent_method '{args.agent_method}' for EDT.")

    agent.run(
        mode=args.mode,
        test_samples=test_samples,
        data_processor=None,
        config=config,
    )


if __name__ == "__main__":
    main()
