#!/usr/bin/env python3
"""
Runner script for SeriousGame/BeerGame evaluation, mirroring bizbench/run.py pattern.

This script:
- loads scenarios -> test_samples
- instantiates agent method
- calls agent.run(...)

The evaluation loop (MCP episode rollout) lives in the agent (via utils.seriousgame_tools),
so run.py remains a thin entrypoint.
"""

import argparse
import json
import os

from datetime import datetime

from .beergame_data_processor import BeerGameDataProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="SeriousGame - BeerGame Runner")
    parser.add_argument(
        "--mode",
        type=str,
        default="eval_only",
        choices=["online", "eval_only"],
    )
    parser.add_argument(
        "--agent_method",
        type=str,
        default="amem",
        choices=[
            "amem",
            "cot",
            "ace",
            "mem0",
            "debate",
            "discussion",
            "reflexion",
            "self_refine",
            "dynamic_cheatsheet",
            "gepa",
        ],
        help="Agent method to run",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        default="openai",
        choices=["openai", "deepseek", "together", "sambanova", "usd_guiji"],
        help="LLM API provider",
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default="gpt-5",
        help="Generator model name",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
    )

    # BeerGame scenarios
    parser.add_argument(
        "--scenario_path",
        type=str,
        default="",
        help="Path to scenarios.json or scenarios.jsonl",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="If >0, repeat scenarios to reach this count",
    )

    # MCP server
    parser.add_argument(
        "--server_path",
        type=str,
        default="SeriousGame/beergame_mcp_server.py",
        help="Path to beergame_mcp_server.py (relative to repo root)",
    )
    parser.add_argument(
        "--mcp_timeout_sec",
        type=float,
        default=60.0,
    )

    # Optional tool mapping if your server uses different names
    parser.add_argument(
        "--toolmap_json",
        type=str,
        default="",
        help="JSON string for tool name mapping",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("SeriousGame Runner - BeerGame")
    print(f"{'='*60}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Agent Method: {args.agent_method.upper()}")
    print(f"Generator Model: {args.generator_model}")
    print(f"Scenarios file: {args.scenario_path or '(default scenarios)'}")
    print(f"MCP server: {args.server_path}")
    print(f"{'='*60}\n")

    raw = BeerGameDataProcessor.load_scenarios(
        args.scenario_path if args.scenario_path else None
    )
    test_samples = BeerGameDataProcessor.process_task_data(raw)

    if args.episodes and args.episodes > 0:
        # repeat scenarios deterministically to reach requested count
        base = list(test_samples)
        out = []
        i = 0
        while len(out) < args.episodes:
            s = dict(base[i % len(base)])
            s["scenario_id"] = f"{s['scenario_id']}_rep{len(out):04d}"
            out.append(s)
            i += 1
        test_samples = out

    # Build config (mirrors bizbench.run passing config dict into agent)
    config = {
        "task_name": "SeriousGame/BeerGame",
        "save_dir": args.save_dir,
        "json_mode": False,
        "test_workers": 1,  # episodes are run serially for determinism
        "mcp": {
            "command": "python",
            "args": [args.server_path],
            "timeout_sec": args.mcp_timeout_sec,
            "max_retry_attempts": 2,
            "retry_backoff_seconds_base": 1.0,
            "cache_tools_list": False,
        },
        # toolmap (optional)
        "toolmap": json.loads(args.toolmap_json)
        if args.toolmap_json
        else {
            # 客户端逻辑里的“语义键” -> MCP 服务器实际的工具名
            "new_episode": "init-beer-env",
            "step": "step-beer-env",
            "metrics": "get-beer-metrics",
            # 你当前 server 没有“完整trace”的工具，只有 get-beer-state（当前状态）
            # 如果你不打算用trace来算bullwhip，可以先不配trace，让评测逻辑自动跳过。
            # 如果想以后扩展，可以再加一个 get-beer-trace 工具。
            # "trace":     "get-beer-state",
            "close": "close-beer-env",
        },
        # how to wrap config for MCP new_episode call: {"config": episode_cfg}
        "mcp_episode_config_key": "config",
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
        from Agents.ace import ACE

        agent = ACE(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=args.generator_model,
            curator_model=args.generator_model,
            max_tokens=args.max_tokens,
            initial_playbook=None,
            use_bulletpoint_analyzer=False,
            bulletpoint_analyzer_threshold=0.90,
            agent_method=args.agent_method,
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
        from Agents.debate.agent import DebateAgent

        agent = DebateAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "discussion":
        from Agents.discussion.agent import DiscussionAgent

        agent = DiscussionAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "reflexion":
        from Agents.reflexion.agent import ReflexionAgent

        agent = ReflexionAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "self_refine":
        from Agents.self_refine.agent import SelfRefineAgent

        agent = SelfRefineAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "dynamic_cheatsheet":
        from Agents.dynamic_cheatsheet.agent import DynamicCheatsheetAgent

        agent = DynamicCheatsheetAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    elif args.agent_method == "gepa":
        from Agents.gepa.agent import GEPAAgent

        agent = GEPAAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
    else:
        raise ValueError(f"Unsupported agent_method: {args.agent_method}")

    agent.run(
        mode=args.mode,
        test_samples=test_samples,
        data_processor=None,  # kept for signature alignment
        config=config,
    )


if __name__ == "__main__":
    main()
