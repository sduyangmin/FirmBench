#!/usr/bin/env python3
"""
Runner script for StructuredReasoning tasks supporting multiple agent methods.
"""
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

from .data_processor import DataProcessor
from utils.easyhard import convert_domain_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ACE System - StructuredReasoning")

    parser.add_argument(
        "--agent_method",
        type=str,
        default="ace",
        choices=[
            "ace",
            "cot",
            "aoa",
            "amem",
            "dynamic_cheatsheet",
            "self_refine",
            "self-refine",
            "reflexion",
            "gepa",
            "debate",
            "discussion",
        ],
        help="Agent method to run. 'ace' 为默认流程, 'cot' 是轻量 baseline, "
        "'dynamic_cheatsheet' 启用动态小抄, "
        "'self_refine' 采用自我反馈迭代，'reflexion' 为自反思改写，'gepa' 为 prompt 演化，"
        "'debate' 为正反辩论+裁判汇总，'discussion' 为多专家讨论+主持人汇总。",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=False,
        default=None,
        choices=[
            "FinCode",
            "CodeFinQA",
            "CodeTAT-QA",
            "SEC-NUM",
            "TAT-QA",
            "ConvFinQA",
            "FinKnow",
            "FormulaEval",
            "finer",
            "formula",
        ],
        help="StructuredReasoning task to run（run_mode=dataname 时必填，easyhard 时忽略）",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="dataname",
        choices=["dataname", "easyhard"],
        help="dataname: 按数据集正常运行；easyhard: 按 domain 对现有结果做难度聚合转换。",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="当 run_mode=easyhard 时必填，取值: Finance_Reasoning, Span_extraction, Knowledge_understand（下划线替代表达空格）。",
    )
    # ace specific knobs
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"],
                        help="Run mode")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Optional initial playbook")
    parser.add_argument("--config_path", type=str,
                        default="./StructuredReasoning/data/task_config.json",
                        help="Path to task config JSON")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save results")

    parser.add_argument("--api_provider", type=str, default="usd_guiji",
                        choices=["sambanova", "together", "openai", "usd_guiji"])
    parser.add_argument("--generator_model", type=str,
                        default="USD-guiji/deepseek-v3")
    parser.add_argument("--reflector_model", type=str,
                        default="USD-guiji/deepseek-v3")
    parser.add_argument("--curator_model", type=str,
                        default="USD-guiji/deepseek-v3")

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--online_eval_frequency", type=int, default=15)
    parser.add_argument("--save_steps", type=int, default=50)

    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--test_workers", type=int, default=20)
    # ConsultingInterview-specific args removed
    parser.add_argument(
        "--data_embedding_dir",
        type=str,
        default="StructuredReasoning/data/data_embedding",
        help="Embedding CSV 目录，默认 StructuredReasoning/data/data_embedding",
    )

    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true")
    parser.add_argument("--bulletpoint_analyzer_threshold",
                        type=float, default=0.90)

    # self-refine specific knobs
    parser.add_argument(
        "--self_refine_rounds",
        type=int,
        default=2,
        help="自我反馈的迭代次数，0 表示仅初始生成。",
    )
    parser.add_argument(
        "--self_refine_initial_temperature",
        type=float,
        default=0.0,
        help="Self-refine 初始回答温度。",
    )
    parser.add_argument(
        "--self_refine_feedback_temperature",
        type=float,
        default=0.2,
        help="Self-refine 反馈-改写阶段温度。",
    )

    # reflexion specific knobs
    parser.add_argument(
        "--reflexion_rounds",
        type=int,
        default=2,
        help="Reflexion 反思-重写的迭代次数，0 表示仅初始生成。",
    )
    parser.add_argument(
        "--reflexion_initial_temperature",
        type=float,
        default=0.0,
        help="Reflexion 初始回答温度。",
    )
    parser.add_argument(
        "--reflexion_reflect_temperature",
        type=float,
        default=0.2,
        help="Reflexion 反思/重写阶段温度。",
    )
    parser.add_argument(
        "--reflexion_strict_sequential",
        action="store_true",
        help="开启后全程串行评估+记忆更新（最贴合论文时序，但性能较低）。默认按窗口并行评估、串行更新。",
    )
    parser.add_argument(
        "--reflexion_memory_top_k",
        type=int,
        default=3,
        help="检索同题历史反思的最大条数（精确匹配 question+context，无相似度检索）。",
    )

    # GEPA specific knobs
    parser.add_argument(
        "--gepa_budget",
        type=int,
        default=80,
        help="GEPA: 总预算（按样本评估次数计数）。",
    )
    parser.add_argument(
        "--gepa_window_budget",
        type=int,
        default=150,
        help="GEPA: online 窗口内单窗预算（覆盖总预算均分）。",
    )
    parser.add_argument(
        "--gepa_mini_batch_size",
        type=int,
        default=2,
        help="GEPA: mini-batch 反馈样本数。",
    )
    parser.add_argument(
        "--gepa_num_initial",
        type=int,
        default=6,
        help="GEPA: 初始候选 prompt 数量。",
    )
    parser.add_argument(
        "--gepa_exploit_prob",
        type=float,
        default=0.95,
        help="GEPA: 选择 exploit 的概率 P。",
    )
    parser.add_argument(
        "--gepa_merge_prob",
        type=float,
        default=0.9,
        help="GEPA: 在 exploit 下不 merge 的概率 Q（反之触发 merge）。",
    )
    parser.add_argument(
        "--gepa_seed_prompt",
        type=str,
        default=None,
        help="GEPA: 种子 prompt（为空则用默认 SEED_PROMPT）。",
    )
    parser.add_argument(
        "--gepa_feedback_ratio",
        type=float,
        default=0.5,
        help="GEPA: eval_only 下测试集拆分为反馈集的比例。",
    )
    parser.add_argument(
        "--gepa_max_workers",
        type=int,
        default=8,
        help="GEPA: 并发线程数。",
    )
    parser.add_argument(
        "--gepa_target_temperature",
        type=float,
        default=0.0,
        help="GEPA: 目标模型温度。",
    )
    parser.add_argument(
        "--gepa_reflection_temperature",
        type=float,
        default=0.2,
        help="GEPA: 反思/合并模型温度。",
    )


    # Dynamic Cheatsheet specific knobs
    parser.add_argument(
        "--dc_approach",
        type=str,
        default="DynamicCheatsheet_RetrievalSynthesis",
        choices=[
            "DynamicCheatsheet_Cumulative",
            "default",
            "FullHistoryAppending",
            "Dynamic_Retrieval",
            "DynamicCheatsheet_RetrievalSynthesis",
        ],
        help="Dynamic Cheatsheet approach variant.",
    )
    parser.add_argument(
        "--dc_generator_prompt_path",
        type=str,
        default="Agents/dynamic_cheatsheet/prompts/generator_prompt.txt",
        help="Path to the Dynamic Cheatsheet generator prompt.",
    )
    parser.add_argument(
        "--dc_cheatsheet_prompt_path",
        type=str,
        default="Agents/dynamic_cheatsheet/prompts/cheatsheet_cumulative.txt",
        help="Path to the Dynamic Cheatsheet curator prompt (if required).",
    )
    parser.add_argument(
        "--dc_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Dynamic Cheatsheet generations.",
    )
    parser.add_argument(
        "--dc_retrieve_top_k",
        type=int,
        default=3,
        help="Top-k retrieval size for Dynamic Cheatsheet retrieval modes.",
    )
    parser.add_argument(
        "--dc_disable_code_execution",
        action="store_true",
        help="Disable EXECUTE CODE! loops for Dynamic Cheatsheet.",
    )
    parser.add_argument(
        "--dc_disable_previous_answers",
        action="store_true",
        help="Do not inject previous answers into the cheatsheet context.",
    )

    # Debate specific knobs
    parser.add_argument(
        "--debate_rounds",
        type=int,
        default=1,
        help="Debate: 正反交锋轮数（每轮包含 PRO+CON），最后由 JUDGE 给最终答案。",
    )
    parser.add_argument(
        "--debate_pro_temperature",
        type=float,
        default=0.0,
        help="Debate: PRO 侧温度。",
    )
    parser.add_argument(
        "--debate_con_temperature",
        type=float,
        default=0.2,
        help="Debate: CON 侧温度。",
    )
    parser.add_argument(
        "--debate_judge_temperature",
        type=float,
        default=0.0,
        help="Debate: JUDGE 温度。",
    )

    # Discussion specific knobs
    parser.add_argument(
        "--discussion_num_experts",
        type=int,
        default=3,
        help="Discussion: 专家数量（每个专家独立给出方案）。",
    )
    parser.add_argument(
        "--discussion_rounds",
        type=int,
        default=1,
        help="Discussion: 讨论轮数（>1 时专家会看到上一轮其他专家的输出再给新一轮）。",
    )
    parser.add_argument(
        "--discussion_expert_temperature",
        type=float,
        default=0.2,
        help="Discussion: 专家温度。",
    )
    parser.add_argument(
        "--discussion_moderator_temperature",
        type=float,
        default=0.0,
        help="Discussion: 主持人温度。",
    )

    # AOA (agent-of-agents) specific knobs
    parser.add_argument(
        "--aoa_experience_path",
        type=str,
        default="results/StructuredReasoning_run/aoa_mode/experience/experience.latest.json",
        help="AOA: experience JSON (findings + routing_policy) used by router.",
    )
    parser.add_argument(
        "--aoa_router_model",
        type=str,
        default="",
        help="AOA: router model name (default: same as --generator_model).",
    )
    parser.add_argument(
        "--aoa_router_temperature",
        type=float,
        default=0.0,
        help="AOA: router temperature.",
    )
    parser.add_argument(
        "--aoa_router_max_tokens",
        type=int,
        default=512,
        help="AOA: router max tokens for the routing decision call.",
    )
    parser.add_argument(
        "--aoa_candidates",
        type=str,
        default="cot,amem,self_refine,reflexion,debate,discussion",
        help="AOA: comma-separated candidate underlying agent methods.",
    )
    parser.add_argument(
        "--aoa_include_dynamic_cheatsheet",
        action="store_true",
        help="AOA: allow router to choose dynamic_cheatsheet (stateful, slower).",
    )
    parser.add_argument(
        "--aoa_labels_jsonl",
        type=str,
        default="",
        help="AOA: optional classifications.jsonl for this task (index-aligned to test_samples) to provide capability/difficulty labels.",
    )

    return parser.parse_args()


def load_data(data_path: str):
    """Load jsonl data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def preprocess_data(task_name: str, config: dict, mode: str):
    """Load and preprocess data according to mode."""
    processor = DataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None

        if "test_data" not in config:
            raise ValueError(f"{mode} mode requires test_data in config.")

        test_samples = load_data(config["test_data"])
        test_samples = processor.process_task_data(test_samples)

        if mode == "online":
            print(f"Online mode: Training and testing on {len(test_samples)} examples")
        else:
            print(f"Eval only mode: Testing on {len(test_samples)} examples")

    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []

        print(
            f"Offline mode: Training on {len(train_samples)} examples, "
            f"validating on {len(val_samples)}, testing on {len(test_samples)}"
        )

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path: str):
    """Load initial playbook if path exists."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def main():
    args = parse_args()
    # easyhard 模式：基于现有结果做转换，不重新跑模型
    if args.run_mode == "easyhard":
        if not args.domain:
            raise ValueError("run_mode=easyhard 时必须提供 --domain")
        domain = args.domain.replace(" ", "_")
        convert_domain_results(
            domain=domain,
            agent_method=args.agent_method,
            mode=args.mode,
            config_path=args.config_path,
            results_root=args.save_path,
        )
        return
    if not args.task_name:
        raise ValueError("run_mode=dataname 时必须提供 --task_name")

    if args.agent_method == "self-refine":
        # 统一成模块友好的命名
        args.agent_method = "self_refine"

    print(f"\n{'='*60}")
    print("StructuredReasoning Runner")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Agent Method: {args.agent_method.upper()}")
    print(f"Generator Model: {args.generator_model}")
    print(f"{'='*60}\n")

    with open(args.config_path, "r", encoding="utf-8") as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name,
        task_config[args.task_name],
        args.mode,
    )

    initial_playbook = None
    if args.agent_method == "ace":
        initial_playbook = load_initial_playbook(args.initial_playbook_path)
        if initial_playbook:
            print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
        else:
            print("Using empty playbook as initial playbook\n")
    elif args.initial_playbook_path:
        print("Warning: --initial_playbook_path is ignored for non-ACE agent methods.\n")

    # 保留所有 CLI 参数到 config，方便 run_config.json 完整记录
    config = vars(args).copy()
    config["save_dir"] = args.save_path

    if args.agent_method == "ace":
        from Agents.ace import ACE

        ace_system = ACE(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=args.reflector_model,
            curator_model=args.curator_model,
            max_tokens=args.max_tokens,
            initial_playbook=initial_playbook,
            use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
            bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
            agent_method=args.agent_method,
        )

        ace_system.run(
            mode=args.mode,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    elif args.agent_method == "cot":
        from Agents.cot import ChainOfThoughtAgent

        if args.mode not in ["online", "eval_only"]:
            raise ValueError(f"{args.agent_method.upper()} agent 当前只支持 online/eval_only 模式，收到 {args.mode}")

        cot_agent = ChainOfThoughtAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
        cot_agent.run(
            mode=args.mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    elif args.agent_method == "amem":
        from Agents.amem import AMemAgent

        if args.mode not in ["online", "eval_only"]:
            raise ValueError(f"{args.agent_method.upper()} agent 当前只支持 online/eval_only 模式，收到 {args.mode}")

        amem_agent = AMemAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )

        amem_agent.run(
            mode=args.mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    elif args.agent_method == "dynamic_cheatsheet":
        from Agents.dynamic_cheatsheet.agent import DynamicCheatsheetConfig
        from Agents.dynamic_cheatsheet import DynamicCheatsheetAgent

        if args.mode not in ["online", "eval_only"]:
            raise ValueError(f"{args.agent_method.upper()} agent 当前只支持 online/eval_only 模式，收到 {args.mode}")

        dc_config = DynamicCheatsheetConfig(
            approach_name=args.dc_approach,
            generator_prompt_path=Path(args.dc_generator_prompt_path),
            cheatsheet_prompt_path=Path(args.dc_cheatsheet_prompt_path)
            if args.dc_cheatsheet_prompt_path
            else None,
            temperature=args.dc_temperature,
            max_num_rounds=args.max_num_rounds,
            retrieve_top_k=args.dc_retrieve_top_k,
            allow_code_execution=not args.dc_disable_code_execution,
            add_previous_answers=not args.dc_disable_previous_answers,
        )

        dc_agent = DynamicCheatsheetAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
            dc_config=dc_config,
        )

        dc_agent.run(
            mode=args.mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    elif args.agent_method == "self_refine":
        from Agents.self_refine import SelfRefineAgent

        if args.mode not in ["online", "eval_only"]:
            raise ValueError(f"{args.agent_method.upper()} agent 当前只支持 online/eval_only 模式，收到 {args.mode}")

        self_refine_agent = SelfRefineAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            refine_rounds=args.self_refine_rounds,
            initial_temperature=args.self_refine_initial_temperature,
            feedback_temperature=args.self_refine_feedback_temperature,
            agent_method=args.agent_method,
        )

        self_refine_agent.run(
            mode=args.mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    elif args.agent_method == "reflexion":
        from Agents.reflexion import ReflexionAgent

        if args.mode not in ["online", "eval_only"]:
            raise ValueError(f"{args.agent_method.upper()} agent 当前只支持 online/eval_only 模式，收到 {args.mode}")

        reflexion_agent = ReflexionAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            reflexion_rounds=args.reflexion_rounds,
            initial_temperature=args.reflexion_initial_temperature,
            reflect_temperature=args.reflexion_reflect_temperature,
            agent_method=args.agent_method,
            strict_sequential=args.reflexion_strict_sequential,
            memory_top_k=args.reflexion_memory_top_k,
        )

        reflexion_agent.run(
            mode=args.mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    elif args.agent_method == "gepa":
        from Agents.gepa import GEPAAgent
        from Agents.gepa.config import GepaConfig

        if args.mode not in ["online", "eval_only", "offline"]:
            raise ValueError(f"{args.agent_method.upper()} agent 当前只支持 online/eval_only/offline 模式，收到 {args.mode}")

        gepa_cfg = GepaConfig(
            budget=args.gepa_budget,
            window_budget=args.gepa_window_budget,
            mini_batch_size=args.gepa_mini_batch_size,
            num_initial=args.gepa_num_initial,
            exploit_prob=args.gepa_exploit_prob,
            merge_prob=args.gepa_merge_prob,
            seed_prompt=args.gepa_seed_prompt,
            feedback_ratio=args.gepa_feedback_ratio,
            max_workers=args.gepa_max_workers,
            target_temperature=args.gepa_target_temperature,
            reflection_temperature=args.gepa_reflection_temperature,
            max_tokens=args.max_tokens,
        )

        gepa_agent = GEPAAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=args.reflector_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
            gepa_config=gepa_cfg,
        )

        gepa_agent.run(
            mode=args.mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
            train_samples=train_samples,
            val_samples=val_samples,
        )
    elif args.agent_method == "debate":
        from Agents.debate import DebateAgent

        if args.mode not in ["online", "eval_only"]:
            raise ValueError(
                f"{args.agent_method.upper()} agent 当前只支持 online/eval_only 模式，收到 {args.mode}"
            )

        debate_agent = DebateAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
            rounds=args.debate_rounds,
            pro_temperature=args.debate_pro_temperature,
            con_temperature=args.debate_con_temperature,
            judge_temperature=args.debate_judge_temperature,
        )
        debate_agent.run(
            mode=args.mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    elif args.agent_method == "discussion":
        from Agents.discussion import DiscussionAgent

        if args.mode not in ["online", "eval_only"]:
            raise ValueError(
                f"{args.agent_method.upper()} agent 当前只支持 online/eval_only 模式，收到 {args.mode}"
            )

        discussion_agent = DiscussionAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
            num_experts=args.discussion_num_experts,
            rounds=args.discussion_rounds,
            expert_temperature=args.discussion_expert_temperature,
            moderator_temperature=args.discussion_moderator_temperature,
        )
        discussion_agent.run(
            mode=args.mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    elif args.agent_method == "aoa":
        from Agents.aoa.agent import AOAAgent, AOAConfig

        if args.mode not in ["online", "eval_only"]:
            raise ValueError(f"{args.agent_method.upper()} agent 当前只支持 online/eval_only 模式，收到 {args.mode}")

        router_model = args.aoa_router_model.strip() or args.generator_model
        candidates = [c.strip() for c in (args.aoa_candidates or "").split(",") if c.strip()]
        aoa_cfg = AOAConfig(
            router_model=router_model,
            router_temperature=args.aoa_router_temperature,
            router_max_tokens=args.aoa_router_max_tokens,
            candidates=tuple(candidates),
            include_dynamic_cheatsheet=bool(args.aoa_include_dynamic_cheatsheet),
        )

        # Pass extra knobs through config for logging
        config["aoa_experience_path"] = args.aoa_experience_path
        config["aoa_labels_jsonl"] = args.aoa_labels_jsonl

        aoa_agent = AOAAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method="aoa",
            aoa_config=aoa_cfg,
            experience_path=args.aoa_experience_path,
        )
        aoa_agent.run(
            mode=args.mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    else:
        raise ValueError(f"未知的 agent_method: {args.agent_method}")

if __name__ == "__main__":
    main()


