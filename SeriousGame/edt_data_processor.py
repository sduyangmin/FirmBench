#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDT scenario data processor.

目标：
- 仿照 SeriousGame/beergame_data_processor.py 的模式；
- 把 EDT 的场景整理成统一的 test_samples：
    [{"scenario_id": "...", "config": {...}}, ...]

当前版本：
- 只准备一个固定场景 "interactive"；
- config 固定为 EDT_FIXED_CONFIG。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------
# 固定的 EDT 场景配置：EDT_FIXED_CONFIG
# ---------------------------------------------------------

# 固定 EDT 默认场景，但让 BptkServer 自动选择 ABM 指标
EDT_FIXED_CONFIG = {
    # 场景选择：与 EDT 教程一致
    "scenario_managers": ["smEDT"],
    "scenarios": ["interactive"],

    # 不主动指定方程（如果模型有 SD 部分，可后续再加）
    "equations": [],

    # ABM 相关先全部留空，让 auto_select_abm_metrics 去做
    "agents": [],
    "agent_states": [],
    "agent_properties": [],
    "agent_property_types": [],

    # episode 配置
    "max_steps": 96,
    "reward_key_contains": "accumulated_earnings",
    "reward_mode": "delta",

    # 让 Bptk 自动选可用的 ABM metrics（这是原始 EDT 示例的默认行为）
    "auto_select_abm_metrics": True,
}



class EDTDataProcessor:
    """
    Data processor for EDT serious game.

    用法示例（在 SeriousGame/run_edt.py 中）：
        raw = EDTDataProcessor.load_scenarios(args.scenario_path or None)
        test_samples = EDTDataProcessor.process_task_data(raw)
    """

    @staticmethod
    def load_scenarios(path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        加载原始 EDT 场景定义。

        - 如果提供了 path 且文件存在：
            * 支持 .json / .jsonl；
            * 预期文件内容是一个 list[dict] 或逐行 JSON，每条场景一个 dict。
        - 如果未提供 path 或文件不存在：
            * 使用内置默认场景：
                scenario_id = "interactive"
                config = EDT_FIXED_CONFIG
        """
        # 若用户提供了配置文件，则优先用文件
        if path and os.path.isfile(path):
            if path.endswith(".jsonl"):
                scenarios: List[Dict[str, Any]] = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        scenarios.append(json.loads(line))
                return scenarios
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    if "scenarios" in data and isinstance(data["scenarios"], list):
                        return data["scenarios"]
                    return [data]
                else:
                    raise ValueError(f"Unsupported EDT scenario file format: {path}")

        # 否则：构造一个默认的单一场景（固定 EDT_FIXED_CONFIG）
        default_scenario = {
            "scenario_id": "interactive",
            "config": dict(EDT_FIXED_CONFIG),
        }
        return [default_scenario]

    @staticmethod
    def process_task_data(raw_scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将原始场景列表归一化为：
            {"scenario_id": str, "config": dict}

        - 如果场景中已有 scenario_id / config 字段，则直接规范化；
        - 如果只有一个 dict，则把除 scenario_id 之外的字段都当作 config。
        """
        processed: List[Dict[str, Any]] = []
        for i, s in enumerate(raw_scenarios):
            if not isinstance(s, dict):
                raise ValueError(f"EDT scenario[{i}] must be a dict, got: {type(s)}")

            # 确保有 scenario_id
            if "scenario_id" not in s:
                s = dict(s)
                s["scenario_id"] = f"scenario_{i:04d}"

            scenario_id = str(s["scenario_id"])

            # 确保有 config
            if "config" in s and isinstance(s["config"], dict):
                cfg = dict(s["config"])
            else:
                cfg = {k: v for k, v in s.items() if k != "scenario_id"}

            # 如果调用方没有写全，就用 EDT_FIXED_CONFIG 补充
            merged_cfg = dict(EDT_FIXED_CONFIG)
            merged_cfg.update(cfg)

            processed.append({"scenario_id": scenario_id, "config": merged_cfg})

        return processed
