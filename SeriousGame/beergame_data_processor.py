#!/usr/bin/env python3
"""
Beer Game scenario data processor.

This mirrors BizBench's pattern:
- load raw data
- process_task_data -> normalized samples for agent.run()

A "test_sample" is treated as one full episode configuration.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


class BeerGameDataProcessor:
    DEFAULT_SCENARIOS: List[Dict[str, Any]] = [
        {
            "scenario_id": "default_typical",
            "config": {
                # 对齐 BeerGameConfig 字段
                "controlled_role": "retailer",
                "opponent_policy": "typical",   # 典型策略
                "horizon_weeks": 25,

                # 需求脚本：第一阶段 100，之后 400
                "demand_low": 100,
                "demand_high": 400,
                # 不显式设置 demand_step_week，使用 BeerGameConfig 默认值 2 即可

                "target_inventory": 400,
            },
        },
        {
            "scenario_id": "default_smoothing_4",
            "config": {
                "controlled_role": "retailer",
                # 平滑策略，窗口 T=4，通过策略名传递
                "opponent_policy": "smoothing:4",
                "horizon_weeks": 25,

                "demand_low": 100,
                "demand_high": 400,

                "target_inventory": 400,
                # smoothing_alpha 由策略名解析得到（pol.split(':')[1]），无需单独字段
            },
        },
    ]

    @staticmethod
    def load_scenarios(path: Optional[str]) -> List[Dict[str, Any]]:
        """
        Load scenarios from json / jsonl. If path is None, returns DEFAULT_SCENARIOS.
        """
        if not path:
            return list(BeerGameDataProcessor.DEFAULT_SCENARIOS)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Scenario file not found: {path}")

        scenarios: List[Dict[str, Any]] = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    scenarios.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    scenarios = obj
                else:
                    scenarios = obj.get("scenarios", [])

        if not scenarios:
            raise ValueError(f"No scenarios found in {path}")
        return scenarios

    @staticmethod
    def process_task_data(raw_scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scenario entries to:
          {"scenario_id": str, "config": dict}
        """
        processed: List[Dict[str, Any]] = []
        for i, s in enumerate(raw_scenarios):
            if "scenario_id" not in s:
                s = dict(s)
                s["scenario_id"] = f"scenario_{i:04d}"
            if "config" not in s:
                # allow passing a config dict directly
                s = {
                    "scenario_id": s["scenario_id"],
                    "config": {k: v for k, v in s.items() if k != "scenario_id"},
                }
            processed.append({"scenario_id": str(s["scenario_id"]), "config": dict(s["config"])})
        return processed
