"""Chain-of-Thought (CoT) baseline agent.

This agent is a lightweight, memory-free baseline.

Project convention:
- Consulting / SeriousGame tasks should be runnable by different agents while
  sharing task-native orchestration and prompt helpers in utils/.

This file implements Consulting support by delegating to utils/consulting_tools.py
for:
- candidate state extraction and prompt rendering
- evaluation orchestration (prepare / evaluate / save)

StructuredReasoning (BizBench-style) continues to use the generator-only workflow
(utils.tools.evaluate_test_set).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.llm import timed_llm_call
from utils.tools import evaluate_test_set
from utils.consulting_tools import (
    consulting_prepare_run,
    consulting_evaluate_run,
    consulting_save_run,
    consulting_build_candidate_query,
    consulting_render_candidate_prompt,
    consulting_extract_candidate_reply,
)


from utils.seriousgame_tools import (
    edt_prepare_run,
    edt_evaluate_run,
    edt_save_run,
    build_edt_decision_context,
    render_edt_prompt,
    normalize_edt_schema,
    beergame_prepare_run,
    beergame_evaluate_run,
    beergame_save_run,
    beergame_build_query,
    beergame_base_rule_order,
    beergame_render_prompt,
    beergame_extract_order_and_note,
)

import utils.seriousgame_tools as seriousgame_tools


from .generator import ChainOfThoughtGenerator


class ChainOfThoughtAgent:
    """Memory-free baseline agent."""

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        agent_method: str = "cot",
    ):
        self.agent_method = agent_method
        self.generator = ChainOfThoughtGenerator(
            api_provider=api_provider,
            model_name=generator_model,
            max_tokens=max_tokens,
        )
        self.generator_model = generator_model
        self.generator_client = self.generator.client
        self.max_tokens = max_tokens
        self.temperature = 0.7

    # ==========================================================
    # Unified entry point
    # ==========================================================

    def run(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch to a task-specific run_* function based on task_name."""
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )

        if not test_samples:
            raise ValueError(f"{self.agent_method.upper()} agent requires non-empty test_samples.")

        task_name = str(config.get("task_name", getattr(data_processor, "task_name", ""))).lower()

        if "consult" in task_name:
            return self.run_consulting(mode, test_samples, data_processor, config)

        if "beer" in task_name:
            return self.run_beergame(mode, test_samples, data_processor, config)

        if "edt" in task_name:
            return self.run_edt(mode, test_samples, data_processor, config)

        # Default: StructuredReasoning / BizBench style evaluation
        return self.run_bizbench(mode, test_samples, data_processor, config)

    # ==========================================================
    # StructuredReasoning / BizBench
    # ==========================================================

    def run_bizbench(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        save_dir = config.get("save_dir")
        if not save_dir:
            raise ValueError("Configuration missing 'save_dir' for ChainOfThoughtAgent.")

        task_name = str(config.get("task_name", getattr(data_processor, "task_name", "unknown_task")))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name, self.agent_method, mode, timestamp)
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - CHAIN-OF-THOUGHT EVALUATION")
        print(f"{'='*60}")
        print(f"Samples: {len(test_samples)}")
        print(f"Log dir: {log_dir}")
        print(f"{'='*60}\n")

        results, error_log = evaluate_test_set(
            data_processor=data_processor,
            generator=self.generator,
            playbook="",
            test_samples=test_samples,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            max_workers=config.get("test_workers", 20),
            use_json_mode=config.get("json_mode", False),
        )

        with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump({"test_results": results, "error_log": error_log}, f, indent=2, ensure_ascii=False)

        config_payload = dict(config)
        config_payload["run_subdir"] = run_subdir
        config_payload["resolved_save_path"] = resolved_save_path

        with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - RUN COMPLETE")
        print(f"{'='*60}")
        if isinstance(results, dict) and "accuracy" in results:
            print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Results saved to: {resolved_save_path}")
        print(f"{'='*60}\n")

        return results

    # ==========================================================
    # Consulting support: required hooks
    # ==========================================================

    def on_case_start(self, case_id: str) -> None:
        """Required by utils.consulting_tools. CoT baseline keeps no memory."""
        _ = case_id
        return None

    def on_case_end(self, case_id: str, case_text: str, history: List[Dict[str, str]]) -> None:
        """Required by utils.consulting_tools. CoT baseline keeps no memory."""
        _ = (case_id, case_text, history)
        return None

    def _call_llm_json(self, *, system: str, user: str, call_id: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Call the model with JSON mode; parse robustly."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_text, _call_info = timed_llm_call(
            self.generator_client,
            self.generator.api_provider,
            self.generator.model,
            prompt=user,
            role="cot_candidate",
            call_id=call_id,
            max_tokens=min(self.max_tokens, max_tokens),
            use_json_mode=True,
            temperature=self.temperature,
            messages=messages,
        )

        text = (response_text or "").strip()
        if not text:
            return {}

        # In case the backend ignores json_mode and wraps extra text, try to salvage JSON.
        if not text.startswith("{"):
            i = text.find("{")
            if i >= 0:
                text = text[i:]
        if not text.endswith("}"):
            j = text.rfind("}")
            if j >= 0:
                text = text[: j + 1]

        try:
            return json.loads(text)
        except Exception:
            return {}

    def reply(self, case_id: str, history: List[Dict[str, str]]) -> str:
        """Candidate reply for Consulting (called by utils.consulting_tools)."""
        state = consulting_build_candidate_query(case_id=case_id, history=history)
        system, user = consulting_render_candidate_prompt(
            case_id=case_id,
            state=state,
            retrieved="",  # CoT baseline has no memory.
        )

        # CoT augmentation: instruct internal step-by-step planning, but do not reveal it.
        user = (
            user
            + "\n\nBefore responding, think step-by-step in private to plan a structured consulting answer. "
            "Do NOT reveal your chain-of-thought. Output ONLY a JSON object {\"reply\": \"...\"}."
        )

        data = self._call_llm_json(
            system=system,
            user=user,
            call_id=f"test_cot_consulting_{case_id}_t{int(state.get('turns', 0) or 0)}",
            max_tokens=512,
        )
        return consulting_extract_candidate_reply(data)


    # ==========================================================
    # BeerGame support: required hook + evaluation entry point
    # ==========================================================

    def _decide_order_qty(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """BeerGame single-step decision hook.

        Called by utils.seriousgame_tools.beergame_evaluate_run via policy_fn.
        CoT baseline has no memory: `retrieved` is always empty.
        """
        role = str(ctx.get("role", obs.get("role", "retailer")))
        # max_order_qty can be set by run_beergame() before evaluation starts
        max_order_qty = int(getattr(self, "max_order_qty", 5000))

        query = beergame_build_query(obs)
        _ = query  # reserved for potential logging / future memoryless retrieval

        base_order = beergame_base_rule_order(
            obs=obs,
            ctx=ctx,
            max_order_qty=max_order_qty,
        )

        system, user = beergame_render_prompt(
            role=role,
            obs=obs,
            retrieved="",
            base_order=base_order,
        )

        # CoT augmentation: put it in USER message (do not modify system guardrails)
        user = (
            user
            + "\n\nCoT instruction: Think step-by-step in private to select the best order quantity. "
              "Do NOT reveal your chain-of-thought. "
        )

        js = self._call_llm_json(
            system=system,
            user=user,
            call_id=f"test_cot_beergame_{ctx.get('scenario_id','')}_{ctx.get('episode_id','')}_w{obs.get('week','')}",
            max_tokens=256,
        )
        order_qty, note = beergame_extract_order_and_note(
            js=js,
            base_order=base_order,
            max_order_qty=max_order_qty,
        )
        self._last_beergame_note = note
        return int(order_qty)

    def run_beergame(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """BeerGame evaluation entry point (delegates to utils.seriousgame_tools)."""
        _ = data_processor  # BeerGame flow is defined in utils.seriousgame_tools.

        # Allow either top-level max_order_qty or config["beergame"]["max_order_qty"].
        beergame_cfg = dict(config.get("beergame", {}) or {})
        self.max_order_qty = int(
            beergame_cfg.get("max_order_qty", config.get("max_order_qty", 5000))
        )

        ctx = beergame_prepare_run(
            mode=mode,
            test_samples=test_samples,
            config=config,
            allowed_modes=self.SUPPORTED_MODES,
            agent_method=self.agent_method,
        )
        results, error_log = beergame_evaluate_run(
            agent=self,
            test_samples=test_samples,
            config=config,
            ctx=ctx,
        )
        beergame_save_run(
            results=results,
            error_log=error_log,
            config=config,
            ctx=ctx,
        )
        return results

    # ==========================================================
    # EDT support: required hooks + evaluation entry point
    # ==========================================================
    async def _decide_edt_scenario_schema(
            self,
            base_summary: Dict[str, Any],
            scenario_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # 1) 构造通用 ctx（含 horizon 兜底推断、default_P 等）
        ctx = build_edt_decision_context(
            base_summary=base_summary,
            scenario_meta=scenario_meta,
            max_steps_hint=(scenario_meta or {}).get("max_steps"),
        )

        # 2) prompt 也是通用逻辑（模型无关）
        system, user = render_edt_prompt(ctx)

        user = (
                user
                + "\n\n"
                + "Think step-by-step internally to derive the best business decision. "
                  "Do NOT reveal your chain-of-thought. "
                  "Output strictly valid JSON."
        )

        # 3) 仅这一句是模型相关：调用 LLM 返回 JSON
        response = self.generator_client.chat.completions.create(
            model=self.generator_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        text = response.choices[0].message.content
        if not text.startswith("{"):
            i = text.find("{")
            if i >= 0:
                text = text[i:]
        if not text.endswith("}"):
            j = text.rfind("}")
            if j >= 0:
                text = text[: j + 1]
        raw = json.loads(text)
        # 4) 归一化/防退化也是通用逻辑（模型无关）
        return normalize_edt_schema(raw, ctx)

    def run_edt(self, mode, test_samples, data_processor, config):
        ctx = edt_prepare_run(
            mode=mode,
            test_samples=test_samples,
            config=config,
            allowed_modes=self.SUPPORTED_MODES,
        )
        results, error_log = edt_evaluate_run(
            agent=self,
            test_samples=test_samples,
            config=config,
            ctx=ctx,
        )
        edt_save_run(
            results=results,
            error_log=error_log,
            config=config,
            ctx=ctx,
        )
        return results

    # ==========================================================
    # Consulting: evaluation entry point
    # ==========================================================

    def run_consulting(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Consulting evaluation entry point (delegates to utils.consulting_tools)."""
        _ = data_processor  # Consulting flow is defined in utils.consulting_tools.

        ctx = consulting_prepare_run(
            mode=mode,
            test_samples=test_samples,
            config=config,
            allowed_modes=self.SUPPORTED_MODES,
            agent_method=self.agent_method,
        )
        results, error_log = consulting_evaluate_run(
            agent=self,
            test_samples=test_samples,
            config=config,
            ctx=ctx,
        )
        consulting_save_run(
            results=results,
            error_log=error_log,
            config=config,
            ctx=ctx,
        )
        return results
