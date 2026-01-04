"""
Debate agent adapter for StructuredReasoning.

The agent runs a debate-style generator (PRO/CON/JUDGE) and evaluates on test set
using the common utils.tools.evaluate_test_set pipeline.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.consulting_tools import evaluate_consulting_set


from utils.tools import evaluate_test_set
from utils.seriousgame_tools import (
    beergame_prepare_run,
    beergame_evaluate_run,
    beergame_save_run,
    beergame_build_query,
    beergame_base_rule_order,
    beergame_render_prompt,
    beergame_extract_order_and_note,
    edt_prepare_run,
    edt_evaluate_run,
    edt_save_run,
    build_edt_decision_context,
    render_edt_prompt,
    normalize_edt_schema,
)
from .generator import DebateGenerator, DebateConfig


class DebateAgent:
    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        agent_method: str = "debate",
        rounds: int = 1,
        pro_temperature: float = 0.0,
        con_temperature: float = 0.2,
        judge_temperature: float = 0.0,
    ):
        self.agent_method = agent_method
        self.max_tokens = max_tokens
        self.debate_cfg = DebateConfig(
            rounds=rounds,
            pro_temperature=pro_temperature,
            con_temperature=con_temperature,
            judge_temperature=judge_temperature,
        )
        self.generator = DebateGenerator(
            api_provider=api_provider,
            model_name=generator_model,
            max_tokens=max_tokens,
            debate_config=self.debate_cfg,
        )
        self.generator_client = self.generator.client
        self.temperature = 0.7

        # Consulting episode state
        self._current_case_id: Optional[str] = None
        self._dialogue_history: List[Dict[str, str]] = []

    def _call_llm_json(self, system: str, user: str) -> Dict[str, Any]:
        import json as _json

        resp = self.generator_client.chat.completions.create(
            model=self.generator.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=min(self.max_tokens, 512),
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text.startswith("{"):
            i = text.find("{")
            if i >= 0:
                text = text[i:]
        if not text.endswith("}"):
            j = text.rfind("}")
            if j >= 0:
                text = text[: j + 1]
        try:
            return _json.loads(text)
        except Exception:
            return {}

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        """Best-effort extraction of the first JSON object from free-form text."""
        if not isinstance(text, str):
            return "{}"
        t = text.strip()
        if not t:
            return "{}"
        if t.startswith("{") and t.endswith("}"):
            return t

        # locate first '{' and then find a matching closing brace
        start = t.find("{")
        if start < 0:
            return "{}"
        depth = 0
        for i in range(start, len(t)):
            ch = t[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return t[start : i + 1]
        # fallback: try trimming to last brace
        end = t.rfind("}")
        if end >= 0 and end > start:
            return t[start : end + 1]
        return "{}"

    def _safe_load_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(self._extract_first_json_object(text))
        except Exception:
            return {}

    def on_case_start(self, case_id: str) -> None:
        self._current_case_id = case_id

    def on_case_end(
        self,
        case_id: str,
        case_text: str,
        history: List[Dict[str, str]],
    ) -> None:
        _ = (case_id, case_text, history)
        self._current_case_id = None

    def reply(self, case_id: str, history: List[Dict[str, str]]) -> str:
        """
        Consulting candidate reply using debate workflow:
        PRO/CON rounds then JUDGE synthesis via DebateGenerator.
        """
        turns = sum(1 for h in history if h.get("role") == "candidate")

        # 最近 interviewer 提问
        last_interviewer_msg = ""
        for h in reversed(history):
            if h.get("role") == "interviewer":
                last_interviewer_msg = h.get("content", "")
                break

        transcript_lines = [
            f"{h.get('role', 'unknown')}: {h.get('content', '')}"
            for h in history
        ]
        transcript_text = "\n".join(transcript_lines) or "[no previous dialogue]"

        # 走 debate 生成链：question=最新提问，context=全量对话
        response_text, _, meta = self.generator.generate(
            question=last_interviewer_msg or "(Interviewer message missing.)",
            playbook="",
            context=transcript_text,
            reflection="(empty)",
            use_json_mode=True,
            call_id=f"consult_debate_{case_id}_t{turns}",
            log_dir=None,
        )

        # 解析 JSON 的 final_answer/reply，失败则回退原文本
        reply = None
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                reply = parsed.get("final_answer") or parsed.get("reply")
        except Exception:
            try:
                start = response_text.find("{")
                end = response_text.rfind("}")
                if 0 <= start < end:
                    parsed = json.loads(response_text[start : end + 1])
                    if isinstance(parsed, dict):
                        reply = parsed.get("final_answer") or parsed.get("reply")
            except Exception:
                reply = None

        if not isinstance(reply, str) or not reply.strip():
            reply = response_text.strip()
        if not reply:
            reply = (
                "Let me outline the key drivers, share a hypothesis, and the first "
                "analyses I'd run to validate it."
            )

        return reply.strip()

    # ==========================================================
    # BeerGame support: decision hook + evaluation entry point
    # ==========================================================

    def _decide_order_qty(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """
        BeerGame 单步决策：使用 Debate（PRO/CON/JUDGE 多轮）生成，JSON-only，失败回退基线。
        """
        role = str(ctx.get("role", obs.get("role", "retailer")))
        max_order_qty = int(getattr(self, "max_order_qty", 5000))

        base_order = beergame_base_rule_order(
            obs=obs,
            ctx=ctx,
            max_order_qty=max_order_qty,
        )

        system, user = beergame_render_prompt(
            role=role,
            obs=obs,
            retrieved="",  # debate 版本无记忆
            base_order=base_order,
        )

        question_text = (
            f"{system}\n\n{user}\n\n"
            "You must respond strictly in JSON ONLY, exactly in the form:\n"
            "{\n"
            '  \"order_qty\": <integer>,\n'
            '  \"note\": \"<brief rationale>\"\n'
            "}\n"
            "No other keys. No text before or after JSON. Do NOT reveal chain-of-thought."
        )

        response_text, _trace, _meta = self.generator.generate(
            question=question_text,
            playbook="",
            context="",
            reflection="",
            use_json_mode=True,
            call_id=f"beergame_debate_{ctx.get('scenario_id','')}_{ctx.get('episode_id','')}_w{obs.get('week','')}",
            log_dir=None,
        )

        order_qty, used_fallback = self._extract_order_qty_from_debate(
            response_text=response_text,
            base_order=base_order,
            max_order_qty=max_order_qty,
        )
        fallback_tag = " (fallback_base_order)" if used_fallback else ""
        self._last_beergame_note = f"debate_final{fallback_tag}: {response_text[:500]}"
        return int(order_qty)

    def _extract_order_qty_from_debate(
        self, response_text: str, base_order: int, max_order_qty: int
    ) -> tuple[int, bool]:
        """
        从 Debate 生成的 JSON 文本中提取订单；失败则回退 base_order，并返回是否回退。
        """
        candidate = None
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                for key in (
                    "order_qty",
                    "order",
                    "quantity",
                    "final_answer",
                    "reply",
                    "orderQty",
                    "decision",
                    "final_value",
                ):
                    val = data.get(key)
                    if isinstance(val, dict):
                        for subkey in ("order_qty", "order", "quantity", "value"):
                            subval = val.get(subkey)
                            if isinstance(subval, (int, float)):
                                candidate = int(subval)
                                break
                            if isinstance(subval, str) and subval.strip():
                                import re

                                m = re.search(r"-?\d+", subval)
                                if m:
                                    candidate = int(m.group(0))
                                    break
                        if candidate is not None:
                            break
                    if isinstance(val, (int, float)):
                        candidate = int(val)
                        break
                    if isinstance(val, str) and val.strip():
                        import re

                        m = re.search(r"-?\d+", val)
                        if m:
                            candidate = int(m.group(0))
                            break
        except Exception:
            try:
                start = response_text.find("{")
                end = response_text.rfind("}")
                if 0 <= start < end:
                    data = json.loads(response_text[start : end + 1])
                    if isinstance(data, dict):
                        for key in (
                            "order_qty",
                            "order",
                            "quantity",
                            "final_answer",
                            "reply",
                            "orderQty",
                            "decision",
                            "final_value",
                        ):
                            val = data.get(key)
                            if isinstance(val, dict):
                                for subkey in ("order_qty", "order", "quantity", "value"):
                                    subval = val.get(subkey)
                                    if isinstance(subval, (int, float)):
                                        candidate = int(subval)
                                        break
                                    if isinstance(subval, str) and subval.strip():
                                        import re

                                        m = re.search(r"-?\d+", subval)
                                        if m:
                                            candidate = int(m.group(0))
                                            break
                                if candidate is not None:
                                    break
                            if isinstance(val, (int, float)):
                                candidate = int(val)
                                break
                            if isinstance(val, str) and val.strip():
                                import re

                                m = re.search(r"-?\d+", val)
                                if m:
                                    candidate = int(m.group(0))
                                    break
            except Exception:
                candidate = None

        used_fallback = candidate is None
        if candidate is None:
            candidate = base_order
            print(f"[Debate][BeerGame] parse failed, fallback to base_order={base_order}")

        candidate = max(0, min(int(candidate), max_order_qty))
        return candidate, used_fallback

    def run_beergame(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """BeerGame 评测入口（委托 seriousgame_tools 统一逻辑）。"""
        _ = data_processor  # BeerGame 流程在 seriousgame_tools 中定义

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
    # Consulting: evaluation entry point
    # ==========================================================

    def run_consulting(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )
        if not test_samples:
            raise ValueError("Consulting requires non-empty test_samples")

        save_dir = config.get("save_dir", "results")
        task_name = config.get("task_name", "Consulting")
        run_subdir = (
            f"{task_name}/{self.agent_method}/{mode}/"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - CONSULTING EVALUATION")
        print(f"{'='*60}")
        print(f"Cases: {len(test_samples)}")
        print(f"Save dir: {resolved_save_path}")
        print(f"{'='*60}\n")

        results, error_log = evaluate_consulting_set(
            agent=self,
            test_samples=test_samples,
            config=config,
            log_dir=resolved_save_path,
        )

        with open(
            os.path.join(resolved_save_path, "test_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {"test_results": results, "error_log": error_log},
                f,
                indent=2,
            )

        config_payload = dict(config)
        config_payload.update(
            {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
                "debate_rounds": self.debate_cfg.rounds,
                "debate_pro_temperature": self.debate_cfg.pro_temperature,
                "debate_con_temperature": self.debate_cfg.con_temperature,
                "debate_judge_temperature": self.debate_cfg.judge_temperature,
            }
        )
        with open(
            os.path.join(resolved_save_path, "run_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config_payload, f, indent=2)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - CONSULTING RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Num cases: {results.get('num_cases')}, "
              f"finished: {results.get('num_finished')}, "
              f"failed: {results.get('num_failed')}")
        print(f"Metrics: {results.get('metrics')}")
        print(f"Results saved to: {resolved_save_path}")
        print(f"{'='*60}\n")

        return results

    # ==================================================================
    # EDT decision hook (required by utils.seriousgame_tools.evaluate_edt_set)
    # ==================================================================
    async def _decide_edt_scenario_schema(
        self,
        base_summary: Dict[str, Any],
        scenario_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Decide an EDT scenario schema using Debate's PRO/CON/JUDGE mechanism.

        This is the only agent-specific step in the EDT pipeline:
        seriousgame_tools will handle scenario generation, simulation, and scoring.
        """
        scenario_meta = scenario_meta or {}

        # 1) Build a robust, tool-layer context (incl. horizon inference, defaults).
        ctx = build_edt_decision_context(
            base_summary=base_summary,
            scenario_meta=scenario_meta,
            max_steps_hint=scenario_meta.get("max_steps"),
        )

        # 2) Render the standard EDT prompt (model-agnostic).
        system, user = render_edt_prompt(ctx)

        # 3) Enforce Debate output contract:
        #    the judge must return JSON with keys {reasoning, final_answer},
        #    and final_answer must be a JSON object schema: {"C":..., "R":..., "P":[...]}.
        user = (
            user
            + "\n\nIMPORTANT OUTPUT CONSTRAINTS:\n"
            "- Respond STRICTLY in JSON as required by the debate judge.\n"
            "- Set `final_answer` to a JSON OBJECT (not a string).\n"
            "- `final_answer` must contain ONLY keys: C, R, P.\n"
            "- Do not include extra commentary outside the JSON.\n"
        )

        final_text, _transcript, _trace = self.generator.generate(
            question=user,
            context=system,
            use_json_mode=False,
            call_id="edt",
        )

        envelope = self._safe_load_json(final_text)
        candidate = envelope.get("final_answer", envelope)

        if isinstance(candidate, str):
            candidate = self._safe_load_json(candidate)

        if not isinstance(candidate, dict):
            candidate = {}

        # 4) Normalize/clip to enforce non-degenerate constraints.
        return normalize_edt_schema(candidate, ctx)

    # ==================================================================
    # EDT run entry (align with other agents: Prepare / Evaluate / Save)
    # ==================================================================
    def run_edt(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        ctx = edt_prepare_run(mode=mode, test_samples=test_samples, config=config, allowed_modes=self.SUPPORTED_MODES)
        results, error_log = edt_evaluate_run(agent=self, test_samples=test_samples, config=config, ctx=ctx)
        edt_save_run(results=results, error_log=error_log, config=config, ctx=ctx)
        return results

    # -----------------------
    # BizBench / StructuredReasoning task runner (previously: run)
    # -----------------------
    def run_bizbench(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor,
        config: Dict[str, Any],
    ):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )
        if not test_samples:
            raise ValueError(f"{self.agent_method.upper()} agent requires test samples but none were provided.")

        task_name = str(config.get("task_name", getattr(data_processor, "task_name", ""))).lower()
        if "beer" in task_name:
            return self.run_beergame(mode, test_samples, data_processor, config)
        if "consult" in task_name:
            return self.run_consulting(mode, test_samples, data_processor, config)

        save_dir = config.get("save_dir")
        if not save_dir:
            raise ValueError(f"Configuration missing 'save_dir' for {self.agent_method}.")

        task_name_safe = task_name or "unknown_task"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name_safe, self.agent_method, mode, timestamp)
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - DEBATE EVALUATION")
        print(f"{'='*60}")
        print(f"Samples: {len(test_samples)}")
        print(f"Rounds: {self.debate_cfg.rounds}")
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
            json.dump({"test_results": results, "error_log": error_log}, f, indent=2)

        config_payload = dict(config)
        config_payload.update(
            {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
                "debate_rounds": self.debate_cfg.rounds,
                "debate_pro_temperature": self.debate_cfg.pro_temperature,
                "debate_con_temperature": self.debate_cfg.con_temperature,
                "debate_judge_temperature": self.debate_cfg.judge_temperature,
            }
        )
        with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Accuracy: {results.get('accuracy', 0.0):.3f}")
        print(f"Results saved to: {resolved_save_path}")
        print(f"{'='*60}\n")

        return results

    def run(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor: Any,
        config: Dict[str, Any],
    ):
        """Unified entry point. Routes to the correct task runner without modifying existing task logic."""
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )
        if not test_samples:
            raise ValueError(f"{self.agent_method.upper()} agent requires test samples but none were provided.")

        task_name = str(config.get("task_name", getattr(data_processor, "task_name", ""))).lower()

        # Keep existing tasks untouched; only add EDT routing.
        if "edt" in task_name:
            return self.run_edt(mode, test_samples, data_processor, config)
        if "consult" in task_name:
            return self.run_consulting(mode, test_samples, data_processor, config)

        # Default: StructuredReasoning / BizBench
        return self.run_bizbench(mode, test_samples, data_processor, config)


