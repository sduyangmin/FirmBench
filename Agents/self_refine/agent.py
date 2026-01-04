"""
Self-refine agent that mirrors the BizBench agent interface.

The agent performs an initial generation followed by iterative self-feedback
refinement steps, reusing the common evaluate_test_set utility for scoring.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

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
    build_edt_decision_context,
    render_edt_prompt,
    normalize_edt_schema,
    edt_prepare_run,
    edt_evaluate_run,
    edt_save_run,
)
from .generator import SelfRefineGenerator


class SelfRefineAgent:
    """
    Minimal self-refine agent that keeps parity with the ACE/COT interface.
    Only online/eval_only modes are supported because the workflow is
    generation-only (no training loop).
    """

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        refine_rounds: int = 2,
        initial_temperature: float = 0.0,
        feedback_temperature: float = 0.2,
        agent_method: str = "self_refine",
    ):
        self.agent_method = agent_method
        self.max_tokens = max_tokens
        self.refine_rounds = refine_rounds
        self.generator = SelfRefineGenerator(
            api_provider=api_provider,
            model_name=generator_model,
            max_tokens=max_tokens,
            refine_rounds=refine_rounds,
            initial_temperature=initial_temperature,
            feedback_temperature=feedback_temperature,
        )
        self.generator_client = self.generator.init_role.client
        self.temperature = 0.7

    # ==========================================================
    # EDT helpers
    # ==========================================================
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
        end = t.rfind("}")
        if end >= 0 and end > start:
            return t[start : end + 1]
        return "{}"

    def _safe_load_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(self._extract_first_json_object(text))
        except Exception:
            return {}

    def _build_edt_self_refine_feedback_prompt(
            self,
            *,
            ctx: Dict[str, Any],
            schema: Dict[str, Any],
    ) -> Tuple[str, str]:
        system = (
            "You are a strict schema reviewer for an enterprise digital twin (EDT) scenario design.\n"
            "You must audit whether the proposed schema is actionable, consistent, and respects constraints.\n"
            "Focus on: (1) controllable decisions only, (2) internal consistency across C/R/P, (3) missing fields, "
            "(4) unrealistic or invalid values, (5) budget/utilization tradeoffs implied by the context.\n"
            "Do NOT propose a completely new schema. Provide feedback for improving the current schema.\n"
            "Return ONLY JSON: {\"is_valid\": true/false, \"feedback\": \"...\"}.\n"
            "feedback must be concise (<= 8 sentences) and directly actionable."
        )

        # keep context compact to avoid bloating
        ctx_compact = {
            "base_scenario": ctx.get("base_scenario"),
            "company": ctx.get("company"),
            "constraints": ctx.get("constraints"),
            "decision_space": ctx.get("decision_space"),
            "notes": ctx.get("notes"),
        }

        user = (
            "EDT decision context (abridged):\n"
            f"{json.dumps(ctx_compact, ensure_ascii=False)[:3000]}\n\n"
            "Proposed schema (C/R/P):\n"
            f"{json.dumps(schema, ensure_ascii=False)[:2000]}\n\n"
            "Audit this schema and output JSON {is_valid, feedback}."
        )
        return system, user

    def _build_edt_self_refine_iterate_prompt(
            self,
            *,
            ctx: Dict[str, Any],
            schema: Dict[str, Any],
            feedback_json: Dict[str, Any],
    ) -> Tuple[str, str]:
        system = (
            "You are improving an EDT scenario design schema.\n"
            "Rewrite the schema to address the review feedback.\n"
            "Hard constraints:\n"
            "- Output ONLY one JSON object.\n"
            "- The JSON must contain ONLY keys: C, R, P.\n"
            "- Do NOT include markdown fences or extra text.\n"
            "- Keep changes minimal but effective; do not invent uncontrollable fields.\n"
        )

        ctx_compact = {
            "base_scenario": ctx.get("base_scenario"),
            "company": ctx.get("company"),
            "constraints": ctx.get("constraints"),
            "decision_space": ctx.get("decision_space"),
        }

        feedback_text = feedback_json.get("feedback", "")
        user = (
            "EDT decision context (abridged):\n"
            f"{json.dumps(ctx_compact, ensure_ascii=False)[:3000]}\n\n"
            "Current schema (C/R/P):\n"
            f"{json.dumps(schema, ensure_ascii=False)[:2000]}\n\n"
            "Reviewer feedback:\n"
            f"{feedback_text}\n\n"
            "Now output the improved schema JSON with ONLY keys C, R, P."
        )
        return system, user

    # ==========================================================
    # EDT decision hook (required by utils.seriousgame_tools)
    # ==========================================================
    async def _decide_edt_scenario_schema(
            self,
            base_summary: Dict[str, Any],
            scenario_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scenario_meta = scenario_meta or {}

        ctx = build_edt_decision_context(
            base_summary=base_summary,
            scenario_meta=scenario_meta,
            max_steps_hint=scenario_meta.get("max_steps"),
        )

        # --------- Step 1: initial schema generation (single shot) ---------
        system, user = render_edt_prompt(ctx)
        user = (
                user
                + "\n\nIMPORTANT OUTPUT CONSTRAINTS:\n"
                + "- Output ONLY a single JSON object.\n"
                + "- The JSON object must contain ONLY keys: C, R, P.\n"
                + "- Do not wrap JSON in markdown fences.\n"
        )

        resp = self.generator_client.chat.completions.create(
            model=self.generator.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=min(self.max_tokens, 512),
        )
        text = (resp.choices[0].message.content or "").strip()
        raw = self._safe_load_json(text)
        schema = raw.get("final_answer", raw)
        if isinstance(schema, str):
            schema = self._safe_load_json(schema)
        if not isinstance(schema, dict):
            schema = {}
        schema = normalize_edt_schema(schema, ctx)

        # --------- Step 2: self-refine loop (feedback -> iterate) ---------
        rounds = int(getattr(self, "refine_rounds", 0) or 0)
        rounds = max(0, rounds)

        for r in range(rounds):
            fb_sys, fb_user = self._build_edt_self_refine_feedback_prompt(ctx=ctx, schema=schema)
            fb_resp = self.generator_client.chat.completions.create(
                model=self.generator.model,
                messages=[{"role": "system", "content": fb_sys}, {"role": "user", "content": fb_user}],
                temperature=0.2,
                max_tokens=min(self.max_tokens, 256),
            )
            fb_text = (fb_resp.choices[0].message.content or "").strip()
            fb_json = self._safe_load_json(fb_text)

            # If reviewer says valid, early stop (self-refine’s early-stop spirit)
            if isinstance(fb_json, dict) and fb_json.get("is_valid") is True:
                break

            it_sys, it_user = self._build_edt_self_refine_iterate_prompt(ctx=ctx, schema=schema,
                                                                         feedback_json=fb_json or {})
            it_resp = self.generator_client.chat.completions.create(
                model=self.generator.model,
                messages=[{"role": "system", "content": it_sys}, {"role": "user", "content": it_user}],
                temperature=0.0,
                max_tokens=min(self.max_tokens, 512),
            )
            it_text = (it_resp.choices[0].message.content or "").strip()
            it_raw = self._safe_load_json(it_text)
            new_schema = it_raw.get("final_answer", it_raw)
            if isinstance(new_schema, str):
                new_schema = self._safe_load_json(new_schema)
            if not isinstance(new_schema, dict):
                new_schema = {}

            schema = normalize_edt_schema(new_schema, ctx)

        return schema

    # ==========================================================
    # EDT evaluation entry (single-pass; uses seriousgame_tools pipeline)
    # ==========================================================
    def run_edt(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        _ = data_processor
        ctx = edt_prepare_run(
            mode=mode,
            test_samples=test_samples,
            config=config,
            allowed_modes=self.SUPPORTED_MODES,
        )
        results, error_log = edt_evaluate_run(agent=self, test_samples=test_samples, config=config, ctx=ctx)
        edt_save_run(results=results, error_log=error_log, config=config, ctx=ctx)
        return results

    # ==========================================================
    # Consulting support: on_case_start / reply / on_case_end
    # ==========================================================

    def _call_llm_json(self, system: str, user: str) -> Dict[str, Any]:
        """
        Lightweight JSON helper for consulting chat turns.
        """
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

    def on_case_start(self, case_id: str) -> None:
        """
        Hook at the beginning of a consulting case. No cross-case memory kept.
        """
        self._current_case_id = case_id

    def on_case_end(
        self,
        case_id: str,
        case_text: str,
        history: List[Dict[str, str]],
    ) -> None:
        """
        Hook when a consulting case finishes. No persistence for baseline.
        """
        _ = (case_id, case_text, history)
        self._current_case_id = None

    def reply(self, case_id: str, history: List[Dict[str, str]]) -> str:
        """
        Consulting candidate reply using self-refine workflow:
        initial answer -> self feedback -> rewrite (until rounds or early stop).
        """
        turns = sum(1 for h in history if h.get("role") == "candidate")

        # 最近 interviewer 问题
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

        # 直接复用自反思生成链：question=最新面试官问题，context=全量对话
        response_text, _, meta = self.generator.generate(
            question=last_interviewer_msg or "(Interviewer message missing.)",
            playbook="",
            context=transcript_text,
            reflection="(empty)",
            use_json_mode=True,
            call_id=f"consult_sf_{case_id}_t{turns}",
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
                "Let me structure the key drivers, share a hypothesis, and outline the "
                "first analyses I'd run to validate it."
            )

        return reply.strip()

    # ==========================================================
    # BeerGame: decision hook + evaluation entry (memory-free)
    # ==========================================================
    def _decide_order_qty(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """
        BeerGame 单步决策：使用 self-refine 的 init/feedback/iterate，多轮自我改写，
        但保持无记忆、JSON-only 输出与基线相同的安全兜底。
        """
        role = str(ctx.get("role", obs.get("role", "retailer")))
        max_order_qty = int(getattr(self, "max_order_qty", 5000))

        _ = beergame_build_query(obs)  # 预留检索/日志

        base_order = beergame_base_rule_order(
            obs=obs,
            ctx=ctx,
            max_order_qty=max_order_qty,
        )

        system, user = beergame_render_prompt(
            role=role,
            obs=obs,
            retrieved="",  # self-refine baseline 无记忆
            base_order=base_order,
        )

        # 将 BeerGame 提示拼成 self-refine 的 question，保持 JSON-only 约束
        question_text = (
            f"{system}\n\n{user}\n\n"
            "Respond strictly in JSON with keys reasoning and final_answer (the order quantity as integer). "
            "Do NOT reveal chain-of-thought."
        )

        response_text, _trace, _meta = self.generator.generate(
            question=question_text,
            playbook="",
            context="",
            reflection="",
            use_json_mode=True,
            call_id=f"beergame_selfrefine_{ctx.get('scenario_id','')}_{ctx.get('episode_id','')}_w{obs.get('week','')}",
            log_dir=None,
        )

        order_qty = self._extract_order_qty_from_sf(
            response_text=response_text,
            base_order=base_order,
            max_order_qty=max_order_qty,
        )
        # note 兼容已有字段：记录最终使用的解释
        self._last_beergame_note = f"self_refine_final: {response_text[:500]}"
        return int(order_qty)

    def _extract_order_qty_from_sf(
        self, response_text: str, base_order: int, max_order_qty: int
    ) -> int:
        """
        从 self-refine 生成的 JSON 文本中提取订单；失败则回退 base_order。
        """
        candidate = None
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                for key in ("order_qty", "order", "quantity", "final_answer", "reply"):
                    val = data.get(key)
                    if isinstance(val, (int, float)):
                        candidate = int(val)
                        break
                    if isinstance(val, str) and val.strip():
                        # 尝试从字符串中提取整数
                        import re

                        m = re.search(r"-?\d+", val)
                        if m:
                            candidate = int(m.group(0))
                            break
        except Exception:
            # 尝试截取花括号后再解析
            try:
                start = response_text.find("{")
                end = response_text.rfind("}")
                if 0 <= start < end:
                    data = json.loads(response_text[start : end + 1])
                    if isinstance(data, dict):
                        for key in ("order_qty", "order", "quantity", "final_answer", "reply"):
                            val = data.get(key)
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

        if candidate is None:
            candidate = base_order

        # clamp
        candidate = max(0, min(int(candidate), max_order_qty))
        return candidate

    def run_beergame(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """BeerGame 评测入口（委托 seriousgame_tools 通用流程）。"""
        _ = data_processor

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
        """
        Consulting / case-interview evaluation entry for the self-refine baseline.
        """
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
                "self_refine_rounds": self.refine_rounds,
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

    # ==========================================================
    # Unified run with consulting routing
    # ==========================================================
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
            raise ValueError("Configuration missing 'save_dir' for SelfRefineAgent.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name or "unknown_task", self.agent_method, mode, timestamp)
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - SELF-REFINE EVALUATION")
        print(f"{'='*60}")
        print(f"Samples: {len(test_samples)}")
        print(f"Refinement rounds: {self.refine_rounds}")
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
                "self_refine_rounds": self.refine_rounds,
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

    # ==========================================================
    # Unified entry point (router)
    # ==========================================================
    def run(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor,
        config: Dict[str, Any],
    ):
        """
        Route evaluation by dataset.

        - EDT: run_edt
        - Otherwise: run_bizbench (original SelfRefine behavior, including Consulting routing)
        """
        task_name = str(config.get("task_name", getattr(data_processor, "task_name", ""))).lower()
        if "edt" in task_name:
            return self.run_edt(
                mode=mode,
                test_samples=test_samples or [],
                data_processor=data_processor,
                config=config,
            )
        return self.run_bizbench(mode=mode, test_samples=test_samples, data_processor=data_processor, config=config)
