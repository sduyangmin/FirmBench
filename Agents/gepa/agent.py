from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import replace

from utils.consulting_tools import evaluate_consulting_set
from utils.tools import extract_answer, initialize_clients
from utils.llm import timed_llm_call
from utils.seriousgame_tools import (
    beergame_prepare_run,
    beergame_evaluate_run,
    beergame_save_run,
    beergame_build_query,
    beergame_base_rule_order,
    beergame_render_prompt,
    beergame_extract_order_and_note,
    build_edt_decision_context,
    edt_evaluate_run,
    edt_prepare_run,
    edt_save_run,
    normalize_edt_schema,
    render_edt_prompt,
)

from .config import GepaConfig
from .core import GepaResult, _build_query, run_gepa
from .prompts import SEED_PROMPT


class GEPAAgent:
    """GEPA prompt 演化 agent。支持 online / eval_only / offline。"""

    SUPPORTED_MODES = {"online", "eval_only", "offline"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        max_tokens: int,
        agent_method: str = "gepa",
        gepa_config: Optional[GepaConfig] = None,
    ):
        self.agent_method = agent_method
        self.api_provider = api_provider
        self.generator_model = generator_model
        self.reflector_model = reflector_model
        self.max_tokens = max_tokens
        self.cfg = gepa_config or GepaConfig()
        if self.cfg.max_tokens != max_tokens:
            # 保持与全局 max_tokens 一致
            self.cfg.max_tokens = max_tokens

        # 复用全局客户端初始化
        self.generator_client, self.reflector_client, _ = initialize_clients(api_provider)
        self.temperature = 0.7
        # BeerGame prompt state (selected via prompt-search in run_beergame)
        self._beergame_policy_prompt: Optional[str] = None
        self._beergame_best_prompt_path: Optional[str] = None
        self._beergame_log_dir: Optional[str] = None

    def _split_eval_only(
        self, samples: List[Dict[str, Any]], ratio: float, mini_batch_size: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """将测试样本拆为 Dfeedback (ratio) 与 Dpareto (其余)。"""
        if not samples:
            return [], []
        shuffled = list(samples)
        random.shuffle(shuffled)
        n_feedback = max(int(len(shuffled) * ratio), mini_batch_size)
        n_feedback = min(n_feedback, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
        Dfeedback = shuffled[:n_feedback]
        Dpareto = shuffled[n_feedback:] if n_feedback < len(shuffled) else shuffled
        return Dpareto, Dfeedback

    def _evaluate_with_prompt(
        self,
        prompt: str,
        samples: List[Dict[str, Any]],
        data_processor,
        log_dir: str,
        max_workers: int,
        use_json_mode: bool,
        task_name: str = "",
        use_accuracy: bool = False,
    ) -> Dict[str, Any]:
        """使用给定 prompt 评测完整样本集，返回 accuracy 与错误列表。"""
        def _eval_single(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
            contents = _build_query(sample, prompt)
            resp, _ = timed_llm_call(
                self.generator_client,
                api_provider=self.api_provider,
                model=self.generator_model,
                prompt=contents,
                role="generator",
                call_id=f"gepa_final_{idx}",
                max_tokens=self.max_tokens,
                use_json_mode=False,  # 生成端固定不使用 JSON，避免结构化错误响应
                temperature=self.cfg.target_temperature,
            )
            # 尽量解析最终答案；若解析失败则回退原始响应，避免统一变成 "No final answer found"
            if hasattr(data_processor, "extract_answer_from_response"):
                final_answer = data_processor.extract_answer_from_response(resp)
            else:
                final_answer = extract_answer(resp)
                if final_answer == "No final answer found":
                    final_answer = resp
            is_correct = data_processor.answer_is_correct(final_answer, sample.get("target", ""))
            return {
                "index": idx,
                "final_answer": final_answer,
                "target": sample.get("target", ""),
                "is_correct": is_correct,
            }

        results: List[Dict[str, Any]] = []
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_eval_single, i, s): i for i, s in enumerate(samples)}
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x["index"])

        accuracy = (
            sum(1 for r in results if r["is_correct"]) / len(results) if results else 0.0
        )
        errors = [
            {"index": r["index"], "prediction": r["final_answer"], "ground_truth": r["target"]}
            for r in results
            if not r["is_correct"]
        ]
        return {
            "accuracy": accuracy,
            "total": len(results),
            "correct": len(results) - len(errors),
            "errors": errors,
        }

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

    async def _decide_edt_scenario_schema(
        self,
        base_summary: Dict[str, Any],
        scenario_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """EDT decision hook.

        Must return a JSON-compatible schema with keys {C, R, P}.
        Simulation, scoring, and logging are handled by utils.seriousgame_tools.

        Non-online version: use current seed prompt to guide schema decision.
        """
        scenario_meta = scenario_meta or {}
        ctx = build_edt_decision_context(
            base_summary=base_summary,
            scenario_meta=scenario_meta,
            max_steps_hint=scenario_meta.get("max_steps"),
        )

        system, user = render_edt_prompt(ctx)
        policy_prompt = self.cfg.seed_prompt or SEED_PROMPT

        # Strengthen JSON-only constraint to reduce parsing failures.
        full_prompt = (
            f"{system}\n\n"
            f"POLICY GUIDELINES (follow these, but still output ONLY JSON):\n{policy_prompt}\n\n"
            f"{user}\n\n"
            "IMPORTANT OUTPUT CONSTRAINTS:\n"
            "- Output ONLY a single JSON object.\n"
            "- The JSON object must contain ONLY keys: C, R, P.\n"
            "- Do not wrap JSON in markdown fences.\n"
        )

        resp, _ = timed_llm_call(
            self.generator_client,
            api_provider=self.api_provider,
            model=self.generator_model,
            prompt=full_prompt,
            role="generator",
            call_id="gepa_edt_schema",
            max_tokens=min(self.max_tokens, 512),
            use_json_mode=False,
            temperature=self.cfg.target_temperature,
        )

        raw = self._safe_load_json(resp)
        # Some generators may return an envelope like {"final_answer": {...}}
        candidate = raw.get("final_answer", raw)
        if isinstance(candidate, str):
            candidate = self._safe_load_json(candidate)
        if not isinstance(candidate, dict):
            candidate = {}
        return normalize_edt_schema(candidate, ctx)

    # ==========================================================
    # Consulting support
    # ==========================================================

    def _call_llm_json(self, system: str, user: str) -> Dict[str, Any]:
        """
        JSON-only LLM call wrapper with retries (timed_llm_call) to avoid timeouts.
        """
        import json as _json

        call_id = "test_gepa_beergame_json"
        try:
            resp_text, _ = timed_llm_call(
                self.generator_client,
                api_provider=self.api_provider,
                model=self.generator_model,
                prompt="",
                role="gepa_beergame",
                call_id=call_id,
                max_tokens=min(int(self.max_tokens), 512),
                log_dir=self._beergame_log_dir,
                use_json_mode=True,
                temperature=float(getattr(self, "temperature", 0.7)),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception:
            return {}

        text = (resp_text or "").strip()
        if not text:
            return {}
        if not text.startswith("{"):
            i = text.find("{")
            if i >= 0:
                text = text[i:]
        if not text.endswith("}"):
            j = text.rfind("}")
            if j >= 0:
                text = text[: j + 1]
        try:
            obj = _json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _extract_order_qty_from_gepa(
        js: Dict[str, Any],
        *,
        base_order: int,
        max_order_qty: int,
        ctx: Dict[str, Any],
    ) -> Tuple[int, str]:
        """
        Robust extraction with explicit fallback printing.
        Expected: {"order_qty": <int>, "note": "<str>"} but tolerate minor schema drift.
        """
        note = ""
        try:
            if isinstance(js, dict):
                note = js.get("note", "") if isinstance(js.get("note", ""), str) else ""
        except Exception:
            note = ""

        order_raw: Any = base_order
        if isinstance(js, dict):
            for k in ("order_qty", "order", "quantity", "decision", "final_value"):
                if k in js:
                    order_raw = js.get(k)
                    break

        order_qty = base_order
        ok = True
        try:
            order_qty = int(order_raw)
        except Exception:
            ok = False
            order_qty = int(base_order)

        order_qty = max(0, min(int(max_order_qty), int(order_qty)))
        if not ok:
            print(
                f"[GEPA][BeerGame] parse failed -> fallback base_order={int(base_order)} "
                f"(scenario_id={ctx.get('scenario_id','')}, episode_id={ctx.get('episode_id','')}, week={ctx.get('week', '')})",
                flush=True,
            )
        return int(order_qty), (note or "").strip()

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
        Consulting reply via GEPA-style prompt: use current best prompt (seed)
        to answer latest interviewer question conditioned on full transcript.
        """
        turns = sum(1 for h in history if h.get("role") == "candidate")

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

        prompt = self.cfg.seed_prompt or SEED_PROMPT
        sample = {"question": last_interviewer_msg, "context": transcript_text}
        contents = _build_query(sample, prompt)

        reply = ""
        try:
            resp, _ = timed_llm_call(
                self.generator_client,
                api_provider=self.api_provider,
                model=self.generator_model,
                prompt=contents,
                role="gepa_consult_reply",
                call_id=f"gepa_consult_{case_id}_t{turns}",
                max_tokens=self.max_tokens,
                use_json_mode=False,
                temperature=self.cfg.target_temperature,
            )
            reply = resp
        except Exception:
            reply = ""

        if not isinstance(reply, str) or not reply.strip():
            reply = (
                "Let me structure the issues, propose a hypothesis, and outline "
                "the first analyses I'd run."
            )
        return reply.strip()

    # ==========================================================
    # BeerGame: decision hook + evaluation entry
    # ==========================================================
    def _decide_order_qty(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """
        BeerGame 单步决策：使用 run_beergame 选出的 best prompt（若有），否则回退 seed prompt。
        """
        role = str(ctx.get("role", obs.get("role", "retailer")))
        max_order_qty = int(getattr(self, "max_order_qty", 5000))

        _ = beergame_build_query(obs)  # 预留日志/扩展

        # expose week for fallback logging
        try:
            ctx = dict(ctx)
            ctx["week"] = obs.get("week")
        except Exception:
            pass

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

        policy_prompt = (
            self._beergame_policy_prompt
            or (self.cfg.seed_prompt or "")
            or "You are a supply-chain decision agent. Minimize long-run cost (inventory + backlog) and avoid bullwhip."
        )

        user = (
            user
            + "\n\n[Policy prompt]\n"
            + str(policy_prompt).strip()
            + "\n\nThink step-by-step privately to choose the best order quantity. "
              "Do NOT reveal your chain-of-thought. "
              "Return ONLY JSON: {\"order_qty\": <int>, \"note\": \"<short>\"}."
        )

        js = self._call_llm_json(system=system, user=user)
        order_qty, note = self._extract_order_qty_from_gepa(
            js,
            base_order=base_order,
            max_order_qty=max_order_qty,
            ctx=ctx,
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
        """
        BeerGame 评测入口：先用前 K 个 episode 做 prompt search（按成本选择 best prompt），
        再用 best prompt 跑完整评测集。
        """
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
        self._beergame_log_dir = ctx.get("log_dir")

        # -------------------------
        # Load/persist best prompt
        # -------------------------
        save_dir = str(config.get("save_dir", "results"))
        best_prompt_path = str(
            beergame_cfg.get("best_prompt_path")
            or os.path.join(save_dir, "gepa_beergame_best_prompt.txt")
        )
        self._beergame_best_prompt_path = best_prompt_path
        load_best = bool(beergame_cfg.get("load_best_prompt", True))
        if load_best and os.path.exists(best_prompt_path):
            try:
                with open(best_prompt_path, "r", encoding="utf-8") as f:
                    loaded = f.read().strip()
                if loaded:
                    self._beergame_policy_prompt = loaded
            except Exception:
                pass

        # -------------------------
        # Prompt search (early episodes)
        # -------------------------
        prompt_search_episodes = int(beergame_cfg.get("prompt_search_episodes", 2))
        do_search = (
            prompt_search_episodes > 0
            and len(test_samples) > 1
            and not self._beergame_policy_prompt  # if already loaded, skip by default
        )

        if do_search:
            seed = (
                str(beergame_cfg.get("seed_prompt") or "").strip()
                or str(self.cfg.seed_prompt or "").strip()
                or "You are a supply-chain decision agent. Minimize long-run cost (inventory + backlog) and avoid bullwhip."
            )
            candidates: List[str] = []
            candidates.append(seed)
            candidates.append(
                seed
                + "\n\nHeuristic: use inventory position inv_pos = inventory + supply_line - backorder. "
                  "Order toward target_inventory gradually; avoid large swings."
            )
            candidates.append(
                seed
                + "\n\nHeuristic: smoothing. Let desired = incoming_order + (target_inventory - inv_pos)/4. "
                  "Clip changes vs last_order to reduce bullwhip."
            )
            candidates.append(
                seed
                + "\n\nHeuristic: be conservative when backlog is small; be aggressive only when backlog grows. "
                  "Prefer small adjustments to avoid oscillations."
            )
            # de-dup
            candidates = [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]

            subset = test_samples[:prompt_search_episodes]
            best_prompt = candidates[0]
            best_cost = float("inf")
            records: List[Dict[str, Any]] = []

            for i, cand in enumerate(candidates):
                self._beergame_policy_prompt = cand
                sub_log = os.path.join(ctx["resolved_save_path"], f"prompt_search_{i}")
                os.makedirs(sub_log, exist_ok=True)
                sub_ctx = dict(ctx)
                sub_ctx["log_dir"] = sub_log
                try:
                    res_i, _ = beergame_evaluate_run(
                        agent=self,
                        test_samples=subset,
                        config=config,
                        ctx=sub_ctx,
                    )
                    cost = res_i.get("avg_total_cost_controlled")
                    cost_f = float(cost) if isinstance(cost, (int, float)) else float("inf")
                except Exception:
                    cost_f = float("inf")
                records.append({"idx": i, "avg_total_cost_controlled": cost_f, "prompt": cand})
                if cost_f < best_cost:
                    best_cost = cost_f
                    best_prompt = cand

            self._beergame_policy_prompt = best_prompt
            try:
                with open(os.path.join(ctx["resolved_save_path"], "prompt_search_results.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "prompt_search_episodes": prompt_search_episodes,
                            "best_avg_total_cost_controlled": best_cost,
                            "records": records,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            except Exception:
                pass

            # persist globally for next runs
            try:
                with open(best_prompt_path, "w", encoding="utf-8") as f:
                    f.write(best_prompt)
            except Exception:
                pass

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

    def run_consulting(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if mode not in {"online", "eval_only"}:
            raise ValueError(f"Consulting mode must be online/eval_only, got {mode}")
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

        # ------------------------------------------------------
        # Prompt search on first K cases using LLM judge score
        # ------------------------------------------------------
        def _candidate_prompts(base: str) -> List[str]:
            base = base or SEED_PROMPT
            tweaks = [
                "Focus on MECE structure and state explicit hypotheses before answering.",
                "Be concise: lead with 2-3 key drivers, then propose a crisp next-step plan.",
                "Quantify whenever possible: estimate ballpark numbers and sanity-check.",
                "Clarify assumptions and ask for missing data only if critical.",
                "Frame the answer in a top-down, bullet style to guide the interviewer.",
            ]
            prompts = [base]
            for t in tweaks:
                prompts.append(base + "\n\n[STYLE EMPHASIS]\n" + t)
            return prompts

        prompt_search_k = int(config.get("gepa_prompt_search_k", 3))
        candidates = _candidate_prompts(self.cfg.seed_prompt or SEED_PROMPT)
        best_prompt = candidates[0]
        best_score = -1.0

        if prompt_search_k > 0 and candidates:
            subset = test_samples[:prompt_search_k]
            original_prompt = self.cfg.seed_prompt
            for idx, cand in enumerate(candidates):
                self.cfg.seed_prompt = cand
                tmp_log_dir = os.path.join(resolved_save_path, f"prompt_search_{idx}")
                os.makedirs(tmp_log_dir, exist_ok=True)
                try:
                    cand_results, _ = evaluate_consulting_set(
                        agent=self,
                        test_samples=subset,
                        config=config,
                        log_dir=tmp_log_dir,
                    )
                    score = cand_results.get("metrics", {}).get("overall", 0.0) or 0.0
                except Exception:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best_prompt = cand
            self.cfg.seed_prompt = best_prompt
            # 记录选择结果
            try:
                with open(
                    os.path.join(resolved_save_path, "prompt_search_choice.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "prompt_search_k": prompt_search_k,
                            "best_score": best_score,
                            "best_prompt": best_prompt,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception:
                pass

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
                ensure_ascii=False,
            )

        cfg_payload = dict(config)
        cfg_payload.update(
            {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
                "gepa_config": self.cfg.__dict__,
            }
        )
        with open(
            os.path.join(resolved_save_path, "run_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(cfg_payload, f, indent=2, ensure_ascii=False)

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
    # EDT evaluation entry (single-run; uses seriousgame_tools pipeline)
    # ==========================================================

    def run_edt(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run EDT evaluation (single-pass)."""
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
    # BizBench / StructuredReasoning runner (previously: run)
    # ==========================================================

    def run_bizbench(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor,
        config: Dict[str, Any],
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"{self.agent_method.upper()} 仅支持 {self.SUPPORTED_MODES}，收到 {mode}")
        if mode in {"online", "eval_only"} and not test_samples:
            raise ValueError(f"{self.agent_method.upper()} 需要 test_samples")
        if mode == "offline" and (train_samples is None or val_samples is None):
            raise ValueError(f"{self.agent_method.upper()} offline 需要 train_samples 与 val_samples")

        task_name = config.get("task_name", "unknown_task")
        name_lower = str(task_name).lower()
        if "beer" in name_lower:
            return self.run_beergame(
                mode=mode,
                test_samples=test_samples or [],
                data_processor=data_processor,
                config=config,
            )
        if "consult" in name_lower:
            return self.run_consulting(
                mode=mode,
                test_samples=test_samples or [],
                data_processor=data_processor,
                config=config,
            )

        save_dir = config.get("save_dir")
        if not save_dir:
            raise ValueError("配置缺少 save_dir")

        task_name = config.get("task_name", "unknown_task")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name, self.agent_method, mode, timestamp)
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)
        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        # 构造 Dpareto / Dfeedback / 窗口逻辑
        if mode == "online":
            # 窗口化：按 ACE online_eval_frequency 切分窗口，先测后训，跨窗口继承 best prompt
            window_size = config.get("online_eval_frequency", 100)
            num_windows = (len(test_samples) + window_size - 1) // window_size if window_size > 0 else 1
            best_prompt = self.cfg.seed_prompt or SEED_PROMPT
            overall_correct, overall_total = 0, 0
            window_records: List[Dict[str, Any]] = []
            trace_entries: List[Dict[str, Any]] = []
            seen_pool: List[Dict[str, Any]] = []

            for w in range(num_windows):
                start = w * window_size
                end = min((w + 1) * window_size, len(test_samples))
                window_samples = test_samples[start:end]
                seen_pool.extend(window_samples)

                # 构造 Dpareto / Dfeedback：从累计池抽样，确保 Dfeedback >= 2 * |Dpareto|
                pool_size = len(seen_pool)
                min_pareto = max(15, self.cfg.mini_batch_size * 2)
                pareto_size = min(pool_size, min_pareto)
                feedback_target = max(pareto_size * 2, self.cfg.mini_batch_size * 4)
                feedback_size = min(pool_size, feedback_target)

                if pareto_size == 0:
                    raise ValueError("累计样本池为空，无法构建 Dpareto/Dfeedback。")

                Dpareto = random.sample(seen_pool, pareto_size) if pool_size > pareto_size else list(seen_pool)
                remaining_pool = [s for s in seen_pool if s not in Dpareto]
                if len(remaining_pool) >= feedback_size:
                    Dfeedback = random.sample(remaining_pool, feedback_size)
                else:
                    Dfeedback = list(seen_pool) if pool_size >= feedback_size else list(seen_pool)

                # 预算下限提示：初始评估 + 一次改进 + 两个 mini-batch
                required_budget = self.cfg.num_initial * len(Dpareto) + len(Dpareto) + self.cfg.mini_batch_size * 2
                window_budget = self.cfg.window_budget or self.cfg.budget
                if window_budget < required_budget:
                    print(
                        f"[GEPA][warn] window {w+1} 预算可能不足："
                        f"window_budget={window_budget}, 建议至少 {required_budget} "
                        f"(Dpareto={len(Dpareto)}, mini_batch_size={self.cfg.mini_batch_size}, num_initial={self.cfg.num_initial})"
                    )

                # Step 1: 先评测
                eval_res = self._evaluate_with_prompt(
                    best_prompt,
                    window_samples,
                    data_processor,
                    log_dir=log_dir,
                    max_workers=self.cfg.max_workers,
                    use_json_mode=self.cfg.use_json_mode,
                    task_name=config.get("task_name", ""),
                )
                overall_correct += eval_res["correct"]
                overall_total += eval_res["total"]
                window_records.append(
                    {
                        "window": w + 1,
                        "start": start,
                        "end": end,
                        "accuracy": eval_res["accuracy"],
                        "correct": eval_res["correct"],
                        "total": eval_res["total"],
                        "errors": eval_res["errors"],
                    }
                )

                # Step 2: 在窗口样本上优化 GEPA，每窗口单独预算
                window_budget = self.cfg.window_budget or self.cfg.budget
                win_cfg = replace(self.cfg, budget=window_budget, num_initial=1)
                prev_best_prompt = best_prompt
                gepa_result: GepaResult = run_gepa(
                    generator_client=self.generator_client,
                    reflection_client=self.reflector_client,
                    api_provider=self.api_provider,
                    target_model=self.generator_model,
                    reflection_model=self.reflector_model,
                    cfg=win_cfg,
                    Dpareto=Dpareto,
                    Dfeedback=Dfeedback,
                    initial_candidates=[best_prompt],
                    use_accuracy=True,
                    data_processor=data_processor,
                debug=True,
                )
                # 仅当本窗最优候选优于初始候选才更新跨窗 best_prompt
                init_score = None
                for c in gepa_result.candidates:
                    if c.get("id") == 0:
                        init_score = c.get("mean_score")
                        break
                best_score = gepa_result.best_candidate.get("mean_score")
                improved = (
                    init_score is None
                    or best_score is None
                    or best_score > init_score
                )

                for t in gepa_result.trace:
                    t["window"] = w + 1
                trace_entries.extend(gepa_result.trace)
                if improved:
                    best_prompt = gepa_result.best_prompt
                else:
                    best_prompt = prev_best_prompt
                    trace_entries.append(
                        {
                            "step": len(trace_entries),
                            "window": w + 1,
                            "strategy": "cross-window-keep",
                            "accepted": False,
                            "reason": "no_improvement_on_window",
                            "init_mean": init_score,
                            "best_mean": best_score,
                        }
                    )

            accuracy = overall_correct / overall_total if overall_total else 0.0
            final_results = {
                "accuracy": accuracy,
                "total": overall_total,
                "correct": overall_correct,
                "window_results": window_records,
            }

            trace_path = os.path.join(resolved_save_path, "gepa_trace.jsonl")
            with open(trace_path, "w", encoding="utf-8") as f:
                for item in trace_entries:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
                json.dump({"test_results": final_results}, f, indent=2, ensure_ascii=False)

            cfg_payload = {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
                "gepa_config": self.cfg.__dict__,
                "best_prompt": best_prompt,
                "gepa_data_usage": "online windowed: each window test then optimize on same window",
            }
            cfg_payload.update(config)
            with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg_payload, f, indent=2, ensure_ascii=False)

            print(f"\n{'='*60}")
            print(f"{self.agent_method.upper()} - RUN COMPLETE")
            print(f"{'='*60}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Results saved to: {resolved_save_path}")
            print(f"{'='*60}\n")
            return final_results
        elif mode == "offline":
            # baseline 测试（可选）
            baseline_results = None
            if test_samples:
                baseline_results = self._evaluate_with_prompt(
                    prompt=self.cfg.seed_prompt or SEED_PROMPT,
                    samples=test_samples,
                    data_processor=data_processor,
                    log_dir=log_dir,
                    max_workers=self.cfg.max_workers,
                    use_json_mode=self.cfg.use_json_mode,
                    task_name=config.get("task_name", ""),
                )

            # 训练/优化：Dpareto=val，Dfeedback=train
            gepa_result: GepaResult = run_gepa(
                generator_client=self.generator_client,
                reflection_client=self.reflector_client,
                api_provider=self.api_provider,
                target_model=self.generator_model,
                reflection_model=self.reflector_model,
                cfg=self.cfg,
                Dpareto=val_samples or [],
                Dfeedback=train_samples or [],
            )

            trace_path = os.path.join(resolved_save_path, "gepa_trace.jsonl")
            with open(trace_path, "w", encoding="utf-8") as f:
                for item in gepa_result.trace:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            eval_set = test_samples if test_samples else val_samples
            final_results = self._evaluate_with_prompt(
                gepa_result.best_prompt,
                eval_set or [],
                data_processor,
                log_dir=log_dir,
                max_workers=self.cfg.max_workers,
                use_json_mode=self.cfg.use_json_mode,
                task_name=config.get("task_name", ""),
            )

            with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
                payload = {"final_test_results": final_results}
                if baseline_results:
                    payload["baseline_results"] = baseline_results
                json.dump(payload, f, indent=2, ensure_ascii=False)

            cfg_payload = {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
                "gepa_config": self.cfg.__dict__,
                "best_prompt": gepa_result.best_prompt,
                "best_candidate": gepa_result.best_candidate,
                "gepa_data_usage": "offline: Dfeedback=train, Dpareto=val; final eval on test if provided else val",
            }
            cfg_payload.update(config)
            with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg_payload, f, indent=2, ensure_ascii=False)

            print(f"\n{'='*60}")
            print(f"{self.agent_method.upper()} - RUN COMPLETE")
            print(f"{'='*60}")
            print(f"Accuracy: {final_results.get('accuracy', 0.0):.3f}")
            print(f"Results saved to: {resolved_save_path}")
            print(f"{'='*60}\n")
            return final_results
        else:
            Dpareto, Dfeedback = self._split_eval_only(
                test_samples, self.cfg.feedback_ratio, self.cfg.mini_batch_size
            )

            # 运行 GEPA
            gepa_result: GepaResult = run_gepa(
                generator_client=self.generator_client,
                reflection_client=self.reflector_client,
                api_provider=self.api_provider,
                target_model=self.generator_model,
                reflection_model=self.reflector_model,
                cfg=self.cfg,
                Dpareto=Dpareto,
                Dfeedback=Dfeedback,
                use_accuracy=True,
                data_processor=data_processor,
                debug=True,
            )

            # 记录 trace
            trace_path = os.path.join(resolved_save_path, "gepa_trace.jsonl")
            with open(trace_path, "w", encoding="utf-8") as f:
                for item in gepa_result.trace:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # 使用 best prompt 完整评测
            final_results = self._evaluate_with_prompt(
                gepa_result.best_prompt,
                test_samples,
                data_processor,
                log_dir=log_dir,
                max_workers=self.cfg.max_workers,
                use_json_mode=self.cfg.use_json_mode,
                task_name=config.get("task_name", ""),
            )

            with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
                json.dump({"test_results": final_results}, f, indent=2, ensure_ascii=False)

            cfg_payload = {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
                "gepa_config": self.cfg.__dict__,
                "best_prompt": gepa_result.best_prompt,
                "best_candidate": gepa_result.best_candidate,
                "gepa_data_usage": "eval_only splits test into feedback/pareto; online windowed; offline uses train/val",
            }
            cfg_payload.update(config)
            with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg_payload, f, indent=2, ensure_ascii=False)

            print(f"\n{'='*60}")
            print(f"{self.agent_method.upper()} - RUN COMPLETE")
            print(f"{'='*60}")
            print(f"Accuracy: {final_results.get('accuracy', 0.0):.3f}")
            print(f"Results saved to: {resolved_save_path}")
            print(f"{'='*60}\n")

            return final_results

    # ==========================================================
    # Unified entry point (router)
    # ==========================================================

    def run(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor,
        config: Dict[str, Any],
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Route evaluation by dataset.

        - EDT: run_edt (single-pass evaluation; no changes to seriousgame_tools)
        - Otherwise: run_bizbench (original GEPA behavior, including Consulting routing)

        This function is intentionally minimal to avoid impacting existing tasks.
        """
        task_name = str(config.get("task_name", "unknown_task")).lower()
        if "edt" in task_name:
            if mode not in {"online", "eval_only"}:
                raise ValueError(f"EDT 模式必须为 online/eval_only，收到 {mode}")
            if not test_samples:
                raise ValueError("EDT 需要非空 test_samples")
            return self.run_edt(
                mode=mode,
                test_samples=test_samples,
                data_processor=data_processor,
                config=config,
            )

        return self.run_bizbench(
            mode=mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
            train_samples=train_samples,
            val_samples=val_samples,
        )