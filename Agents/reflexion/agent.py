"""
Reflexion agent adapter for BizBench。

支持论文式“失败→反思→记忆→再用”流程：
- 默认窗口并行只读评估 + 窗口内串行写记忆（对齐 dynamic_cheatsheet 的性能折中）
- 可通过 strict_sequential 启用全串行，确保严格时序
记忆以 JSONL 存放于当前 run 目录，检索按 question+context 哈希精确匹配，不做相似度。
"""

from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from utils.consulting_tools import evaluate_consulting_set
from utils.tools import extract_answer
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
from .generator import ReflexionGenerator


class ReflexionAgent:
    """Lightweight Reflexion agent: only online / eval_only modes."""

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        reflexion_rounds: int = 2,
        initial_temperature: float = 0.0,
        reflect_temperature: float = 0.2,
        agent_method: str = "reflexion",
        strict_sequential: bool = True,
        memory_top_k: int = 3,
    ):
        self.agent_method = agent_method
        self.max_tokens = max_tokens
        self.reflexion_rounds = reflexion_rounds
        self.strict_sequential = strict_sequential
        self.memory_top_k = max(memory_top_k, 1)
        self.generator = ReflexionGenerator(
            api_provider=api_provider,
            model_name=generator_model,
            max_tokens=max_tokens,
            reflexion_rounds=reflexion_rounds,
            initial_temperature=initial_temperature,
            reflect_temperature=reflect_temperature,
        )
        self.generator_client = self.generator.init_role.client
        self.temperature = 0.7

        self.memory_path: Optional[str] = None
        self.memory: List[Dict[str, Any]] = []

        # EDT runtime state
        self._edt_last_ctx: Optional[Dict[str, Any]] = None
        self._edt_online_cfg: Dict[str, Any] = {}
        self._edt_mode: str = "eval_only"

    # ---------- memory utilities ----------
    @staticmethod
    def _hash_sample(question: str, context: str) -> str:
        digest = hashlib.sha256()
        digest.update((question or "").encode("utf-8"))
        digest.update(b"\n")
        digest.update((context or "").encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _hash_edt(base_summary: Dict[str, Any], scenario_meta: Dict[str, Any]) -> str:
        """Stable-ish key to group memory within the same EDT base scenario."""
        digest = hashlib.sha256()
        digest.update(json.dumps(base_summary, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
        digest.update(json.dumps(scenario_meta or {}, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        return digest.hexdigest()

    def _load_memory(self, path: str) -> None:
        self.memory_path = path
        if not os.path.exists(path):
            self.memory = []
            return
        with open(path, "r", encoding="utf-8") as f:
            self.memory = [json.loads(line) for line in f if line.strip()]

    def _append_memory(self, entry: Dict[str, Any]) -> None:
        if not self.memory_path:
            return
        self.memory.append(entry)
        # 长记忆截断：仅保留最近 memory_top_k 条
        if len(self.memory) > self.memory_top_k:
            self.memory = self.memory[-self.memory_top_k :]
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _retrieve_reflections(self, *, domain: str, key: Optional[str] = None) -> List[str]:
        """
        按原始实现思路：不做相似度/哈希匹配，直接取最近的若干失败反思。
        保留 hash 字段仅用于记录，检索时不依赖它。
        """
        if domain == "bizbench":
            matched = [m for m in self.memory if not m.get("success")]
            matched = matched[-self.memory_top_k :]
            return [m.get("reflection", "") for m in matched if m.get("reflection")]

        if domain == "edt":
            if key:
                scoped = [
                    m
                    for m in self.memory
                    if m.get("domain") == "edt"
                    and (m.get("edt_key") == key)
                    and (not m.get("success"))
                    and m.get("reflection")
                ]
                scoped = scoped[-self.memory_top_k :]
                if scoped:
                    return [m["reflection"] for m in scoped if m.get("reflection")]

            # fallback: any EDT failures
            matched = [m for m in self.memory if m.get("domain") == "edt" and not m.get("success") and m.get("reflection")]
            matched = matched[-self.memory_top_k :]
            return [m.get("reflection", "") for m in matched if m.get("reflection")]

        return []

    # ---------- JSON helpers (for EDT schema output) ----------
    @staticmethod
    def _extract_first_json_object(text: str) -> str:
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

    # ==========================================================
    # EDT: decision hook (called by seriousgame_tools)
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
        system, user = render_edt_prompt(ctx)

        edt_key = self._hash_edt(base_summary, scenario_meta)
        prior_refs: List[str] = []
        if self._edt_mode == "online":
            prior_refs = self._retrieve_reflections(domain="edt", key=edt_key)

        if prior_refs:
            # Keep it compact; EDT prompt should not be bloated
            mem_block = "\n".join([f"- {r.strip()}" for r in prior_refs if r and r.strip()])[:4000]
            system = (
                system
                + "\n\nRECENT LESSONS (from previous EDT runs; follow them if applicable):\n"
                + mem_block
            )

        user = (
            user
            + "\n\nIMPORTANT OUTPUT CONSTRAINTS:\n"
            + "- Output ONLY a single JSON object.\n"
            + "- The JSON object must contain ONLY keys: C, R, P.\n"
            + "- Do not wrap JSON in markdown fences.\n"
        )

        resp = self.generator_client.chat.completions.create(
            model=self.generator.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=min(self.max_tokens, 512),
        )
        text = (resp.choices[0].message.content or "").strip()
        raw = self._safe_load_json(text)

        candidate = raw.get("final_answer", raw)
        if isinstance(candidate, str):
            candidate = self._safe_load_json(candidate)
        if not isinstance(candidate, dict):
            candidate = {}

        return normalize_edt_schema(candidate, ctx)

    # ==========================================================
    # EDT: online memory hook (called by seriousgame_tools after an episode)
    # ==========================================================
    def _extract_utilization(self, learn_metrics: Dict[str, Any]) -> Optional[float]:
        """
        Try best-effort extraction of utilization in [0,1].
        You can adapt key names here without touching seriousgame_tools.
        """
        if not isinstance(learn_metrics, dict):
            return None
        for k in ("utilization", "avg_utilization", "mean_utilization", "resource_utilization"):
            v = learn_metrics.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def _build_edt_reflection_prompt(
        self,
        *,
        base_summary: Dict[str, Any],
        scenario_meta: Dict[str, Any],
        schema: Dict[str, Any],
        learn_metrics: Dict[str, Any],
        utilization: Optional[float],
        threshold: float,
    ) -> Tuple[str, str]:
        """
        Generate a concise, actionable reflection to improve utilization for the next episode.
        Output is a single paragraph (no bullets) to keep memory compact.
        """
        system = (
            "You are a senior operations manager reflecting on a simulated enterprise digital twin run.\n"
            "Your job: write ONE short actionable lesson that can be applied in the NEXT run.\n"
            "Focus only on decisions a manager can control in the schema (C/R/P), and prioritize improving utilization.\n"
            "Do NOT mention prompts, LLMs, or evaluation. Do NOT include JSON. Do NOT use bullet points.\n"
        )
        user = (
            "Episode summary:\n"
            f"- Utilization: {utilization if utilization is not None else 'unknown'} (target >= {threshold})\n"
            f"- Metrics (raw): {json.dumps(learn_metrics, ensure_ascii=False)[:2000]}\n\n"
            "The schema you used (C/R/P JSON):\n"
            f"{json.dumps(schema, ensure_ascii=False)[:2000]}\n\n"
            "Company/base scenario context (abbrev):\n"
            f"{json.dumps(base_summary, ensure_ascii=False)[:1500]}\n\n"
            "Write ONE short lesson (1-3 sentences) describing what to change in the next schema to increase utilization."
        )
        return system, user

    def _record_edt_experience(
        self,
        base_summary: Dict[str, Any],
        scenario_meta: Dict[str, Any],
        schema: Dict[str, Any],
        learn_metrics: Dict[str, Any],
        run_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        """
        Called by seriousgame_tools after each episode when mode == online.
        Stores a reflection memory based on utilization signal.
        """
        if self._edt_mode != "online":
            return

        online_cfg = self._edt_online_cfg or {}
        threshold = float(online_cfg.get("utilization_threshold", 0.70))
        reflect_on_success = bool(online_cfg.get("reflect_on_success", False))

        utilization = self._extract_utilization(learn_metrics)
        success = (utilization is not None) and (utilization >= threshold)

        if success and not reflect_on_success:
            # record success marker (optional)
            entry = {
                "domain": "edt",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "utilization": utilization,
                "edt_key": self._hash_edt(base_summary, scenario_meta),
                "run_id": run_id,
                "episode_id": episode_id,
            }
            self._append_memory(entry)
            return

        # Generate reflection text
        sys_p, user_p = self._build_edt_reflection_prompt(
            base_summary=base_summary,
            scenario_meta=scenario_meta,
            schema=schema,
            learn_metrics=learn_metrics,
            utilization=utilization,
            threshold=threshold,
        )
        resp = self.generator_client.chat.completions.create(
            model=self.generator.model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
            ],
            temperature=0.2,
            max_tokens=min(self.max_tokens, 256),
        )
        reflection = (resp.choices[0].message.content or "").strip()
        # keep it compact
        reflection = " ".join(reflection.split())
        if len(reflection) > 600:
            reflection = reflection[:600].rstrip()

        entry = {
            "domain": "edt",
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "utilization": utilization,
            "learn_metrics": learn_metrics,
            "schema": schema,
            "reflection": reflection,
            "edt_key": self._hash_edt(base_summary, scenario_meta),
            "run_id": run_id,
            "episode_id": episode_id,
        }
        self._append_memory(entry)

    # ==========================================================
    # EDT evaluation entry (uses seriousgame_tools pipeline)
    # ==========================================================
    def _init_edt_memory(self, ctx: Dict[str, Any], mode: str, config: Dict[str, Any]) -> None:
        """
        Initialize EDT memory file under the run directory.
        We keep EDT memory separate from BizBench memory to avoid contamination.
        """
        self._edt_last_ctx = ctx
        self._edt_mode = mode

        # online cfg: optional
        edt_cfg = config.get("edt") if isinstance(config, dict) else None
        online_cfg = {}
        if isinstance(edt_cfg, dict):
            online_cfg = edt_cfg.get("online") if isinstance(edt_cfg.get("online"), dict) else {}
        self._edt_online_cfg = online_cfg or {}

        # Determine run directory (ctx should contain one of these keys)
        resolved_save_path = (
            ctx.get("resolved_save_path")
            or ctx.get("save_path")
            or ctx.get("run_dir")
            or ctx.get("log_dir")
        )
        if not resolved_save_path:
            # fallback: keep in config save_dir root
            resolved_save_path = os.path.join(config.get("save_dir", "results"), "SeriousGame_EDT", "reflexion_tmp")
            os.makedirs(resolved_save_path, exist_ok=True)

        mem_path = os.path.join(resolved_save_path, "edt_reflections.jsonl")
        self._load_memory(mem_path)

        # store in run_config for debugging
        try:
            run_cfg_path = os.path.join(resolved_save_path, "run_config.json")
            if os.path.exists(run_cfg_path):
                with open(run_cfg_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                payload = {}
            payload["reflexion_edt_memory_path"] = mem_path
            payload["reflexion_edt_online_cfg"] = self._edt_online_cfg
            with open(run_cfg_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

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

        # Initialize per-run EDT memory (for online mode this is essential)
        self._init_edt_memory(ctx=ctx, mode=mode, config=config)

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
        Hook called at consulting case start. No cross-case memory kept.
        """
        self._current_case_id = case_id

    def on_case_end(
        self,
        case_id: str,
        case_text: str,
        history: List[Dict[str, str]],
    ) -> None:
        """
        Hook called when a consulting case finishes. No persistence for baseline.
        """
        _ = (case_id, case_text, history)
        self._current_case_id = None

    def reply(self, case_id: str, history: List[Dict[str, str]]) -> str:
        """
        Consulting candidate reply using reflexion workflow:
        initial answer -> reflection -> rewrite (early stop if self-verified).
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

        prior_refs = self._retrieve_reflections(domain="bizbench")

        # 走 reflexion 链：question=最新提问，context=全量对话，附带最近反思
        response_text, _, meta = self.generator.generate(
            question=last_interviewer_msg or "(Interviewer message missing.)",
            playbook="",
            context=transcript_text,
            reflection="(empty)",
            prior_reflections=prior_refs,
            use_json_mode=True,
            call_id=f"consult_reflexion_{case_id}_t{turns}",
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
                "Let me structure the issues, share a hypothesis, and outline the "
                "first analyses I'd run to validate it."
            )

        # 追加反思到持久化（若已设置路径）
        reflections = meta.get("reflections") if isinstance(meta, dict) else None
        if reflections:
            try:
                last_reflection = reflections[-1]
                entry = {
                    "hash": self._hash_sample(last_interviewer_msg, transcript_text),
                    "question": last_interviewer_msg,
                    "context": transcript_text,
                    "reflection": last_reflection,
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                }
                self._append_memory(entry)
            except Exception:
                pass

        return reply.strip()

    # ==========================================================
    # BeerGame: decision hook + evaluation entry (memory-free)
    # ==========================================================
    def _decide_order_qty(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """
        BeerGame 单步决策：使用 Reflexion（初答→反思→重写）多阶段生成，
        读取/写入持久化反思记忆（memory_top_k），并保持 JSON-only 与安全兜底。
        """
        role = str(ctx.get("role", obs.get("role", "retailer")))
        max_order_qty = int(getattr(self, "max_order_qty", 5000))

        _ = beergame_build_query(obs)  # 留作潜在日志/扩展

        base_order = beergame_base_rule_order(
            obs=obs,
            ctx=ctx,
            max_order_qty=max_order_qty,
        )

        system, user = beergame_render_prompt(
            role=role,
            obs=obs,
            retrieved="",  # reflexion baseline 无记忆
            base_order=base_order,
        )

        question_text = (
            f"{system}\n\n{user}\n\n"
            "Respond strictly in JSON ONLY, exactly in the form:\n"
            "{\n"
            '  "order_qty": <integer>,\n'
            '  "note": "<brief rationale>"\n'
            "}\n"
            "No other keys. No text before or after JSON. Do NOT reveal chain-of-thought."
        )

        prior_reflections = self._retrieve_reflections(domain="bizbench") if hasattr(self, "_retrieve_reflections") else None

        response_text, _, _meta = self.generator.generate(
            question=question_text,
            playbook="",
            context="",
            reflection="",
            prior_reflections=prior_reflections,
            use_json_mode=True,
            call_id=f"beergame_reflexion_{ctx.get('scenario_id','')}_{ctx.get('episode_id','')}_w{obs.get('week','')}",
            log_dir=None,
        )

        order_qty, used_fallback = self._extract_order_qty_from_reflexion(
            response_text=response_text,
            base_order=base_order,
            max_order_qty=max_order_qty,
        )
        # 记录 note，包含是否回退基线
        fallback_tag = " (fallback_base_order)" if used_fallback else ""
        self._last_beergame_note = f"reflexion_final{fallback_tag}: {response_text[:500]}"

        # 将本轮决策/反思写入持久化记忆
        try:
            entry = {
                "hash": self._hash_sample(str(obs), f"week={obs.get('week')} role={role}"),
                "question": f"BeerGame week={obs.get('week')} role={role}",
                "context": json.dumps(obs, ensure_ascii=False),
                "reflection": response_text,
                "timestamp": datetime.now().isoformat(),
                "success": False,  # 无即时反馈，先标记未验证
            }
            self._append_memory(entry)
        except Exception:
            pass

        return int(order_qty)

    def _extract_order_qty_from_reflexion(
        self, response_text: str, base_order: int, max_order_qty: int
    ) -> tuple[int, bool]:
        """
        从 Reflexion 生成的 JSON 文本中提取订单；失败则回退 base_order，并返回是否回退。
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
                        # 若 final_answer 是对象，尝试取其中的订单字段
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

        used_fallback = candidate is None

        if candidate is None:
            candidate = base_order
            print(f"[Reflexion][BeerGame] parse failed, fallback to base_order={base_order}")

        candidate = max(0, min(int(candidate), max_order_qty))
        return candidate, used_fallback

    def run_beergame(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """BeerGame 评测入口（委托 seriousgame_tools 通用流程）。"""
        _ = data_processor

        # 持久化记忆：为 BeerGame 单独加载/创建反思记忆文件
        try:
            save_dir = config.get("save_dir", "results")
            memory_path = os.path.join(save_dir, "reflexion_beergame_memory.jsonl")
            self._load_memory(memory_path)
        except Exception:
            pass

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
        Consulting / case-interview evaluation entry for the Reflexion baseline.
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

        # 持久化反思记忆（咨询模式也启用）
        memory_path = os.path.join(resolved_save_path, "reflections.jsonl")
        self._load_memory(memory_path)

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
                "reflexion_rounds": self.reflexion_rounds,
                "reflexion_strict_sequential": self.strict_sequential,
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
            raise ValueError("Configuration missing 'save_dir' for ReflexionAgent.")

        task_name_safe = task_name or "unknown_task"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name_safe, self.agent_method, mode, timestamp)
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - REFLEXION EVALUATION")
        print(f"{'='*60}")
        print(f"Samples: {len(test_samples)}")
        print(f"Reflexion rounds: {self.reflexion_rounds}")
        print(f"Log dir: {log_dir}")
        print(f"{'='*60}\n")

        memory_path = os.path.join(resolved_save_path, "reflections.jsonl")
        self._load_memory(memory_path)

        test_workers = config.get("test_workers", 20)
        online_eval_frequency = config.get("online_eval_frequency", 15)
        json_mode = config.get("json_mode", False)

        results, error_log = self._run_with_memory(
            mode=mode,
            samples=test_samples,
            data_processor=data_processor,
            log_dir=log_dir,
            use_json_mode=json_mode,
            test_workers=test_workers,
            window_size=online_eval_frequency,
        )

        with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump({"test_results": results, "error_log": error_log}, f, indent=2)

        config_payload = dict(config)
        config_payload.update(
            {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
                "reflexion_rounds": self.reflexion_rounds,
                "reflexion_strict_sequential": self.strict_sequential,
                "reflexion_memory_path": memory_path,
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

    # ---------- core run with memory ----------
    def _evaluate_single(
        self,
        idx: int,
        sample: Dict[str, Any],
        log_dir: str,
        use_json_mode: bool,
        memory_snapshot: List[Dict[str, Any]],
        data_processor,
    ) -> Dict[str, Any]:
        # build prior reflections from snapshot (to keep read-only during parallel eval)
        question = sample.get("question", "")
        context = sample.get("context", "")
        target = sample.get("target", "")
        if memory_snapshot is self.memory:
            prior_refs = self._retrieve_reflections(domain="bizbench")
        else:
            prior_refs = [
                m.get("reflection", "")
                for m in memory_snapshot
                if not m.get("success")
            ][-self.memory_top_k :]

        response, _, meta = self.generator.generate(
            question=question,
            playbook="",
            context=context,
            reflection="(empty)",
            prior_reflections=prior_refs,
            use_json_mode=use_json_mode,
            call_id=f"reflexion_{idx}",
            log_dir=log_dir,
        )
        final_answer = extract_answer(response)
        is_correct = data_processor.answer_is_correct(final_answer, target)

        return {
            "index": idx,
            "question": question,
            "context": context,
            "target": target,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "meta": meta,
            "raw_response": response,
        }

    def _run_window_parallel_eval(
        self,
        samples: List[Dict[str, Any]],
        start_idx: int,
        log_dir: str,
        use_json_mode: bool,
        test_workers: int,
        memory_snapshot: List[Dict[str, Any]],
        data_processor,
    ) -> List[Dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        records: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=test_workers) as executor:
            futures = {
                executor.submit(
                    self._evaluate_single,
                    start_idx + i,
                    sample,
                    log_dir,
                    use_json_mode,
                    memory_snapshot,
                    data_processor,
                ): i
                for i, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda x: x["index"])
        return records

    def _run_sequential(
        self,
        samples: List[Dict[str, Any]],
        start_idx: int,
        log_dir: str,
        use_json_mode: bool,
        data_processor,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for local_idx, sample in enumerate(samples):
            rec = self._evaluate_single(
                start_idx + local_idx,
                sample,
                log_dir,
                use_json_mode,
                self.memory,  # current memory (updated along the way)
                data_processor,
            )
            records.append(rec)
            self._maybe_update_memory(rec)
        return records

    def _maybe_update_memory(self, record: Dict[str, Any]) -> None:
        """If incorrect, append reflection entry to memory."""
        if record.get("is_correct"):
            # 可记录成功样本占位以便去重；目前仅跳过
            return
        reflections = record.get("meta", {}).get("reflections") or []
        if not reflections:
            return
        entry = {
            "hash": self._hash_sample(record.get("question", ""), record.get("context", "")),
            "question": record.get("question", ""),
            "context": record.get("context", ""),
            "reflection": reflections[-1],
            "timestamp": datetime.now().isoformat(),
            "success": False,
        }
        self._append_memory(entry)

    def _run_with_memory(
        self,
        mode: str,
        samples: List[Dict[str, Any]],
        data_processor,
        log_dir: str,
        use_json_mode: bool,
        test_workers: int,
        window_size: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.strict_sequential or mode == "eval_only":
            # eval_only 也走无窗口的严格序列，保证语义
            records = self._run_sequential(samples, 0, log_dir, use_json_mode, data_processor)
        else:
            total = len(samples)
            num_windows = (total + window_size - 1) // window_size
            records: List[Dict[str, Any]] = []
            for w in range(num_windows):
                start = w * window_size
                end = min((w + 1) * window_size, total)
                window_samples = samples[start:end]
                print(f"\n{'='*60}")
                print(f"WINDOW {w + 1}/{num_windows} | Samples {start} to {end - 1}")
                print(f"{'='*60}")

                memory_snapshot = list(self.memory)
                window_records = self._run_window_parallel_eval(
                    window_samples,
                    start,
                    log_dir,
                    use_json_mode,
                    test_workers,
                    memory_snapshot,
                    data_processor,
                )
                records.extend(window_records)

                # sequential memory update using ordered results
                for rec in window_records:
                    self._maybe_update_memory(rec)

        # aggregate metrics
        predictions = [r["final_answer"] for r in records]
        targets = [r["target"] for r in records]
        accuracy = data_processor.evaluate_accuracy(predictions, targets) if predictions and targets else 0.0
        errors = [
            {"index": r["index"], "prediction": r["final_answer"], "ground_truth": r["target"]}
            for r in records
            if not r.get("is_correct")
        ]
        results = {
            "accuracy": accuracy,
            "total": len(records),
            "correct": len(records) - len(errors),
            "samples": records,
        }
        error_log = {"errors": errors, "accuracy": accuracy}
        return results, error_log

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
        task_name = str(config.get("task_name", getattr(data_processor, "task_name", ""))).lower()
        if "edt" in task_name:
            return self.run_edt(
                mode=mode,
                test_samples=test_samples or [],
                data_processor=data_processor,
                config=config,
            )
        return self.run_bizbench(mode=mode, test_samples=test_samples, data_processor=data_processor, config=config)
