from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils.tools import initialize_clients
from utils.memory_tools import evaluate_test_set
from utils.seriousgame_tools import evaluate_edt_set
from utils.consulting_tools import (
    consulting_prepare_run,
    consulting_evaluate_run,
    consulting_save_run,
    consulting_build_candidate_query,
    consulting_render_candidate_prompt,
    consulting_extract_candidate_reply,
    consulting_format_memory_note,
)
from .generator import AMemBizBenchGenerator
from .memory_layer import AMemMemory
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
    beergame_format_memory_note,
)

# ======================================================================
# SimpleMemory: 通用占位记忆模块，将来可以换成 memory_layer
# ======================================================================

class _SimpleMemory:
    """
    Lightweight A-mem-style memory bank.

    - 存储若干条文本 note（步经验、episode 总结等）
    - 用简单 token overlap 做检索，接口与真正记忆系统对齐：
        - add(content, meta)
        - retrieve(query, k) -> str
    """

    def __init__(self, max_notes: int = 2000):
        self.max_notes = max_notes
        self.notes: List[Dict[str, Any]] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def add(self, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.notes.append({"content": content, "meta": meta or {}})
        if len(self.notes) > self.max_notes:
            self.notes = self.notes[-self.max_notes :]

    def retrieve(self, query: str, k: int = 6) -> str:
        q = set(self._tokenize(query))
        if not q or not self.notes:
            return ""
        scored: List[Tuple[float, str]] = []
        for n in self.notes[-self.max_notes :]:
            t = set(self._tokenize(n["content"]))
            score = len(q & t) / max(1, len(q))
            if score > 0:
                scored.append((score, n["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return "\n".join([s for _, s in scored[:k]])


# ======================================================================
# 统一 A-mem Agent：BizBench + BeerGame + Consulting + EDT
# ======================================================================


class AMemAgent:
    """
    统一的 A-mem Agent。

    - BizBench: 静态 QA 数据集，经由 AMemBizBenchGenerator + evaluate_test_set
    - BeerGame: 严肃游戏，通过 MCP + evaluate_beergame_set
    - Consulting: case interview，通过 evaluate_consulting_set
    - EDT: 严肃游戏，通过 MCP + evaluate_edt_set

    统一入口：
        run(mode, test_samples, data_processor, config)

      * 由 task_name 自动路由：
          - task_name 包含 'beer' → run_beergame(...)
          - task_name 包含 'consult' → run_consulting(...)
          - task_name 包含 'edt' → run_edt(...)
          - 其他 → run_bizbench(...)
    """

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int = 4096,
        agent_method: str = "amem",
        *,
        temperature: float = 0.2,
        retrieve_k: int = 4,
        max_order_qty: int = 5000,
        note_every_n_steps: int = 2,
        summarize_every_n_steps: int = 4,
        # StructuredReasoning online write-back controls
        writeback: bool = True,
        writeback_max_chars_question: int = 1200,
        writeback_max_chars_context: int = 2000,
    ):
        self.api_provider = api_provider
        self.generator_model = generator_model
        self.max_tokens = max_tokens
        self.agent_method = agent_method

        self.temperature = float(temperature)
        self.retrieve_k = int(retrieve_k)
        self.max_order_qty = int(max_order_qty)
        self.note_every_n_steps = max(1, int(note_every_n_steps))
        self.summarize_every_n_steps = max(1, int(summarize_every_n_steps))

        # 通用 client：BeerGame & BizBench & EDT & Consulting 共享
        self.generator_client, _, _ = initialize_clients(api_provider)

        # 记忆：目前用轻量 SimpleMemory，将来可替换为真正 AMemMemory
        self.memory = AMemMemory(
            embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
            evo_threshold=100,
            default_k=self.retrieve_k,
            llm_model=generator_model,
            llm_backend="openai",
        )
        # self.memory = _SimpleMemory()

        # 轻量节流，避免 rate_limit_check_failed
        self.min_interval_sec = float(os.getenv("MIN_REQUEST_INTERVAL_SEC", "1.2"))
        self._last_call_ts = 0.0

        # BizBench generator: 单独放在 generator.py
        self.bizbench_generator = AMemBizBenchGenerator(
            client=self.generator_client,
            api_provider=self.api_provider,
            model=self.generator_model,
            max_tokens=self.max_tokens,
            memory=self.memory,
            retrieve_k=self.retrieve_k,
            temperature=self.temperature,
        )

        self.writeback = bool(writeback)
        self.writeback_max_chars_question = max(200, int(writeback_max_chars_question))
        self.writeback_max_chars_context = max(200, int(writeback_max_chars_context))
        self._mem_lock = threading.Lock()

    # ==================================================================
    # 公共工具：节流 + LLM JSON 调用
    # ==================================================================

    def _throttle(self) -> None:
        now = time.time()
        wait = self.min_interval_sec - (now - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    def _call_llm_json(self, system: str, user: str) -> Dict[str, Any]:
        """
        通用 JSON 调用：
        - 提示模型输出 JSON
        - 收到文本后尽量截取 {...} 再解析
        """
        import json as _json

        self._throttle()
        resp = self.generator_client.chat.completions.create(
            model=self.generator_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=min(self.max_tokens, 4096),
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

    # ==================================================================
    # Consulting：Agent 提供给评测框架的三个接口
    # ==================================================================

    def on_case_start(self, case_id: str) -> None:
        """
        一场咨询 Case 开始时调用。
        当前策略：每个 case 重置一次 memory，将记忆视为短期工作记忆。
        后续如果你想做“跨 case 长期记忆”，可以改成只清理部分记忆或用分桶。
        """
        # 简单做法：彻底清空，再新建一个 SimpleMemory
        pass

    def on_case_end(
        self,
        case_id: str,
        case_text: str,
        history: List[Dict[str, str]],
    ) -> None:
        """
        一场咨询 Case 结束时调用。
        - 当前实现：把完整 case 文本 + 对话 transcript 存成一条 note 写入 memory。
        - 将来你可以在这里做 meta-learning，总结策略等。
        """
        transcript = "\n".join(
            f"{turn.get('speaker', 'Unknown')}: {turn.get('text', '')}"
            for turn in history
        )
        note = (
            f"[CONSULTING][case_id={case_id}] CASE_TEXT:\n{case_text}\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )
        self.memory.add(note, meta={"case_id": case_id, "type": "case_summary"})

    def reply(self, case_id: str, history: List[Dict[str, str]]) -> str:
        """
        Consulting 候选人回复接口。
        - 与任务固有的 prompt / query / 解析逻辑已下沉到 utils.consulting_tools。
        - agent.py 仅负责：记忆检索、调用模型、写入记忆。
        """
        state = consulting_build_candidate_query(case_id=case_id, history=history)

        retrieved = self.memory.retrieve(state["query"], k=self.retrieve_k)

        system, user = consulting_render_candidate_prompt(
            case_id=case_id,
            state=state,
            retrieved=retrieved,
        )
        data = self._call_llm_json(system=system, user=user)
        reply = consulting_extract_candidate_reply(data)
        return reply

    def _record_edt_experience(
        self,
        scenario: Dict[str, Any],
        schema: Dict[str, Any],
        metrics: Dict[str, Any],
        repeat_idx: int,
    ) -> None:
        """
        EDT 场景重复运行后的记忆写入接口（online 模式用）。

        预期由 utils.seriousgame_tools.evaluate_edt_set 在每次重复实验结束后调用：
            agent._record_edt_experience(scenario, schema, metrics, repeat_idx)

        参数
        ----
        scenario : dict
            原始测试样本（base scenario 的描述、配置等）。
        schema : dict
            本次 run 中 agent 输出的「原始 schema」（LLM 输出的最精简 JSON）。
        metrics : dict
            本次 run 的学习用指标（通常是那 5 个：accumulated_earnings、
            accumulated_revenue、accumulated_expenses、overall_profit_margin、
            overall_avg_utilization；如没有则可能是全量 metrics）。
        repeat_idx : int
            同一 base scenario 下的第几次重复试验（0-based）。
        """
        if self.memory is None:
            return

        # 如果全局关闭 writeback，则直接跳过
        if not getattr(self, "writeback", True):
            return

        # 1) 场景 id 方便检索
        try:
            scenario_id = (
                scenario.get("scenario_id")
                or scenario.get("id")
                or scenario.get("name")
                or "unknown"
            )
        except Exception:
            scenario_id = "unknown"

        # 2) 序列化 scenario / schema / metrics，适当截断，控制长度
        try:
            scenario_repr = json.dumps(scenario, ensure_ascii=False)
        except Exception:
            scenario_repr = str(scenario)
        if len(scenario_repr) > self.writeback_max_chars_context:
            scenario_repr = (
                scenario_repr[: self.writeback_max_chars_context] + "...(truncated)"
            )

        try:
            schema_repr = json.dumps(schema, ensure_ascii=False)
        except Exception:
            schema_repr = str(schema)
        if len(schema_repr) > 800:
            schema_repr = schema_repr[:800] + "...(truncated)"

        try:
            metrics_repr = json.dumps(metrics, ensure_ascii=False)
        except Exception:
            metrics_repr = str(metrics)

        # 3) 拼成一条记忆文本，打上简洁 tag，便于后续通过 query 检索
        note_lines = [
            f"[EDT][scenario_id={scenario_id}][repeat={int(repeat_idx)}]",
            "",
            "Scenario (truncated JSON):",
            scenario_repr,
            "",
            "Chosen schema (agent output):",
            schema_repr,
            "",
            "Outcome metrics (learning signals):",
            metrics_repr,
        ]
        note = "\n".join(note_lines).strip()

        meta = {
            "task": "edt",
            "scenario_id": str(scenario_id),
            "repeat_idx": int(repeat_idx),
        }

        # 4) 实际写入 AMemMemory；加锁保证线程安全
        try:
            with self._mem_lock:
                self.memory.add(note, meta=meta)
        except Exception:
            # 记忆写入失败不应该中断 EDT 仿真主流程
            return

    # ==================================================================
    # EDT 决策
    # ==================================================================

    async def _decide_edt_scenario_schema(
        self,
        base_summary: Dict[str, Any],
        scenario_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        EDT 场景 schema 决策（AMem 版本，带记忆检索）。

        流程：
        1) 用 base_summary + scenario_meta 构造 EDT 决策上下文 ctx；
        2) 基于 scenario_id 优先检索该场景的历史 EDT 经验，再补充全局 EDT 经验；
        3) 将检索结果直接拼接在 EDT user prompt 的末尾，由本方法负责拼接；
        4) 调模型输出 raw schema，并用 normalize_edt_schema 做归一化与防退化。
        """
        # 1) 构造通用 ctx（含 horizon 兜底推断、default_P 等）
        ctx = build_edt_decision_context(
            base_summary=base_summary,
            scenario_meta=scenario_meta,
            max_steps_hint=(scenario_meta or {}).get("max_steps"),
        )

        # 2) 设计检索 query：优先同一 scenario_id，再补充所有 EDT 经验
        retrieved_blocks: List[str] = []
        if self.memory is not None:
            try:
                # 尝试拿一个可读的 scenario_id
                scenario_id = (
                    (scenario_meta or {}).get("scenario_id")
                    or ctx.get("scenario_id")
                    or "unknown"
                )
                scenario_id = str(scenario_id)

                # (a) 主查询：同一 scenario_id 的 EDT 经验
                primary_query = f"[EDT][scenario_id={scenario_id}]"
                primary_text = self.memory.retrieve(primary_query, k=self.retrieve_k)
                if primary_text and primary_text.strip():
                    retrieved_blocks.append(primary_text.strip())

                # (b) 补充查询：全局 EDT 经验（主查询太少时再补）
                combined_len = sum(len(b) for b in retrieved_blocks)
                if combined_len < 1000:
                    fallback_query = "[EDT]"
                    k_fb = max(1, self.retrieve_k // 2)
                    fallback_text = self.memory.retrieve(fallback_query, k=k_fb)
                    if fallback_text and fallback_text.strip():
                        retrieved_blocks.append(fallback_text.strip())
            except Exception:
                # 记忆检索失败不影响主流程
                retrieved_blocks = []

        retrieved_text = "\n\n".join(retrieved_blocks).strip()
        if retrieved_text and len(retrieved_text) > 4000:
            retrieved_text = retrieved_text[:4000] + "\n...(truncated)"

        # 3) 先用工具层构造 EDT 基础 prompt，再在 agent 层拼接“历史经验块”
        system, base_user = render_edt_prompt(ctx)

        if retrieved_text:
            user = (
                base_user
                + "\n\n=== PAST POLICIES AND OUTCOMES FROM PREVIOUS RUNS ===\n"
                  "The following notes summarize your previous EDT simulation runs "
                  "stored in memory. Use them as qualitative hints when choosing C, "
                  "R and project windows, but still reason explicitly about this run "
                  "and do not copy any past decision verbatim.\n\n"
                + retrieved_text
            )
        else:
            user = base_user

        # 4) 调用 LLM，解析 JSON，并做归一化
        raw = self._call_llm_json(system=system, user=user)
        return normalize_edt_schema(raw, ctx)

    # ==================================================================
    # BeerGame 决策 + 运行函数
    # ==================================================================
    def _decide_order_qty(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """
        BeerGame 单步决策（精简版）：
        - 任务固有的 query/prompt/基线/解析 全部委托给 utils.seriousgame_tools
        - agent.py 仅保留：记忆检索、调用模型、写入记忆
        """
        role = str(ctx.get("role", obs.get("role", "retailer")))
        week = int(obs.get("week", 0) or 0)

        query = beergame_build_query(obs)
        retrieved = self.memory.retrieve(query, k=self.retrieve_k)

        base_order = beergame_base_rule_order(
            obs=obs,
            ctx=ctx,
            max_order_qty=self.max_order_qty,
        )

        system, user = beergame_render_prompt(
            role=role,
            obs=obs,
            retrieved=retrieved,
            base_order=base_order,
        )

        js = self._call_llm_json(system=system, user=user)
        order_qty, note = beergame_extract_order_and_note(
            js=js,
            base_order=base_order,
            max_order_qty=self.max_order_qty,
        )

        # cache latest BeerGame explanation for logging
        # self._last_beergame_note = note
        self._last_beergame_note = note

        if note:
            self.memory.add(
                content=beergame_format_memory_note(
                    role=role, week=week, obs=obs, order_qty=order_qty, note=note
                ),
                meta={"role": role, "week": week},
            )

        return order_qty
    def run_beergame(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
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

    # ==================================================================
    # EDT 专用：运行函数（接口与 BeerGame/BizBench/Consulting 对齐）
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

    # ==================================================================
    # BizBench 专用：静态 QA 数据集
    # ==================================================================
    def add_memory(self, question, context, response, target, is_correct, call_id):
        q_short = (question or "").strip()
        c_short = (context or "").strip()
        if len(q_short) > self.writeback_max_chars_question:
            q_short = q_short[: self.writeback_max_chars_question] + "…"
        if len(c_short) > self.writeback_max_chars_context:
            c_short = c_short[: self.writeback_max_chars_context] + "…"
        note_lines = [
            f"[StructuredReasoning]",
            "Question:",
            q_short or "(empty)",
            "",
            "Context:",
            c_short or "(empty)",
            "",
            "Your Answer:",
            str(response).strip(),
            "",
            "Ground Truth Answer:",
            str(target).strip(),
            "",
            f"You answer is considered {'correct' if is_correct else 'incorrect'}.",
        ]
        note = "\n".join(note_lines).strip()
        meta = {
            "type": "structured_reasoning_qa",
            "call_id": call_id,
        }
        with self._mem_lock:
            self.memory.add(note, meta=meta)

    def run_bizbench(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        BizBench 评测入口。由 bizbench/run.py 调用，或者由 run() 自动路由。
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )
        if not test_samples:
            raise ValueError("BizBench requires non-empty test_samples")
        if data_processor is None:
            raise ValueError("BizBench tasks require a non-None data_processor")

        task_name = config.get("task_name", getattr(data_processor, "task_name", "BizBench"))
        save_dir = config.get("save_dir", "results")

        run_subdir = (
            f"{task_name}/{self.agent_method}/{mode}/"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - BIZBENCH EVALUATION")
        print(f"{'='*60}")
        print(f"Task: {task_name}")
        print(f"Samples: {len(test_samples)}")
        print(f"Log dir: {log_dir}")
        print(f"{'='*60}\n")

        results, error_log = evaluate_test_set(
            agent=self,
            data_processor=data_processor,
            generator=self.bizbench_generator,
            playbook="",
            test_samples=test_samples,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            max_workers=config.get("test_workers", 20),
            use_json_mode=config.get("json_mode", False),
            mode=mode,
            batch_size=int(config.get("batch_size", 20)),
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

        config_payload = dict(config)
        config_payload["run_subdir"] = run_subdir
        config_payload["resolved_save_path"] = resolved_save_path

        with open(
            os.path.join(resolved_save_path, "run_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config_payload, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - BIZBENCH RUN COMPLETE")
        print(f"{'='*60}")
        if "accuracy" in results:
            print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Results saved to: {resolved_save_path}")
        print(f"{'='*60}\n")

        return results

    # ==================================================================
    # Consulting：完整评测入口
    # ==================================================================

    def run_consulting(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
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


    # ==================================================================
    # 统一入口：四个数据集共用 .run()
    # ==================================================================

    def run(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        统一入口，由 task_name 自动选择具体数据集的运行函数。

        - BeerGame: task_name 中包含 'beer'
        - EDT: task_name 中包含 'edt'
        - Consulting: task_name 中包含 'consult'
        - BizBench: 其他情况默认按 BizBench 处理
        """
        if test_samples is None:
            raise ValueError("test_samples must not be None")

        task_name = str(config.get("task_name", getattr(data_processor, "task_name", "")))
        name_lower = task_name.lower()
        print(f"[AMemAgent] task_name={task_name}")

        if "beer" in name_lower:
            return self.run_beergame(mode, test_samples, data_processor, config)
        if "edt" in name_lower:
            return self.run_edt(mode, test_samples, data_processor, config)
        if "consult" in name_lower:
            return self.run_consulting(mode, test_samples, data_processor, config)

        # 默认走 BizBench
        return self.run_bizbench(mode, test_samples, data_processor, config)
