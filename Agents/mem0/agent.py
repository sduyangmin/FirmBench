"""
Mem0-based agent for FinAgent BizBench-style (StructuredReasoning) evaluation.

设计思路
-------
- 复用 AMem 的 BizBench Generator (AMemBizBenchGenerator)，只替换记忆后端为 mem0.Memory。
- 对上层评测框架暴露与 AMemAgent.run_bizbench 相同的接口：
    - Mem0Agent.run(...)
    - Mem0Agent.run_bizbench(...)
- 当前版本仅支持 BizBench / StructuredReasoning 任务：
    - task_name 中不包含 "beer" / "edt" / "consult" 时，走 BizBench 流程；
    - 其它任务会抛出 NotImplementedError，后续可以按 amem 的模式逐步补齐。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import threading
from utils.tools import initialize_clients
from utils.memory_tools import evaluate_test_set
from Agents.amem.generator import AMemBizBenchGenerator
import time
from utils.consulting_tools import (
    consulting_prepare_run,
    consulting_evaluate_run,
    consulting_save_run,
    consulting_build_candidate_query,
    consulting_render_candidate_prompt,
    consulting_extract_candidate_reply,
    consulting_format_memory_note,
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
    beergame_format_memory_note,
)


try:
    # mem0 官方包
    from mem0 import Memory as _RawMemory  # type: ignore
except Exception:  # ImportError 等
    _RawMemory = None  # 在未安装 mem0 时给出清晰错误

OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', "http://35.220.164.252:3888/v1/")
API_KEY = os.getenv('OPENAI_API_KEY', '')

class Mem0Memory:
    """
    轻量封装 mem0.Memory，使其暴露统一接口：

        add(content: str, meta: Optional[Dict] = None) -> None
        retrieve(query: str, k: int = 6) -> str

    这样可以直接被 AMemBizBenchGenerator 复用（其只依赖 .retrieve）。
    """

    def __init__(
        self,
        user_id: str = "mem0_agent",
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        if self.config is None:
            self.config = {
                "llm": {  # 这是主 LLM，Mem0 用它来做信息抽取/关系抽取等
                    "provider": "openai",
                    "config": {
                        "api_key": API_KEY,  # 你自己的 key，随便起名也行，只要服务端接受
                        "openai_base_url": OPENAI_BASE_URL,
                        "model": "deepseek-v3",  # 这里写你那台服务暴露出来的模型名
                        "temperature": 0.1,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "api_key": API_KEY,  # 你自己的 key，随便起名也行，只要服务端接受
                        "openai_base_url": OPENAI_BASE_URL,
                        "model": "text-embedding-3-small",  # 这里写你那台服务暴露出来的模型名
                        "embedding_dims": 1536
                    },
                }
            }
        if _RawMemory is None:
            raise ImportError(
                "mem0 package is not installed. Please install `mem0` before using Mem0Agent."
            )

        # 简单策略：直接用默认配置初始化 Memory；
        # 如有需要，可以通过 config 传入 from_config 的参数。
        self._mem = _RawMemory.from_config(self.config)

        self._user_id = user_id
        self._run_id = run_id

    def add(self, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        将一条文本记忆写入 mem0。

        这里采用最简单的形式：单轮 user 消息，不再额外让 mem0 调 LLM 进行抽取，
        避免在线评测时出现双重 LLM 调用。
        """
        messages = [{"role": "user", "content": content}]
        try:
            self._mem.add(
                messages=messages,
                user_id=self._user_id,
                run_id=self._run_id,
                metadata=meta or {},
                infer=False,  # 只做 embedding/索引，不再触发额外 LLM
            )
        except Exception:
            # 记忆写入失败不应中断评测，直接忽略异常
            return

    def retrieve(self, query: str, k: int = 6) -> str:
        """
        根据 query 从 mem0 中检索前 k 条记忆，拼接为一个多行字符串返回。
        """
        try:
            res = self._mem.search(
                query=query,
                user_id=self._user_id,
                run_id=self._run_id,
                limit=int(k),
            )
        except Exception:
            return ""

        if isinstance(res, dict):
            hits = res.get("results", [])
        else:
            hits = []

        texts: List[str] = []
        for h in hits:
            # README 示例中使用 entry["memory"] 存储文本
            mem_text = h.get("memory") or h.get("content") or ""
            if mem_text:
                texts.append(str(mem_text))

        return "\n".join(texts)


class Mem0Agent:
    """
    Mem0 Agent：使用 mem0 作为长时记忆后端的 BizBench / StructuredReasoning 基线。

    - 评测入口：run(...)
    - 当前版本：仅支持 BizBench / StructuredReasoning 任务
    """

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int = 4096,
        agent_method: str = "mem0",
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
    ) -> None:
        self.api_provider = api_provider
        self.generator_model = generator_model
        self.max_tokens = int(max_tokens)
        self.agent_method = agent_method

        self.temperature = float(temperature)
        self.retrieve_k = int(retrieve_k)
        self.max_order_qty = int(max_order_qty)
        self.note_every_n_steps = max(1, int(note_every_n_steps))
        self.summarize_every_n_steps = max(1, int(summarize_every_n_steps))
        self._last_beergame_note = None  # cache latest BeerGame explanation
        # 轻量节流，避免 rate_limit 错误
        self.min_interval_sec = float(os.getenv("MIN_REQUEST_INTERVAL_SEC", "1.2"))
        self._last_call_ts = 0.0

        # 通用 LLM client：沿用 utils.tools.initialize_clients 的行为
        self.generator_client, _, _ = initialize_clients(api_provider)

        # mem0 记忆后端
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.memory = Mem0Memory(
            user_id=f"mem0_{agent_method}",
            run_id=run_id,
            config=None,  # 如需自定义存储/向量后端可在此传入
        )
        self.writeback = bool(writeback)
        self.writeback_max_chars_question = max(200, int(writeback_max_chars_question))
        self.writeback_max_chars_context = max(200, int(writeback_max_chars_context))
        self._mem_lock = threading.Lock()
        # BizBench generator：直接复用 AMem 的实现，只是 memory 换成 Mem0Memory
        self.bizbench_generator = AMemBizBenchGenerator(
            client=self.generator_client,
            api_provider=self.api_provider,
            model=self.generator_model,
            max_tokens=self.max_tokens,
            memory=self.memory,
            retrieve_k=self.retrieve_k,
            temperature=self.temperature,
        )

    
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
        # OpenAI-style client: message content 在 choices[0].message.content
        text = getattr(resp.choices[0].message, "content", "") or ""
        if not isinstance(text, str):
            text = str(text)

        # 尽力从返回中截取一个 JSON 对象
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
    # Consulting：Agent 提供给评测框架的接口
    # ==================================================================
    def on_case_start(self, case_id: str) -> None:
        """
        一场咨询 Case 开始时调用。

        当前实现不重置 mem0 记忆，而是让不同 case 共享一个长期记忆池。
        如果你希望每个 case 使用独立记忆，可以在这里重建 self.memory。
        """
        return

    def on_case_end(
        self,
        case_id: str,
        case_text: str,
        history: List[Dict[str, str]],
    ) -> None:
        """
        一场咨询 Case 结束时调用。
        - 当前实现：把完整 case 文本 + 对话 transcript 存成一条 note 写入 mem0。
        """
        transcript = "\n".join(
            f"{turn.get('role', turn.get('speaker', 'Unknown'))}: {turn.get('content', turn.get('text', ''))}"
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
        - 本类仅负责：使用 mem0 检索记忆、调用模型、写入记忆。
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

    def run_consulting(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Consulting 评测入口。接口与 AMemAgent.run_consulting 保持一致。
        """
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
    # BeerGame：单步决策 + 评测入口
    # ==================================================================
    def _decide_order_qty(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """
        BeerGame 单步决策（mem0 版本）：
        - 任务固有的 query/prompt/基线/解析 全部委托给 utils.seriousgame_tools
        - 本类仅保留：记忆检索、调用模型、写入记忆
        """
        role = str(ctx.get("role", obs.get("role", "retailer")))
        week = int(obs.get("week", 0) or 0)

        # 基于当前观测构造检索 query
        query = beergame_build_query(obs)
        retrieved = self.memory.retrieve(query, k=self.retrieve_k)

        # 简单基线策略：用于 fallback 和约束模型输出范围
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
        # expose latest note to seriousgame_tools for logging
        self._last_beergame_note = note


        # 将带有 note 的决策写入 mem0，作为后续 episodes 的经验
        if note:
            self.memory.add(
                content=beergame_format_memory_note(
                    role=role,
                    week=week,
                    obs=obs,
                    order_qty=order_qty,
                    note=note,
                ),
                meta={"role": role, "week": week},
            )

        return int(order_qty)

    def run_beergame(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        BeerGame 评测入口。接口与 AMemAgent.run_beergame 保持一致。
        """
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


    def run_edt(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        EDT 数据集评测入口。接口与 AMemAgent.run_edt 保持一致。
        """
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

# ============================================================
    # BizBench / StructuredReasoning
    # ============================================================
    def add_memory(self, question, context, response, target, is_correct, call_id):
        q_short = (question or "").strip()
        c_short = (context or "").strip()
        if len(q_short) > self.writeback_max_chars_question:
            q_short = q_short[: self.writeback_max_chars_question] + "…"
        if len(c_short) > self.writeback_max_chars_context:
            c_short = c_short[: self.writeback_max_chars_context] + "…"

        note_lines = [
            "[StructuredReasoning]",
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
        # 这里真正写入 mem0 的记忆后端
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
        BizBench 评测入口。与 AMemAgent.run_bizbench 接口保持一致。
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
        print(f"{self.agent_method.upper()} - BIZBENCH (MEM0) EVALUATION")
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

        # 保存结果
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
        print(f"{self.agent_method.upper()} - BIZBENCH (MEM0) RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Correct: {results.get('correct')}/{results.get('total')}")
        print(f"Accuracy: {results.get('accuracy')}")
        print(f"Results saved to: {resolved_save_path}")
        print(f"{'='*60}\n")

        return results

    # ============================================================
    # 统一入口
    # ============================================================
    
    
    def run(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        统一入口：
        - BizBench / StructuredReasoning
        - Consulting
        - BeerGame
        其它任务（EDT）暂不支持。
        """
        if test_samples is None:
            raise ValueError("test_samples must not be None")

        task_name = str(config.get("task_name", getattr(data_processor, "task_name", "")))
        name_lower = task_name.lower()
        print(f"[Mem0Agent] task_name={task_name}")

        if "consult" in name_lower:
            return self.run_consulting(mode, test_samples, data_processor, config)
        if "beer" in name_lower:
            return self.run_beergame(mode, test_samples, data_processor, config)
        if "edt" in name_lower:
            return self.run_edt(mode, test_samples, data_processor, config)

        # 默认走 BizBench
        return self.run_bizbench(mode, test_samples, data_processor, config)
    # ==================================================================
    # EDT 决策
    # ==================================================================

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

        参数说明
        ----------
        scenario : dict
            原始测试样本（base scenario 的描述、配置等），不做假设，直接序列化为文本。
        schema : dict
            本次 run 中 agent 输出的「原始 schema」（尚未或已经 normalize 均可，
            这里仅作为记录）。
        metrics : dict
            本次 run 的学习用指标（例如 accumulated_earnings 等 5 个量，或包含更多字段）。
        repeat_idx : int
            同一 base scenario 下的第几次重复试验（从 0 或 1 开始均可，由调用端约定）。
        """
        # 如果全局关闭 writeback，这里直接跳过，避免在纯评测模式下写记忆
        if not getattr(self, "writeback", True):
            return
        # 1) 尝试识别一个可读的 scenario_id，方便后续检索
        try:
            scenario_id = (
                scenario.get("scenario_id")
                or scenario.get("id")
                or scenario.get("name")
                or "unknown"
            )
        except Exception:
            scenario_id = "unknown"

        # 2) 将 scenario / schema / metrics 序列化为文本，并做长度裁剪
        try:
            scenario_repr = json.dumps(scenario, ensure_ascii=False)
        except Exception:
            scenario_repr = str(scenario)

        max_ctx = int(getattr(self, "writeback_max_chars_context", 2000))
        if len(scenario_repr) > max_ctx:
            scenario_repr = scenario_repr[:max_ctx] + "...(truncated)"

        try:
            schema_repr = json.dumps(schema, ensure_ascii=False)
        except Exception:
            schema_repr = str(schema)
        # schema 通常比 scenario 短，这里给一个单独上限以防万一
        if len(schema_repr) > 800:
            schema_repr = schema_repr[:800] + "...(truncated)"

        try:
            metrics_repr = json.dumps(metrics, ensure_ascii=False)
        except Exception:
            metrics_repr = str(metrics)

        # 3) 组织成一条 mem0 的「记忆文本」
        note_lines = [
            f"The result of simulation {int(repeat_idx)}:",
            "Chosen schema (agent output):",
            schema_repr,
            "",
            "Outcome metrics (learning signals):",
            metrics_repr,
        ]
        note = "\n".join(note_lines)

        meta = {
            "task": "edt",
            "scenario_id": str(scenario_id),
            "repeat_idx": int(repeat_idx),
        }
        # 4) 实际写入 mem0；加锁避免多线程环境下竞争（与 BizBench 写回逻辑保持一致风格）
        try:
            lock = getattr(self, "_mem_lock", None)
            if lock is not None:
                with lock:
                    self.memory.add(note, meta=meta)
            else:
                self.memory.add(note, meta=meta)
        except Exception:
            # 记忆写入失败不应该中断评测，静默忽略
            return

    async def _decide_edt_scenario_schema(
        self,
        base_summary: Dict[str, Any],
        scenario_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        EDT 场景 schema 决策（mem0 版本）。
        逻辑与 AMem 保持一致：构造 ctx -> 渲染 prompt -> 调 LLM -> 归一化 schema。
        """
        ctx = build_edt_decision_context(
            base_summary=base_summary,
            scenario_meta=scenario_meta,
            max_steps_hint=(scenario_meta or {}).get("max_steps"),
        )

        # 2) 设计检索 query：优先匹配同一 scenario_id，其次匹配所有 EDT 经验
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
                primary_query = f"What is the chosen schema and the corresponding outcome of all simulations?"
                primary_text = self.memory.retrieve(primary_query, k=self.retrieve_k)
                if primary_text and primary_text.strip():
                    retrieved_blocks.append(primary_text.strip())

                # (b) 补充查询：所有 EDT 经验（如果主查询结果很少）
                combined_len = sum(len(b) for b in retrieved_blocks)
                if combined_len < 1000:  # 经验太少时再补一点全局的
                    fallback_query = "[EDT]"
                    k_fb = max(1, self.retrieve_k // 2)
                    fallback_text = self.memory.retrieve(fallback_query, k=k_fb)
                    if fallback_text and fallback_text.strip():
                        retrieved_blocks.append(fallback_text.strip())

            except Exception:
                # 检索失败不影响主流程
                retrieved_blocks = []

        # 合并 & 截断检索文本，避免 prompt 爆炸
        retrieved_text = "\n\n".join(retrieved_blocks).strip()
        if retrieved_text and len(retrieved_text) > 4000:
            retrieved_text = retrieved_text[:4000] + "\n...(truncated)"

        # 3) 先用工具函数构造基础 EDT system/user prompt
        system, base_user = render_edt_prompt(ctx)

        # 再在 Agent 这一层，把检索到的经验直接拼接到 user prompt 尾部
        if retrieved_text:
            user = (
                base_user
                + "\n\n=== PAST POLICIES AND OUTCOMES FROM PREVIOUS RUNS ===\n"
                  "The following notes summarize your previous EDT simulation runs stored in memory. "
                  "Each block typically records your chosen C, R and project windows together with "
                  "outcome metrics such as accumulated earnings, revenue, expenses, and profit margin.\n\n"
                  "Use these as qualitative hints, not ground truth. In particular:\n"
                  "- If many past runs with similar C/R/P configurations produced negative or very low profit,\n"
                  "  actively explore different strategies in this run (e.g. change C, adjust R up/down,\n"
                  "  or enable/disable different projects) instead of repeating the same choices.\n"
                  "- If some past runs achieved clearly better profit, treat them as promising directions,\n"
                  "  but still introduce some variation rather than copying any past decision verbatim.\n\n"
                  "Balance exploitation of good patterns with exploration of new strategies, especially\n"
                  "when past performance is poor. You should make a fresh, well-reasoned choice of C, R,\n"
                  "and project windows for THIS run, informed but not constrained by past runs.\n\n"
                + retrieved_text
            )
        else:
            user = base_user

        # 4) 调用 LLM，解析为 JSON，再归一化 schema
        raw = self._call_llm_json(system=system, user=user)
        return normalize_edt_schema(raw, ctx)


