"""
A-mem style generator for BizBench QA-style tasks.

设计目标
--------
- 与 ChainOfThoughtGenerator 的接口保持一致（generate(...)）。
- 不依赖 ACE 的 playbook，只使用 LLM + （可选）记忆检索。
- 输出字符串中应包含 JSON，对应字段：
    {
      "reasoning": "...",
      "bullet_ids": [],
      "final_answer": "..."
    }
  方便 utils.tools.extract_answer 解析。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from utils.llm import timed_llm_call


class AMemBizBenchGenerator:
    """
    A-mem 风格的 BizBench generator。

    注意：
    - 这里不负责创建 API 客户端，由外部传入 client。
    - memory 只需要实现 .retrieve(query: str, k: int) -> str 即可（当前用 SimpleMemory 占位）。
    """

    def __init__(
        self,
        client,
        api_provider: str,
        model: str,
        max_tokens: int = 4096,
        *,
        memory: Optional[Any] = None,
        retrieve_k: int = 0,
        temperature: float = 0.2,
    ):
        self.client = client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.memory = memory
        self.retrieve_k = max(0, int(retrieve_k))
        self.temperature = float(temperature)

    def _build_prompt(
        self,
        question: str,
        context: str,
        retrieved: str = "",
    ) -> str:
        question = (question or "").strip() or "(Question is empty.)"
        context = (context or "").strip() or "(No additional context provided.)"

        lines: List[str] = [
            "You are an advanced financial reasoning assistant.",
            "You will be given a task description and possibly additional context.",
            "Think step by step and solve the problem carefully.",
            "",
            "You MUST respond strictly in JSON with the fields:",
            "{",
            '  "reasoning": "<step-by-step reasoning>",',
            '  "bullet_ids": [],',
            '  "final_answer": "<final concise answer>"',
            "}",
            "",
            "If the question specifies an answer format (for example numerical formatting),",
            "you MUST follow that format in `final_answer`.",
            "",
            "Question (may include task-specific instructions):",
            question,
            "",
            "Context:",
            context,
        ]

        if retrieved:
            lines += [
                "",
                "Relevant notes from your long-term memory (may be useful, but not mandatory):",
                retrieved,
            ]

        lines += [
            "",
            "Respond strictly in JSON. Do not add any extra text outside the JSON object.",
        ]

        return "\n".join(lines)

    def generate(
        self,
        question: str,
        playbook: str = "",
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "amem_bizbench",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        与 evaluate_test_set 要求的接口保持一致：
        返回 (response_text, bullet_ids, call_info)
        """
        # 1) 简单 memory 检索（如提供）
        retrieved = ""
        if self.memory is not None and self.retrieve_k > 0:
            query = f"{question}\n\n{context}"
            try:
                retrieved = self.memory.retrieve(query, k=self.retrieve_k)
            except Exception:
                retrieved = ""

        # 2) 构造 prompt
        prompt = self._build_prompt(question, context, retrieved)

        # 3) 调用 LLM（带计时与重试）
        response, call_info = timed_llm_call(
            client=self.client,
            api_provider=self.api_provider,
            model=self.model,
            prompt=prompt,
            role="amem_bizbench",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )

        # 当前不使用 bullet_ids，返回空列表即可
        return response, [], call_info
