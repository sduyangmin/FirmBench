"""
Chain-of-thought baseline generator implementation.
"""

from __future__ import annotations

from typing import Tuple, List, Optional

from utils.tools import initialize_clients
from utils.llm import timed_llm_call


class ChainOfThoughtGenerator:
    """
    Lightweight generator that produces reasoning traces without relying on the ACE playbook.
    """

    def __init__(self, api_provider: str, model_name: str, max_tokens: int):
        generator_client, _, _ = initialize_clients(api_provider)
        self.client = generator_client
        self.api_provider = api_provider
        self.model = model_name
        self.max_tokens = max_tokens

    def _build_prompt(self, question: str, context: str) -> str:
        question_text = (question or "").strip() or "(Question is empty.)"
        context_text = (context or "").strip() or "(No additional context provided.)"
        prompt_lines = [
            "You are a finance-domain reasoning assistant. Think carefully and solve the task step by step.",
            "Follow the answer-format instructions exactly as they appear inside the question text.",
            "Return a JSON object with the keys `reasoning`, and `final_answer`.",
            "- `reasoning`: include the full chain of thought, calculations, and intermediate steps.",
            "- `final_answer`: restate the answer exactly as required by the question instructions.",
            "",
            "Question (includes any task-specific instructions):",
            question_text,
            "",
            "Context:",
            context_text,
            "",
            "Respond strictly in JSON.",
        ]
        return "\n".join(prompt_lines)

    def generate(
        self,
        question: str,
        playbook: str = "",
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "cot",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], dict]:
        prompt = self._build_prompt(question, context)
        response, call_info = timed_llm_call(
            self.client,
            self.api_provider,
            self.model,
            prompt,
            role="cot_generator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )
        return response, [], call_info

