from __future__ import annotations

import json
import re
from typing import List, Tuple, Optional, Dict

from utils.tools import initialize_clients
from utils.llm import timed_llm_call


class ReflexionInitial:
    """初始回答角色"""

    def __init__(self, client, api_provider: str, model: str, max_tokens: int, temperature: float):
        self.client = client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_prompt(self, question: str, context: str, prior_reflections: Optional[List[str]]) -> str:
        prior_block = ""
        if prior_reflections:
            bullets = "\n- ".join([r.strip() for r in prior_reflections if r.strip()])
            prior_block = "Useful reflections from past similar attempts:\n- " + bullets + "\n\n"

        prompt_lines = [
            "You are a finance-domain reasoning assistant. Think carefully and solve the task step by step.",
            "Provide your best answer with detailed reasoning.",
            "Return a JSON object with ONLY keys `reasoning` and `final_answer`.",
            "- `reasoning`: include the full chain of thought, calculations, and intermediate steps (include code if helpful).",
            "- `final_answer`: strictly follow the required answer format (plain number, [[value]], or code+[[value]]).",
            "",
            prior_block if prior_block else "",
            "Question (may include format instructions):",
            question,
            "",
            "Context:",
            context,
            "",
            "Respond strictly in JSON.",
        ]
        return "\n".join(prompt_lines)

    def generate(
        self,
        question: str,
        context: str,
        prior_reflections: Optional[List[str]],
        call_id: str,
        log_dir: Optional[str],
        use_json_mode: bool,
    ) -> Tuple[str, Dict]:
        prompt = self._build_prompt(question, context, prior_reflections)
        return timed_llm_call(
            self.client,
            self.api_provider,
            self.model,
            prompt,
            role="reflexion_initial",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            temperature=self.temperature,
        )


class ReflexionReflect:
    """自我反思角色：输出自由文本改进建议（不要求 JSON）"""

    def __init__(self, client, api_provider: str, model: str, max_tokens: int, temperature: float):
        self.client = client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_prompt(self, question: str, context: str, previous_response: str) -> str:
        prompt_lines = [
            "You are reviewing your previous answer for a finance question.",
            "Provide concise, actionable feedback as free text (no JSON).",
            "Include: whether it seems correct, the issues, and how to fix them.",
            "",
            "Question (may include format instructions):",
            question,
            "",
            "Context:",
            context,
            "",
            "Previous attempt:",
            previous_response,
            "",
            "Write the reflection as plain text bullets or sentences. Do NOT return JSON.",
        ]
        return "\n".join(prompt_lines)

    def generate(
        self,
        question: str,
        context: str,
        previous_response: str,
        call_id: str,
        log_dir: Optional[str],
        use_json_mode: bool,
    ) -> Tuple[str, Dict]:
        prompt = self._build_prompt(question, context, previous_response)
        return timed_llm_call(
            self.client,
            self.api_provider,
            self.model,
            prompt,
            role="reflexion_reflect",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            temperature=self.temperature,
        )


class ReflexionRewrite:
    """根据反思重写答案"""

    def __init__(self, client, api_provider: str, model: str, max_tokens: int, temperature: float):
        self.client = client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_prompt(self, question: str, context: str, reflections: str, previous_response: str) -> str:
        prompt_lines = [
            "You are revising your previous answer using the provided reflections.",
            "Return a JSON object with ONLY keys `reasoning` and `final_answer`.",
            "- `reasoning`: chain of thought showing fixes (code can be placed here).",
            "- `final_answer`: strictly follow the required answer format (plain number, [[value]], or code+[[value]]).",
            "",
            "Question (may include format instructions):",
            question,
            "",
            "Context:",
            context,
            "",
            "Reflections to apply:",
            reflections,
            "",
            "Previous attempt:",
            previous_response,
            "",
            "Provide the improved JSON answer:",
        ]
        return "\n".join(prompt_lines)

    def generate(
        self,
        question: str,
        context: str,
        reflections: str,
        previous_response: str,
        call_id: str,
        log_dir: Optional[str],
        use_json_mode: bool,
    ) -> Tuple[str, Dict]:
        prompt = self._build_prompt(question, context, reflections, previous_response)
        return timed_llm_call(
            self.client,
            self.api_provider,
            self.model,
            prompt,
            role="reflexion_rewrite",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            temperature=self.temperature,
        )


class ReflexionGenerator:
    """
    Reflexion 生成器：初答 → 反思 → 重写循环，保持 BizBench 的 generator.generate 接口。
    """

    def __init__(
        self,
        api_provider: str,
        model_name: str,
        max_tokens: int,
        reflexion_rounds: int = 2,
        initial_temperature: float = 0.0,
        reflect_temperature: float = 0.2,
    ):
        client, _, _ = initialize_clients(api_provider)
        self.api_provider = api_provider
        self.model = model_name
        self.max_tokens = max_tokens
        self.reflexion_rounds = max(reflexion_rounds, 0)
        self.initial_temperature = initial_temperature
        self.reflect_temperature = reflect_temperature

        self.init_role = ReflexionInitial(
            client=client,
            api_provider=api_provider,
            model=model_name,
            max_tokens=max_tokens,
            temperature=initial_temperature,
        )
        self.reflect_role = ReflexionReflect(
            client=client,
            api_provider=api_provider,
            model=model_name,
            max_tokens=max_tokens,
            temperature=reflect_temperature,
        )
        self.rewrite_role = ReflexionRewrite(
            client=client,
            api_provider=api_provider,
            model=model_name,
            max_tokens=max_tokens,
            temperature=reflect_temperature,
        )

    @staticmethod
    def _normalize(text: Optional[str], fallback: str) -> str:
        text = (text or "").strip()
        return text if text else fallback

    @staticmethod
    def _parse_reflection(response: str) -> Tuple[str, bool]:
        """
        反思输出采用自由文本，返回 (reflection_text, is_correct_flag)。
        is_correct 使用简单关键词启发式判定。
        """
        text = (response or "").strip()
        is_correct = bool(re.search(r"\b(correct|looks\s+good|no\s+issues)\b", text, re.IGNORECASE))
        return text or "(no reflection generated)", is_correct

    def generate(
        self,
        question: str,
        playbook: str = "",  # unused; kept for interface compatibility
        context: str = "",
        reflection: str = "(empty)",
        prior_reflections: Optional[List[str]] = None,
        use_json_mode: bool = False,
        call_id: str = "reflexion",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], dict]:
        question_text = self._normalize(question, "(Question is empty.)")
        context_text = self._normalize(context, "(No additional context provided.)")
        _ = reflection  # reflection input not used; kept for signature parity

        init_response, init_info = self.init_role.generate(
            question=question_text,
            context=context_text,
            prior_reflections=prior_reflections,
            call_id=f"{call_id}_init",
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )

        best_response = init_response
        reflections: List[str] = []
        call_trace: List[Dict[str, Dict]] = [{"stage": "initial", "call_info": init_info}]

        for step in range(1, self.reflexion_rounds + 1):
            fb_response, fb_info = self.reflect_role.generate(
                question=question_text,
                context=context_text,
                previous_response=best_response,
                call_id=f"{call_id}_reflect{step}",
                log_dir=log_dir,
                use_json_mode=use_json_mode,
            )
            reflections_text, is_correct = self._parse_reflection(fb_response)
            reflections.append(reflections_text)
            call_trace.append({"stage": f"reflect_{step}", "call_info": fb_info})

            if is_correct:
                # Early stop: reflection认为已正确
                break

            rewrite_response, rewrite_info = self.rewrite_role.generate(
                question=question_text,
                context=context_text,
                reflections="\n".join(reflections),
                previous_response=best_response,
                call_id=f"{call_id}_rewrite{step}",
                log_dir=log_dir,
                use_json_mode=use_json_mode,
            )
            call_trace.append({"stage": f"rewrite_{step}", "call_info": rewrite_info})
            best_response = rewrite_response

        return best_response, [], {"trace": call_trace, "reflections": reflections}


