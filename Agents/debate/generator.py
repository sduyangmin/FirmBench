from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.tools import initialize_clients
from utils.llm import timed_llm_call


@dataclass
class DebateConfig:
    rounds: int = 3
    pro_temperature: float = 0.0
    con_temperature: float = 0.2
    judge_temperature: float = 0.0


class DebateGenerator:
    """
    Debate-style generator:
    - Proponent proposes solution/answer
    - Opponent critiques and proposes alternative
    - Judge synthesizes and outputs final JSON {reasoning, final_answer}
    """

    def __init__(
        self,
        api_provider: str,
        model_name: str,
        max_tokens: int,
        debate_config: Optional[DebateConfig] = None,
    ):
        client, _, _ = initialize_clients(api_provider)
        self.client = client
        self.api_provider = api_provider
        self.model = model_name
        self.max_tokens = max_tokens
        self.cfg = debate_config or DebateConfig()

    @staticmethod
    def _normalize(text: Optional[str], fallback: str) -> str:
        text = (text or "").strip()
        return text if text else fallback

    def _prompt_pro(self, question: str, context: str) -> str:
        return "\n".join(
            [
                "You are the PRO side in a structured debate.",
                "Your job: solve the task carefully and propose the best possible answer.",
                "Follow any answer-format instructions inside the question.",
                "",
                "Return a JSON object with ONLY keys:",
                "- `reasoning`: your derivation / steps / calculations.",
                "- `final_answer`: the proposed final answer (strict format).",
                "",
                "Question:",
                question,
                "",
                "Context:",
                context,
                "",
                "Respond strictly in JSON.",
            ]
        )

    def _prompt_con(self, question: str, context: str, pro_answer: str) -> str:
        return "\n".join(
            [
                "You are the CON side in a structured debate.",
                "Your job: attempt to falsify the PRO answer by finding mistakes or edge cases.",
                "If PRO is correct, you should acknowledge it and strengthen the justification.",
                "Follow any answer-format instructions inside the question.",
                "",
                "Return a JSON object with ONLY keys:",
                "- `critique`: the strongest critique / verification notes.",
                "- `alternative_reasoning`: your independent solution attempt.",
                "- `final_answer`: your best proposed final answer (may match PRO).",
                "",
                "Question:",
                question,
                "",
                "Context:",
                context,
                "",
                "PRO answer to critique:",
                pro_answer,
                "",
                "Respond strictly in JSON.",
            ]
        )

    def _prompt_judge(self, question: str, context: str, debate_transcript: str) -> str:
        return "\n".join(
            [
                "You are the JUDGE of a structured debate between PRO and CON.",
                "Your job: decide the best final answer. Do not blindly trust either side.",
                "Follow any answer-format instructions inside the question.",
                "",
                "Return a JSON object with ONLY keys:",
                "- `reasoning`: explain the decision, key checks, and final derivation.",
                "- `final_answer`: the definitive final answer (strict format).",
                "",
                "Question:",
                question,
                "",
                "Context:",
                context,
                "",
                "Debate transcript (PRO/CON outputs):",
                debate_transcript,
                "",
                "Respond strictly in JSON.",
            ]
        )

    def generate(
        self,
        question: str,
        playbook: str = "",  # unused; kept for interface compatibility
        context: str = "",
        reflection: str = "(empty)",  # unused; kept for interface compatibility
        use_json_mode: bool = False,
        call_id: str = "debate",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], dict]:
        _ = (playbook, reflection)
        q = self._normalize(question, "(Question is empty.)")
        c = self._normalize(context, "(No additional context provided.)")

        rounds = max(int(self.cfg.rounds), 1)
        transcript_parts: List[str] = []
        call_trace: List[Dict[str, Dict]] = []

        pro_answer = ""
        con_answer = ""

        for r in range(1, rounds + 1):
            pro_prompt = self._prompt_pro(q, c)
            pro_answer, pro_info = timed_llm_call(
                self.client,
                self.api_provider,
                self.model,
                pro_prompt,
                role="debate_pro",
                call_id=f"{call_id}_pro_r{r}",
                max_tokens=self.max_tokens,
                log_dir=log_dir,
                use_json_mode=use_json_mode,
                temperature=self.cfg.pro_temperature,
            )
            transcript_parts.append(f"[ROUND {r}] PRO:\n{pro_answer}")
            call_trace.append({"stage": f"pro_r{r}", "call_info": pro_info})

            con_prompt = self._prompt_con(q, c, pro_answer)
            con_answer, con_info = timed_llm_call(
                self.client,
                self.api_provider,
                self.model,
                con_prompt,
                role="debate_con",
                call_id=f"{call_id}_con_r{r}",
                max_tokens=self.max_tokens,
                log_dir=log_dir,
                use_json_mode=use_json_mode,
                temperature=self.cfg.con_temperature,
            )
            transcript_parts.append(f"[ROUND {r}] CON:\n{con_answer}")
            call_trace.append({"stage": f"con_r{r}", "call_info": con_info})

        transcript = "\n\n".join(transcript_parts)
        judge_prompt = self._prompt_judge(q, c, transcript)
        final_response, judge_info = timed_llm_call(
            self.client,
            self.api_provider,
            self.model,
            judge_prompt,
            role="debate_judge",
            call_id=f"{call_id}_judge",
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            temperature=self.cfg.judge_temperature,
        )
        call_trace.append({"stage": "judge", "call_info": judge_info})

        meta = {
            "trace": call_trace,
            "rounds": rounds,
            "pro_last": pro_answer,
            "con_last": con_answer,
        }
        return final_response, [], meta


