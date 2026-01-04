from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.tools import initialize_clients
from utils.llm import timed_llm_call


@dataclass
class DiscussionConfig:
    num_experts: int = 3
    rounds: int = 2
    expert_temperature: float = 0.2
    moderator_temperature: float = 0.0


class DiscussionGenerator:
    """
    Discussion-style generator:
    - Multiple experts independently propose solutions
    - (Optional) additional rounds where experts refine based on others
    - Moderator synthesizes and outputs final JSON {reasoning, final_answer}
    """

    def __init__(
        self,
        api_provider: str,
        model_name: str,
        max_tokens: int,
        discussion_config: Optional[DiscussionConfig] = None,
    ):
        client, _, _ = initialize_clients(api_provider)
        self.client = client
        self.api_provider = api_provider
        self.model = model_name
        self.max_tokens = max_tokens
        self.cfg = discussion_config or DiscussionConfig()

    @staticmethod
    def _normalize(text: Optional[str], fallback: str) -> str:
        text = (text or "").strip()
        return text if text else fallback

    def _prompt_expert(self, expert_id: int, question: str, context: str, prior: Optional[str]) -> str:
        prior_block = ""
        if prior:
            prior_block = "\n".join(
                [
                    "Here are other experts' proposals (for reference only; do not copy blindly):",
                    prior,
                    "",
                ]
            )
        return "\n".join(
            [
                f"You are Expert #{expert_id} in a collaborative discussion.",
                "Your goal: propose a correct solution and final answer.",
                "Follow any answer-format instructions inside the question.",
                "",
                "Return a JSON object with ONLY keys:",
                "- `reasoning`: your derivation / steps / checks.",
                "- `final_answer`: your proposed final answer (strict format).",
                "",
                prior_block,
                "Question:",
                question,
                "",
                "Context:",
                context,
                "",
                "Respond strictly in JSON.",
            ]
        )

    def _prompt_moderator(self, question: str, context: str, proposals: str) -> str:
        return "\n".join(
            [
                "You are the MODERATOR of a panel discussion.",
                "You have multiple experts' proposed answers. Your job is to reconcile them and output the best final answer.",
                "Follow any answer-format instructions inside the question.",
                "",
                "Return a JSON object with ONLY keys:",
                "- `reasoning`: the synthesized reasoning and any crucial verification.",
                "- `final_answer`: the definitive final answer (strict format).",
                "",
                "Question:",
                question,
                "",
                "Context:",
                context,
                "",
                "Experts' proposals:",
                proposals,
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
        call_id: str = "discussion",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], dict]:
        _ = (playbook, reflection)
        q = self._normalize(question, "(Question is empty.)")
        c = self._normalize(context, "(No additional context provided.)")

        num_experts = max(int(self.cfg.num_experts), 1)
        rounds = max(int(self.cfg.rounds), 1)

        call_trace: List[Dict[str, Dict]] = []
        expert_outputs: List[str] = []

        prior_pool: Optional[str] = None
        for r in range(1, rounds + 1):
            round_outputs: List[str] = []
            for e in range(1, num_experts + 1):
                prompt = self._prompt_expert(e, q, c, prior_pool)
                resp, info = timed_llm_call(
                    self.client,
                    self.api_provider,
                    self.model,
                    prompt,
                    role="discussion_expert",
                    call_id=f"{call_id}_r{r}_e{e}",
                    max_tokens=self.max_tokens,
                    log_dir=log_dir,
                    use_json_mode=use_json_mode,
                    temperature=self.cfg.expert_temperature,
                )
                round_outputs.append(f"[ROUND {r}] Expert {e}:\n{resp}")
                call_trace.append({"stage": f"r{r}_expert{e}", "call_info": info})

            expert_outputs = round_outputs  # keep only latest round for synthesis
            prior_pool = "\n\n".join(round_outputs)

        proposals_text = "\n\n".join(expert_outputs)
        moderator_prompt = self._prompt_moderator(q, c, proposals_text)
        final_response, mod_info = timed_llm_call(
            self.client,
            self.api_provider,
            self.model,
            moderator_prompt,
            role="discussion_moderator",
            call_id=f"{call_id}_moderator",
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            temperature=self.cfg.moderator_temperature,
        )
        call_trace.append({"stage": "moderator", "call_info": mod_info})

        meta = {
            "trace": call_trace,
            "rounds": rounds,
            "num_experts": num_experts,
            "last_round_proposals": expert_outputs,
        }
        return final_response, [], meta


