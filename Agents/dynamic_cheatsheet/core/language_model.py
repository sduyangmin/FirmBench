from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

from utils.tools import initialize_clients
from utils.llm import timed_llm_call
from .executor import extract_and_run_python_code
from .extractor import extract_cheatsheet


class DynamicCheatsheetLanguageModel:
    """
    Port of the original Dynamic Cheatsheet LanguageModel that runs on top of
    BizBench's client stack.
    """

    def __init__(
        self,
        api_provider: str,
        model_name: str,
        max_tokens: int,
        allow_code_execution: bool = True,
    ) -> None:
        self.generator_client, _, _ = initialize_clients(api_provider)
        self.api_provider = api_provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.allow_code_execution = allow_code_execution
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def count_tokens(self, text: str) -> int:
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def _call_model(
        self,
        history: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        call_id: str,
        log_dir: Optional[str],
    ) -> str:
        prompt_preview = history[-1]["content"]
        response, _ = timed_llm_call(
            client=self.generator_client,
            api_provider=self.api_provider,
            model=self.model_name,
            prompt=prompt_preview,
            role="dynamic_cheatsheet",
            call_id=call_id,
            max_tokens=max_tokens,
            log_dir=log_dir,
            use_json_mode=False,
            temperature=temperature,
            messages=history,
        )
        return response

    def generate(
        self,
        history: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        current_depth: int = 1,
        max_depth_num_rounds: int = 3,
        allow_code_execution: Optional[bool] = None,
        code_execution_flag: str = "EXECUTE CODE!",
        final_output: str = "",
        call_id: str = "dynamic_cheatsheet",
        log_dir: Optional[str] = None,
    ) -> str:
        if len(history) == 0:
            raise ValueError("History must contain at least one message.")

        output = self._call_model(
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            call_id=call_id,
            log_dir=log_dir,
        )

        allow_execution = (
            self.allow_code_execution if allow_code_execution is None else allow_code_execution
        )

        pre_code_execution_flag = output.split(code_execution_flag)[0].strip()
        if (
            allow_execution
            and code_execution_flag in output
            and len(pre_code_execution_flag) >= 3
            and pre_code_execution_flag.endswith("```")
        ):
            output_prefix = output.split(code_execution_flag)[0].strip()
            executed_code = extract_and_run_python_code(output_prefix) or ""
            executed_code = executed_code.strip()
            current_output = f"{output_prefix}\n{code_execution_flag}\n\n{executed_code}"
            final_output = f"{final_output}\n\n{current_output}".strip()

            if current_depth <= max_depth_num_rounds:
                warning_txt = ""
                if current_depth == max_depth_num_rounds:
                    warning_txt = (
                        " (This is the last round. No more code execution will be allowed. "
                        "Please present your final solution now.)"
                    )
                new_messages = [
                    {"role": "assistant", "content": current_output},
                    {
                        "role": "user",
                        "content": (
                            "Proceed with any additional steps required and provide the "
                            "completed solution. If everything is already complete, type "
                            "FINAL ANSWER and submit it in the expected format. If you are "
                            f"stuck, please try alternative methods to solve the problem and "
                            f"provide the final solution.{warning_txt}"
                        ),
                    },
                ]
                history = history + new_messages
                return self.generate(
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    current_depth=current_depth + 1,
                    max_depth_num_rounds=max_depth_num_rounds,
                    allow_code_execution=allow_execution,
                    code_execution_flag=code_execution_flag,
                    final_output=final_output,
                    call_id=f"{call_id}_depth_{current_depth+1}",
                    log_dir=log_dir,
                )

            final_output = f"{final_output}\n\n{current_output}".strip()
            return final_output

        final_output = f"{final_output}\n\n{output}".strip()
        return final_output

    def advanced_generate(
        self,
        approach_name: str,
        input_txt: str,
        cheatsheet: Optional[str] = None,
        generator_template: Optional[str] = None,
        cheatsheet_template: Optional[str] = None,
        cheatsheet_question: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_num_rounds: int = 1,
        allow_code_execution: Optional[bool] = None,
        code_execution_flag: str = "EXECUTE CODE!",
        add_previous_answers_to_cheatsheet: bool = True,
        original_input_corpus: Optional[List[str]] = None,
        original_input_embeddings: Optional[np.ndarray] = None,
        generator_outputs_so_far: Optional[List[str]] = None,
        retrieve_top_k: int = 3,
        log_dir: Optional[str] = None,
        call_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        High-level orchestration wrapper for all supported Dynamic Cheatsheet
        variants. Returns a dict mirroring the original implementation.
        """

        call_prefix = call_prefix or f"dc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        if approach_name == "default":
            generator_prompt = (
                generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", "(empty)")
            )
            history = [{"role": "user", "content": generator_prompt}]
            generator_output = self.generate(
                history=history,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_code_execution=allow_code_execution,
                code_execution_flag=code_execution_flag,
                call_id=f"{call_prefix}_default",
                log_dir=log_dir,
            )
            generator_answer = generator_output
            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": None,
                        "new_cheatsheet": None,
                    }
                ],
                "previous_answers": None,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": None,
                "generator_output": generator_output,
            }

        if approach_name == "DynamicCheatsheet_Cumulative":
            if cheatsheet is None or cheatsheet_template is None:
                raise ValueError("Cheatsheet and template must be provided for cumulative mode.")

            steps = []
            previous_answers = []
            new_cheatsheet = cheatsheet
            q_for_cheatsheet = cheatsheet_question or input_txt

            for round_idx in range(max(1, max_num_rounds)):
                generator_cheatsheet_content = new_cheatsheet
                if round_idx > 0 and add_previous_answers_to_cheatsheet and previous_answers:
                    previous_answers_txt = f"PREVIOUS ANSWERS:\n{'; '.join(previous_answers)}"
                    generator_cheatsheet_content = f"{generator_cheatsheet_content}\n\n{previous_answers_txt}"

                generator_prompt = (
                    generator_template.replace("[[QUESTION]]", input_txt).replace(
                        "[[CHEATSHEET]]", generator_cheatsheet_content
                    )
                )
                current_cheatsheet = new_cheatsheet
                history = [{"role": "user", "content": generator_prompt}]
                generator_output = self.generate(
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                    call_id=f"{call_prefix}_generator_round_{round_idx}",
                    log_dir=log_dir,
                )
                generator_answer = generator_output

                cheatsheet_prompt = (
                    cheatsheet_template.replace("[[QUESTION]]", q_for_cheatsheet)
                    .replace("[[MODEL_ANSWER]]", generator_output)
                    .replace("[[PREVIOUS_CHEATSHEET]]", current_cheatsheet)
                )
                cheatsheet_history = [{"role": "user", "content": cheatsheet_prompt}]
                cheatsheet_output = self.generate(
                    history=cheatsheet_history,
                    temperature=temperature,
                    max_tokens=2 * max_tokens,
                    allow_code_execution=False,
                    call_id=f"{call_prefix}_curator_round_{round_idx}",
                    log_dir=log_dir,
                )
                new_cheatsheet = extract_cheatsheet(cheatsheet_output, current_cheatsheet)
                previous_answers.append(f"Round {round_idx + 1}: {generator_answer}")

                steps.append(
                    {
                        "round": round_idx,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": current_cheatsheet,
                        "new_cheatsheet": new_cheatsheet,
                    }
                )

            return {
                "input_txt": input_txt,
                "steps": steps,
                "previous_answers": previous_answers,
                "final_answer": generator_answer,
                "final_cheatsheet": new_cheatsheet,
                "final_output": generator_output,
            }

        if approach_name == "FullHistoryAppending":
            if generator_outputs_so_far is None:
                generator_outputs_so_far = []
            length_of_history = len(generator_outputs_so_far)
            if length_of_history > 0 and original_input_corpus:
                curated_cheatsheet = "### PREVIOUS SOLUTIONS (START)\n\n"
                for i, (prev_input, prev_output) in enumerate(
                    zip(original_input_corpus[:length_of_history], generator_outputs_so_far)
                ):
                    curated_cheatsheet += (
                        f"#### Previous Input #{i+1}:\n\n{prev_input}\n\n"
                        f"#### Model Solution to Previous Input #{i+1}:\n\n{prev_output}\n---\n---\n\n"
                    )
                curated_cheatsheet += "#### PREVIOUS SOLUTIONS (END)"
            else:
                curated_cheatsheet = "(empty)"

            generator_prompt = (
                generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", curated_cheatsheet)
            )
            history = [{"role": "user", "content": generator_prompt}]
            generator_output = self.generate(
                history=history,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_code_execution=allow_code_execution,
                code_execution_flag=code_execution_flag,
                call_id=f"{call_prefix}_full_history",
                log_dir=log_dir,
            )
            generator_answer = generator_output

            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": curated_cheatsheet,
                        "new_cheatsheet": None,
                    }
                ],
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": curated_cheatsheet,
            }

        if approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis"]:
            if original_input_embeddings is None or original_input_corpus is None:
                raise ValueError("Retrieval approaches require input corpus and embeddings.")
            current_embedding = original_input_embeddings[-1]
            prev_embeddings = original_input_embeddings[:-1]

            if len(prev_embeddings) > 0:
                similarities = cosine_similarity([current_embedding], prev_embeddings)
                top_k_indices = np.argsort(similarities[0])[::-1][:retrieve_top_k]
                top_k_original_inputs = [original_input_corpus[i] for i in top_k_indices]
                top_k_original_outputs = [generator_outputs_so_far[i] for i in top_k_indices] if generator_outputs_so_far else []
                top_k_similar_values = similarities[0][top_k_indices]
                curated_cheatsheet = (
                    "### PREVIOUS SOLUTIONS (START)\n\n"
                    "Note: The input-output pairs listed below are taken from previous test "
                    "cases and are meant to assist you in understanding potential solution "
                    "strategies or tool usages...\n\n"
                )
                for i, (previous_input_txt, previous_output_txt, similarity) in enumerate(
                    zip(top_k_original_inputs[::-1], top_k_original_outputs[::-1], top_k_similar_values[::-1])
                ):
                    curated_cheatsheet += (
                        f"#### Previous Input #{i+1} (Similarity: {similarity:.2f}):\n\n"
                        f"{previous_input_txt}\n\n#### Model Solution to Previous Input  #{i+1}:\n\n"
                        f"{previous_output_txt}\n---\n---\n\n"
                    )
                curated_cheatsheet = curated_cheatsheet.strip()
                if curated_cheatsheet != "(empty)":
                    curated_cheatsheet += "\n\n#### PREVIOUS SOLUTIONS (END)"
            else:
                curated_cheatsheet = "(empty)"

            previous_cheatsheet = cheatsheet or "(empty)"
            if approach_name == "DynamicCheatsheet_RetrievalSynthesis" and cheatsheet_template:
                cheatsheet_prompt = (
                    cheatsheet_template.replace("[[PREVIOUS_INPUT_OUTPUT_PAIRS]]", curated_cheatsheet)
                    .replace("[[NEXT_INPUT]]", input_txt)
                    .replace("[[PREVIOUS_CHEATSHEET]]", previous_cheatsheet)
                )
                cheatsheet_history = [{"role": "user", "content": cheatsheet_prompt}]
                cheatsheet_output = self.generate(
                    history=cheatsheet_history,
                    temperature=temperature,
                    max_tokens=2 * max_tokens,
                    allow_code_execution=False,
                    call_id=f"{call_prefix}_retrieval_curator",
                    log_dir=log_dir,
                )
                curated_cheatsheet = extract_cheatsheet(cheatsheet_output, curated_cheatsheet)

            generator_prompt = (
                generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", curated_cheatsheet)
            )
            history = [{"role": "user", "content": generator_prompt}]
            generator_output = self.generate(
                history=history,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_code_execution=allow_code_execution,
                code_execution_flag=code_execution_flag,
                call_id=f"{call_prefix}_retrieval_generator",
                log_dir=log_dir,
            )
            generator_answer = generator_output

            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": curated_cheatsheet,
                        "new_cheatsheet": None,
                    }
                ],
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": curated_cheatsheet,
            }

        raise ValueError(f"Approach '{approach_name}' not found.")








