from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.consulting_tools import evaluate_consulting_set
from utils.seriousgame_tools import (
    beergame_prepare_run,
    beergame_evaluate_run,
    beergame_save_run,
    beergame_build_query,
    beergame_base_rule_order,
    beergame_render_prompt,
    beergame_extract_order_and_note,
)
from .core.language_model import DynamicCheatsheetLanguageModel
from .core.state import CheatsheetState

from utils.seriousgame_tools import (
    build_edt_decision_context,
    render_edt_prompt,
    normalize_edt_schema,
    edt_prepare_run,
    edt_evaluate_run,
    edt_save_run,
)

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _read_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


@dataclass
class DynamicCheatsheetConfig:
    approach_name: str = "DynamicCheatsheet_Cumulative"
    generator_prompt_path: Path = DEFAULT_PROMPTS_DIR / "generator_prompt.txt"
    cheatsheet_prompt_path: Optional[Path] = DEFAULT_PROMPTS_DIR / "cheatsheet_cumulative.txt"
    temperature: float = 0.0
    max_num_rounds: int = 1
    retrieve_top_k: int = 3
    allow_code_execution: bool = True
    add_previous_answers: bool = True


class DynamicCheatsheetAgent:
    """
    BizBench-compatible Agent implementation of Dynamic Cheatsheet.
    """

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        agent_method: str = "dynamic_cheatsheet",
        dc_config: Optional[DynamicCheatsheetConfig] = None,
    ):
        self.agent_method = agent_method
        self.max_tokens = max_tokens
        self.dc_config = dc_config or DynamicCheatsheetConfig()

        self.language_model = DynamicCheatsheetLanguageModel(
            api_provider=api_provider,
            model_name=generator_model,
            max_tokens=max_tokens,
            allow_code_execution=self.dc_config.allow_code_execution,
        )
        self.generator_client = self.language_model.generator_client
        self.model_name = self.language_model.model_name
        self.temperature = 0.7

        self.generator_prompt = _read_file(self.dc_config.generator_prompt_path)
        self.cheatsheet_prompt = (
            _read_file(self.dc_config.cheatsheet_prompt_path)
            if self.dc_config.cheatsheet_prompt_path
            else "(empty)"
        )
        self.retrieval_corpus: List[str] = []
        self.retrieval_embeddings: List[List[float]] = []
        self.retrieval_index: Dict[str, int] = {}
        self.retrieval_outputs: Dict[int, str] = {}
        # BeerGame cheatsheet persistence
        self.beergame_cheatsheet_path: Optional[str] = None
        self.beergame_cheatsheet_text: str = ""

        # ===== EDT runtime state (Mode A) =====
        self._edt_state: Optional[CheatsheetState] = None
        self._edt_paths: Optional[Dict[str, str]] = None
        self._edt_window_size: int = 1
        self._edt_pending: List[Dict[str, Any]] = []
        self._edt_history_path: Optional[str] = None

    # ==========================================================
    # Shared helpers
    # ==========================================================
    def _prepare_dirs(self, task_name: str, mode: str, save_dir: str) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name, self.agent_method, mode, timestamp)
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)
        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        cheatsheet_history_path = os.path.join(resolved_save_path, "cheatsheet_history.jsonl")

        return {
            "run_subdir": run_subdir,
            "resolved_save_path": resolved_save_path,
            "log_dir": log_dir,
            "cheatsheet_history_path": cheatsheet_history_path,
        }

    def _load_initial_cheatsheet(self, path: Optional[str]) -> str:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "(empty)"

    def _format_input(self, sample: Dict[str, Any], index: int, question_override: Optional[str] = None) -> str:
        raw_question = question_override if question_override is not None else sample.get("question", "")
        raw_context = sample.get("context", "")
        question = (raw_question or "").strip()
        context = (raw_context or "").strip()
        parts = [f"Question #{index + 1}:\n{question}"]
        if context:
            parts.append(f"\nContext:\n{context}")
        return "\n".join(parts).strip()

    def _select_prompts_for_task(self, task_name: str, base_generator_prompt: str, base_cheatsheet_prompt: str) -> Tuple[str, str]:
        """
        针对特定任务做 prompt 特化：
        - FormulaEval：只输出当前缺失的函数体，单一 python 代码块，无 JSON/解释/EXECUTE。
        """
        if task_name.lower() == "formulaeval":
            generator_prompt = (
                "You are a code completion assistant. The function/method signature is provided above.\n"
                "Fill in ONLY the missing body lines with correct indentation.\n"
                "Do NOT repeat the class definition or the signature.\n"
                "Do NOT return JSON or analysis; only return one ```python``` code block containing the body lines.\n"
                "If the signature is present, just give the indented body lines; if only the body is missing, fill it directly.\n"
                "Ensure the code is executable and self-contained (no external files/paths/imports beyond stdlib if needed).\n\n"
                "Question:\n[[QUESTION]]"
            )
            cheatsheet_prompt = base_cheatsheet_prompt  # 保持原有检索小抄，但输出格式受 generator_prompt 约束
            return generator_prompt, cheatsheet_prompt

        return base_generator_prompt, base_cheatsheet_prompt

    def _append_history(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _snapshot_state(self, state: CheatsheetState) -> CheatsheetState:
        """
        创建 cheatsheet 状态的浅拷贝，用于并行只读评估。
        """
        return CheatsheetState(
            cheatsheet_text=state.cheatsheet_text,
            previous_answers=list(state.previous_answers),
            input_history=list(state.input_history),
            output_history=list(state.output_history),
            embeddings=list(state.embeddings) if state.embeddings else None,
        )

    def _process_sample(
        self,
        sample: Dict[str, Any],
        idx: int,
        state: CheatsheetState,
        data_processor,
        paths: Dict[str, str],
        strip_task_instruction_fn,
        update_state: bool = True,
        log_history: bool = True,
        verbose: bool = False,
        call_prefix: Optional[str] = None,
        retrieval_corpus: Optional[List[str]] = None,
        retrieval_embeddings: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        统一的单样本处理逻辑，可用于训练（更新 state）或只读评估。
        """
        input_txt = self._format_input(sample, idx)
        stripped_question = strip_task_instruction_fn(sample.get("question", ""))
        input_txt_for_cheatsheet = self._format_input(sample, idx, question_override=stripped_question)
        target = sample.get("target")

        original_input_corpus = state.input_history + [input_txt]
        original_input_embeddings = None
        if retrieval_corpus and retrieval_embeddings:
            idx_in_retrieval = self.retrieval_index.get(input_txt)
            if idx_in_retrieval is None or idx_in_retrieval >= len(retrieval_embeddings):
                idx_in_retrieval = len(retrieval_embeddings) - 1
            permutation = [i for i in range(len(retrieval_corpus)) if i != idx_in_retrieval] + [idx_in_retrieval]
            prev_inputs = [retrieval_corpus[i] for i in permutation]
            prev_embeddings = [retrieval_embeddings[i] for i in permutation]
            original_input_corpus = prev_inputs
            original_input_embeddings = prev_embeddings
            generator_outputs_so_far = [self.retrieval_outputs.get(i, "") for i in permutation]
        else:
            generator_outputs_so_far = state.output_history

        output_dict = self.language_model.advanced_generate(
            approach_name=self.dc_config.approach_name,
            input_txt=input_txt,
            cheatsheet=state.cheatsheet_text,
            generator_template=self.generator_prompt_for_run,
            cheatsheet_template=self.cheatsheet_prompt_for_run,
            cheatsheet_question=input_txt_for_cheatsheet,
            temperature=self.dc_config.temperature,
            max_tokens=self.max_tokens,
            max_num_rounds=self.dc_config.max_num_rounds,
            allow_code_execution=self.dc_config.allow_code_execution,
            code_execution_flag="EXECUTE CODE!",
            add_previous_answers_to_cheatsheet=self.dc_config.add_previous_answers,
            original_input_corpus=original_input_corpus,
            original_input_embeddings=original_input_embeddings,
            generator_outputs_so_far=generator_outputs_so_far,
            retrieve_top_k=self.dc_config.retrieve_top_k,
            log_dir=paths["log_dir"],
            call_prefix=call_prefix or f"sample_{idx}",
        )

        final_answer = output_dict.get("final_answer")
        final_cheatsheet = output_dict.get("final_cheatsheet") or state.cheatsheet_text
        is_correct = data_processor.answer_is_correct(final_answer, target)

        if update_state:
            state.update_cheatsheet(final_cheatsheet)
            state.append_example(
                input_txt=input_txt,
                generator_output=output_dict.get("final_output", ""),
                generator_answer=final_answer or "",
            )
            if retrieval_corpus and retrieval_embeddings:
                idx_in_retrieval = self.retrieval_index.get(input_txt)
                if idx_in_retrieval is not None:
                    self.retrieval_outputs[idx_in_retrieval] = final_answer or ""

        if log_history and update_state:
            history_payload = {
                "index": idx,
                "input": input_txt,
                "steps": output_dict.get("steps"),
                "final_cheatsheet": final_cheatsheet,
            }
            self._append_history(paths["cheatsheet_history_path"], history_payload)

        if verbose:
            print(f"[Sample {idx + 1}] Correct: {is_correct} | Final Answer: {final_answer}")

        return {
            "index": idx,
            "input": input_txt,
            "target": target,
            "final_answer": final_answer,
            "final_output": output_dict.get("final_output"),
            "cheatsheet": final_cheatsheet,
            "is_correct": is_correct,
        }

    def _load_embedding_csv(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")

        corpus: List[str] = []
        embeddings: List[List[float]] = []
        index_map: Dict[str, int] = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = row.get("input", "")
                emb_json = row.get("embedding_json") or row.get("embedding")
                if not text or not emb_json:
                    continue
                try:
                    emb = json.loads(emb_json)
                except json.JSONDecodeError:
                    continue
                corpus.append(text)
                embeddings.append(emb)
                index_map[text] = i

        if not corpus or not embeddings:
            raise ValueError(f"No valid embeddings found in {path}")

        self.retrieval_corpus = corpus
        self.retrieval_embeddings = embeddings
        self.retrieval_index = index_map

    # ==========================================================
    # Consulting support
    # ==========================================================

    def _call_llm_json(self, system: str, user: str) -> Dict[str, Any]:
        import json as _json

        resp = self.generator_client.chat.completions.create(
            model=self.model_name,
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
        Consulting candidate reply using dynamic cheatsheet with persistence:
        - retrieves current cheatsheet (per run)
        - runs cumulative generation
        - updates cheatsheet for subsequent cases.
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

        cheatsheet_text = getattr(self, "_consulting_cheatsheet", "(empty)")

        input_txt = (
            f"[Consulting case: {case_id}] Latest interviewer question:\n"
            f"{last_interviewer_msg or '(Interviewer message missing.)'}\n\n"
            f"Dialogue so far:\n{transcript_text}"
        )

        try:
            output = self.language_model.advanced_generate(
                approach_name="DynamicCheatsheet_Cumulative",
                input_txt=input_txt,
                cheatsheet=cheatsheet_text,
                generator_template=self.generator_prompt,
                cheatsheet_template=self.cheatsheet_prompt,
                temperature=self.dc_config.temperature,
                max_tokens=self.max_tokens,
                max_num_rounds=self.dc_config.max_num_rounds,
                allow_code_execution=self.dc_config.allow_code_execution,
                code_execution_flag="EXECUTE CODE!",
                add_previous_answers_to_cheatsheet=self.dc_config.add_previous_answers,
                retrieve_top_k=self.dc_config.retrieve_top_k,
                log_dir=None,
                call_prefix=f"consult_dc_{case_id}_t{turns}",
            )
            reply = output.get("final_answer") or output.get("final_output") or ""
            new_cheatsheet = output.get("final_cheatsheet") or cheatsheet_text
        except Exception:
            reply = ""
            new_cheatsheet = cheatsheet_text

        # 更新并持久化 cheatsheet
        try:
            self._consulting_cheatsheet = new_cheatsheet
            path = getattr(self, "_consulting_cheatsheet_path", None)
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_cheatsheet)
        except Exception:
            pass

        if not isinstance(reply, str) or not reply.strip():
            reply = (
                "Let me structure the issues, share a hypothesis, and outline the "
                "first analyses I'd run to validate it."
            )
        return reply.strip()

    # ==========================================================
    # BeerGame: decision hook + evaluation entry (cheatsheet persistence)
    # ==========================================================

    def _load_beergame_cheatsheet(self, path: str) -> None:
        """加载/初始化 BeerGame cheatsheet（跨 episode 持久化）。"""
        self.beergame_cheatsheet_path = path
        try:
            # 优先加载“当前 cheatsheet”文件；否则退回读取追加日志文件（可能很长）
            current_path = path + ".current"
            if os.path.exists(current_path):
                with open(current_path, "r", encoding="utf-8") as f:
                    self.beergame_cheatsheet_text = f.read().strip()
            elif os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.beergame_cheatsheet_text = f.read().strip()
            else:
                self.beergame_cheatsheet_text = ""
        except Exception:
            self.beergame_cheatsheet_text = ""
    def _decide_order_qty(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """
        BeerGame 决策：使用 DynamicCheatsheet（cumulative cheatsheet + generator/curator），
        读取/写入持久化 cheatsheet，并强制 JSON-only 输出。
        """
        role = str(ctx.get("role", obs.get("role", "retailer")))
        week = int(obs.get("week", 0) or 0)
        max_order_qty = int(getattr(self, "max_order_qty", 5000))

        base_order = beergame_base_rule_order(
            obs=obs,
            ctx=ctx,
            max_order_qty=max_order_qty,
        )

        system, user = beergame_render_prompt(
            role=role,
            obs=obs,
            retrieved=self.beergame_cheatsheet_text,
            base_order=base_order,
        )

        question_text = (
            f"{system}\n\n{user}\n\n"
            "You must respond strictly in JSON ONLY, exactly in the form:\n"
            "{\n"
            '  "order_qty": <integer>,\n'
            '  "note": "<brief rationale>"\n'
            "}\n"
            "No other keys. No text before or after JSON. Do NOT reveal chain-of-thought."
        )

        context_text = (
            "Cheatsheet (persistent across BeerGame runs):\n"
            f"{self.beergame_cheatsheet_text or '(empty)'}"
        )

        # 用 DynamicCheatsheetLanguageModel.advanced_generate 生成，并更新 cheatsheet
        # 注意：该实现内部不支持 json_mode 参数，因此通过 prompt 约束 JSON-only。
        generator_template = (
            "You are a BeerGame ordering agent.\n"
            "You MUST output strictly valid JSON ONLY, exactly in the form:\n"
            "{\n"
            '  \"order_qty\": <integer>,\n'
            '  \"note\": \"<brief rationale>\"\n'
            "}\n"
            "No other keys. No text before or after JSON.\n\n"
            "Cheatsheet (learned heuristics):\n[[CHEATSHEET]]\n\n"
            "Task:\n[[QUESTION]]\n"
        )
        cheatsheet_template = (
            "You are updating a BeerGame decision cheatsheet.\n"
            "Given the latest task and model answer, update the cheatsheet with concise reusable heuristics.\n"
            "Return ONLY:\n<cheatsheet>\n...updated cheatsheet...\n</cheatsheet>\n\n"
            "Task:\n[[QUESTION]]\n\n"
            "Model answer:\n[[MODEL_ANSWER]]\n\n"
            "Previous cheatsheet:\n[[PREVIOUS_CHEATSHEET]]\n"
        )

        output = self.language_model.advanced_generate(
            approach_name="DynamicCheatsheet_Cumulative",
            input_txt=question_text,
            cheatsheet=self.beergame_cheatsheet_text or "(empty)",
            generator_template=generator_template,
            cheatsheet_template=cheatsheet_template,
            cheatsheet_question=question_text,
            temperature=self.dc_config.temperature,
            max_tokens=min(self.max_tokens, 512),
            max_num_rounds=max(1, int(self.dc_config.max_num_rounds)),
            allow_code_execution=False,  # BeerGame 不需要执行代码
            code_execution_flag="EXECUTE CODE!",
            add_previous_answers_to_cheatsheet=True,
            retrieve_top_k=max(1, int(self.dc_config.retrieve_top_k)),
            log_dir=None,
            call_prefix=f"beergame_dyncht_{ctx.get('scenario_id','')}_{ctx.get('episode_id','')}_w{week}",
        )

        response_text = str(output.get("final_answer") or output.get("final_output") or "").strip()
        new_cheatsheet = str(output.get("final_cheatsheet") or self.beergame_cheatsheet_text or "").strip()
        if new_cheatsheet and new_cheatsheet != self.beergame_cheatsheet_text:
            self.beergame_cheatsheet_text = new_cheatsheet

        order_qty = self._extract_order_qty_from_dyncht(
            response_text=response_text,
            base_order=base_order,
            max_order_qty=max_order_qty,
        )

        # 持久化记录到 cheatsheet（追加决策 + 覆写最新 cheatsheet）
        try:
            entry = (
                f"week={week} role={role} order={order_qty} base={base_order} "
                f"note={response_text.strip()[:400]}"
            )
            if self.beergame_cheatsheet_path:
                # 1) 追加决策日志
                with open(self.beergame_cheatsheet_path, "a", encoding="utf-8") as f:
                    f.write(entry + "\n")
                # 2) 另存一份“当前 cheatsheet”，便于下次加载
                with open(self.beergame_cheatsheet_path + ".current", "w", encoding="utf-8") as f:
                    f.write(self.beergame_cheatsheet_text or "")
        except Exception:
            pass

        return int(order_qty)

    def _extract_order_qty_from_dyncht(
        self, response_text: str, base_order: int, max_order_qty: int
    ) -> int:
        """
        从 DynamicCheatsheet 输出解析订单；失败回退 base_order 并打印。
        """
        candidate = None
        try:
            import re

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
                        for subkey in ("order_qty", "order", "quantity", "value"):
                            subval = val.get(subkey)
                            if isinstance(subval, (int, float)):
                                candidate = int(subval)
                                break
                            if isinstance(subval, str) and subval.strip():
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
                        import re

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
                                for subkey in ("order_qty", "order", "quantity", "value"):
                                    subval = val.get(subkey)
                                    if isinstance(subval, (int, float)):
                                        candidate = int(subval)
                                        break
                                    if isinstance(subval, str) and subval.strip():
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
                                m = re.search(r"-?\d+", val)
                                if m:
                                    candidate = int(m.group(0))
                                    break
            except Exception:
                candidate = None

        if candidate is None:
            candidate = base_order
            print(f"[DynamicCheatsheet][BeerGame] parse failed, fallback to base_order={base_order}")

        candidate = max(0, min(int(candidate), max_order_qty))
        self._last_beergame_note = f"dyncht_final{' (fallback_base_order)' if candidate == base_order else ''}: {response_text[:500]}"
        return candidate

    def run_beergame(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """BeerGame 评测入口（委托 seriousgame_tools 通用流程）。"""
        _ = data_processor

        # 加载/初始化跨 episode 持久化的 cheatsheet
        try:
            save_dir = config.get("save_dir", "results")
            self._load_beergame_cheatsheet(
                os.path.join(save_dir, "dynamic_cheatsheet_beergame.txt")
            )
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

    def run_consulting(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
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

        # 持久化 cheatsheet（咨询模式专用，跨 case 有效）
        self._consulting_cheatsheet_path = os.path.join(resolved_save_path, "consulting_cheatsheet.txt")
        try:
            if os.path.exists(self._consulting_cheatsheet_path):
                with open(self._consulting_cheatsheet_path, "r", encoding="utf-8") as f:
                    self._consulting_cheatsheet = f.read().strip() or "(empty)"
            else:
                self._consulting_cheatsheet = "(empty)"
        except Exception:
            self._consulting_cheatsheet = "(empty)"

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
                "dynamic_cheatsheet": {
                    "approach_name": self.dc_config.approach_name,
                    "generator_prompt_path": str(self.dc_config.generator_prompt_path),
                    "cheatsheet_prompt_path": str(self.dc_config.cheatsheet_prompt_path)
                    if self.dc_config.cheatsheet_prompt_path
                    else None,
                },
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
    # Main run with consulting routing
    # ==========================================================
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

    @staticmethod
    def _extract_utilization(learn_metrics: Dict[str, Any]) -> Optional[float]:
        if not isinstance(learn_metrics, dict):
            return None
        for k in ("utilization", "avg_utilization", "mean_utilization", "resource_utilization"):
            v = learn_metrics.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def _render_edt_cheatsheet_block(self, cheatsheet_text: str, max_chars: int = 1200) -> str:
        txt = (cheatsheet_text or "").strip()
        if not txt or txt == "(empty)":
            return ""
        txt = " ".join(txt.split())
        if len(txt) > max_chars:
            txt = txt[:max_chars].rstrip()
        return (
            "\n\nEDT CHEATSHEET (use as guidance; keep decisions controllable and consistent):\n"
            + txt
            + "\n"
        )

    async def _decide_edt_scenario_schema(
        self,
        base_summary: Dict[str, Any],
        scenario_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Called by seriousgame_tools.
        Mode-A: inject current cheatsheet into prompt; generate schema once; normalize.
        """
        scenario_meta = scenario_meta or {}
        ctx = build_edt_decision_context(
            base_summary=base_summary,
            scenario_meta=scenario_meta,
            max_steps_hint=scenario_meta.get("max_steps"),
        )
        system, user = render_edt_prompt(ctx)

        if self._edt_state is not None:
            system = system + self._render_edt_cheatsheet_block(self._edt_state.cheatsheet_text)

        user = (
            user
            + "\n\nIMPORTANT OUTPUT CONSTRAINTS:\n"
            + "- Output ONLY a single JSON object.\n"
            + "- The JSON object must contain ONLY keys: C, R, P.\n"
            + "- Do not wrap JSON in markdown fences.\n"
        )

        resp = self.generator_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=min(self.max_tokens, 512),
        )
        text = (resp.choices[0].message.content or "").strip()
        raw = self._safe_load_json(text)

        candidate = raw.get("final_answer", raw)
        if isinstance(candidate, str):
            candidate = self._safe_load_json(candidate)
        if not isinstance(candidate, dict):
            candidate = {}

        schema = normalize_edt_schema(candidate, ctx)
        return schema

    def _update_edt_cheatsheet_from_window(self, window_records: List[Dict[str, Any]]) -> None:
        """
        Window-level cheatsheet update (Mode A):
        - Uses (schema, metrics) for the window to propose a compact, actionable cheatsheet update.
        - Updates self._edt_state.cheatsheet_text in-place.
        """
        if not self._edt_state:
            return
        if not window_records:
            return

        current = (self._edt_state.cheatsheet_text or "").strip()
        if not current:
            current = "(empty)"

        # compact window summary for prompting
        compact = []
        for r in window_records[-20:]:
            compact.append(
                {
                    "utilization": r.get("utilization"),
                    "learn_metrics": r.get("learn_metrics", {}),
                    "schema": r.get("schema", {}),
                }
            )

        system = (
            "You are maintaining a concise operational cheatsheet for designing EDT scenarios.\n"
            "Goal: improve utilization and overall efficiency.\n"
            "Update the cheatsheet based on recent (schema, metrics) outcomes.\n"
            "Rules:\n"
            "- Keep the cheatsheet SHORT (max 12 bullet points total).\n"
            "- Prefer actionable heuristics and anti-patterns.\n"
            "- Avoid repeating existing bullets.\n"
            "- Do not mention LLMs, prompts, or evaluation.\n"
            "Output ONLY plain text cheatsheet (bullet points allowed)."
        )

        user = (
            "CURRENT CHEATSHEET:\n"
            f"{current[:3000]}\n\n"
            "RECENT WINDOW OUTCOMES (schema + metrics):\n"
            f"{json.dumps(compact, ensure_ascii=False)[:5000]}\n\n"
            "Now output the UPDATED cheatsheet text."
        )

        resp = self.generator_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=min(self.max_tokens, 512),
        )
        new_txt = (resp.choices[0].message.content or "").strip()
        if not new_txt:
            return

        # update state & history
        self._edt_state.update_cheatsheet(new_txt)
        if self._edt_history_path:
            self._append_history(
                self._edt_history_path,
                {
                    "event": "window_cheatsheet_update",
                    "timestamp": datetime.now().isoformat(),
                    "window_size": len(window_records),
                    "updated_cheatsheet": new_txt,
                },
            )

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
        Hook called by seriousgame_tools after each episode (if present).
        Mode-A: cache the experience; update cheatsheet after each window boundary.
        """
        util = self._extract_utilization(learn_metrics)
        rec = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "episode_id": episode_id,
            "utilization": util,
            "learn_metrics": learn_metrics,
            "schema": schema,
        }
        self._edt_pending.append(rec)

        # Write per-episode trace (optional, for debugging)
        if self._edt_history_path:
            self._append_history(self._edt_history_path, {"event": "episode", **rec})

        # window boundary update
        if self._edt_window_size > 0 and len(self._edt_pending) >= self._edt_window_size:
            window = self._edt_pending[: self._edt_window_size]
            self._edt_pending = self._edt_pending[self._edt_window_size :]
            self._update_edt_cheatsheet_from_window(window)

    def run_edt(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        EDT runner using seriousgame_tools, with Mode-A cheatsheet updates:
        - episode runs once
        - cheatsheet updates at window boundaries within the run
        """
        _ = data_processor
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )
        if not test_samples:
            raise ValueError(f"{self.agent_method.upper()} agent requires test samples but none were provided.")

        # Use seriousgame_tools to prepare run directories etc.
        ctx = edt_prepare_run(
            mode=mode,
            test_samples=test_samples,
            config=config,
            allowed_modes=self.SUPPORTED_MODES,
        )

        # Determine save path (ctx should include one of these)
        resolved_save_path = (
            ctx.get("resolved_save_path")
            or ctx.get("save_path")
            or ctx.get("run_dir")
            or ctx.get("log_dir")
            or os.path.join(config.get("save_dir", "results"), "SeriousGame_EDT", self.agent_method)
        )
        os.makedirs(resolved_save_path, exist_ok=True)
        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Cheatsheet history for EDT (separate from bizbench history)
        self._edt_history_path = os.path.join(resolved_save_path, "edt_cheatsheet_history.jsonl")

        # Initialize cheatsheet state (allow user-specified initial cheatsheet file)
        init_path = None
        edt_cfg = config.get("edt") if isinstance(config, dict) else None
        if isinstance(edt_cfg, dict):
            init_path = edt_cfg.get("initial_cheatsheet_path") or config.get("initial_cheatsheet_path")
        else:
            init_path = config.get("initial_cheatsheet_path")

        self._edt_state = CheatsheetState(cheatsheet_text=self._load_initial_cheatsheet(init_path))

        # Window size (Mode A): default 6; can override via config
        window_size = 1
        if isinstance(edt_cfg, dict):
            window_size = int(edt_cfg.get("cheatsheet_window_size", window_size) or window_size)
        self._edt_window_size = max(1, window_size)

        # reset pending cache
        self._edt_pending = []

        # Persist a small config note for reproducibility
        try:
            run_cfg_path = os.path.join(resolved_save_path, "run_config.json")
            payload = {}
            if os.path.exists(run_cfg_path):
                with open(run_cfg_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            payload["dynamic_cheatsheet_edt"] = {
                "mode_a_window_size": self._edt_window_size,
                "initial_cheatsheet_path": init_path,
            }
            with open(run_cfg_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # Run evaluation via seriousgame_tools
        results, error_log = edt_evaluate_run(agent=self, test_samples=test_samples, config=config, ctx=ctx)

        # Flush remaining pending records into a final cheatsheet update (optional)
        if self._edt_pending:
            self._update_edt_cheatsheet_from_window(self._edt_pending)
            self._edt_pending = []

        # Save final cheatsheet
        final_cheatsheet_path = os.path.join(resolved_save_path, "final_edt_cheatsheet.txt")
        with open(final_cheatsheet_path, "w", encoding="utf-8") as f:
            f.write(self._edt_state.cheatsheet_text if self._edt_state else "(empty)")

        # Save run artifacts through seriousgame_tools
        edt_save_run(results=results, error_log=error_log, config=config, ctx=ctx)

        return results

    # ==========================================================
    # BizBench run (original run() implementation) -> renamed
    # ==========================================================
    def run_bizbench(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
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
            raise ValueError("Configuration missing 'save_dir' for DynamicCheatsheetAgent.")
        task_name = task_name or "unknown_task"

        paths = self._prepare_dirs(task_name, mode, save_dir)
        state = CheatsheetState(
            cheatsheet_text=self._load_initial_cheatsheet(config.get("initial_cheatsheet_path"))
        )

        retrieval_required = {
            "Dynamic_Retrieval",
            "DynamicCheatsheet_RetrievalSynthesis",
        }

        data_embedding_dir = config.get("data_embedding_dir", "bizbench/data/data_embedding")
        if self.dc_config.approach_name in retrieval_required:
            dataset_name = task_name.lower()
            embedding_path = os.path.join(data_embedding_dir, f"{dataset_name}_test_embeddings.csv")
            self._load_embedding_csv(embedding_path)

        def strip_task_instruction(question: str) -> str:
            try:
                from bizbench.data_processor import DataProcessor

                instruction_values = list(DataProcessor.TASK_INSTRUCTIONS.values()) + list(
                    DataProcessor.SPECIAL_FIN_INSTRUCTIONS.values()
                )
                for instr in instruction_values:
                    if instr and question.startswith(instr):
                        return question[len(instr):].lstrip()
            except Exception:
                return question
            return question

        # 根据任务选择合适的 prompt（FormulaEval 等任务使用特化提示）
        self.generator_prompt_for_run, self.cheatsheet_prompt_for_run = self._select_prompts_for_task(
            task_name, self.generator_prompt, self.cheatsheet_prompt
        )

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - DYNAMIC CHEATSHEET")
        print(f"{'='*60}")
        print(f"Samples: {len(test_samples)}")
        print(f"Approach: {self.dc_config.approach_name}")
        print(f"Log dir: {paths['log_dir']}")
        print(f"{'='*60}\n")

        test_workers = config.get("test_workers", 20)
        online_eval_frequency = config.get("online_eval_frequency", 15)

        def parallel_evaluate(
            samples: List[Dict[str, Any]],
            state_snapshot: CheatsheetState,
            start_idx: int = 0,
            prefix: str = "eval",
            retrieval_corpus: Optional[List[str]] = None,
            retrieval_embeddings: Optional[List[List[float]]] = None,
        ):
            results = []
            with ThreadPoolExecutor(max_workers=test_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_sample,
                        sample,
                        start_idx + idx,
                        state_snapshot,
                        data_processor,
                        paths,
                        strip_task_instruction,
                        False,  # update_state
                        False,  # log_history
                        False,  # verbose
                        f"{prefix}_{start_idx + idx}",
                        retrieval_corpus=retrieval_corpus,
                        retrieval_embeddings=retrieval_embeddings,
                    ): idx
                    for idx, sample in enumerate(samples)
                }
                for future in as_completed(futures):
                    results.append(future.result())
            results.sort(key=lambda x: x["index"])
            return results

        sample_records: List[Dict[str, Any]] = []
        predictions: List[str] = []
        targets: List[Any] = []
        errors: List[Dict[str, Any]] = []

        retrieval_corpus = self.retrieval_corpus if self.retrieval_corpus else None
        retrieval_embeddings = self.retrieval_embeddings if self.retrieval_embeddings else None

        if mode == "eval_only":
            eval_records = parallel_evaluate(
                test_samples,
                self._snapshot_state(state),
                0,
                prefix="eval",
                retrieval_corpus=retrieval_corpus,
                retrieval_embeddings=retrieval_embeddings,
            )
            for rec in eval_records:
                sample_records.append(rec)
                predictions.append(rec["final_answer"])
                targets.append(rec["target"])
                if not rec["is_correct"]:
                    errors.append({"index": rec["index"], "prediction": rec["final_answer"], "ground_truth": rec["target"]})
        else:  # online
            total = len(test_samples)
            num_windows = (total + online_eval_frequency - 1) // online_eval_frequency
            print(f"Online windowed eval/train: total {total}, window size {online_eval_frequency}, windows {num_windows}")
            for window_idx in range(num_windows):
                start_idx = window_idx * online_eval_frequency
                end_idx = min((window_idx + 1) * online_eval_frequency, total)
                window_samples = test_samples[start_idx:end_idx]

                print(f"\n{'='*60}")
                print(f"WINDOW {window_idx + 1}/{num_windows} | Samples {start_idx} to {end_idx - 1}")
                print(f"{'='*60}")

                # 1) 并行只读评估（使用当前 cheatsheet 快照）
                snapshot = self._snapshot_state(state)
                window_records = parallel_evaluate(
                    window_samples,
                    snapshot,
                    start_idx,
                    prefix=f"eval_w{window_idx+1}",
                    retrieval_corpus=retrieval_corpus,
                    retrieval_embeddings=retrieval_embeddings,
                )
                window_correct = sum(1 for r in window_records if r["is_correct"])
                window_total = len(window_records)
                window_acc = window_correct / window_total if window_total else 0.0
                print(f"[Window {window_idx + 1}] Eval accuracy: {window_acc:.3f} ({window_correct}/{window_total})")

                sample_records.extend(window_records)
                predictions.extend([r["final_answer"] for r in window_records])
                targets.extend([r["target"] for r in window_records])
                errors.extend(
                    [{"index": r["index"], "prediction": r["final_answer"], "ground_truth": r["target"]} for r in window_records if not r["is_correct"]]
                )

                # 2) 串行训练/更新（保持顺序一致）
                print(f"[Window {window_idx + 1}] Start training/cheatsheet update (sequential)")
                for local_idx, sample in enumerate(window_samples):
                    global_idx = start_idx + local_idx
                    self._process_sample(
                        sample,
                        global_idx,
                        state,
                        data_processor,
                        paths,
                        strip_task_instruction,
                        update_state=True,
                        log_history=True,
                        verbose=True,
                        call_prefix=f"train_{global_idx}",
                        retrieval_corpus=retrieval_corpus,
                        retrieval_embeddings=retrieval_embeddings,
                    )

        accuracy = data_processor.evaluate_accuracy(predictions, targets) if predictions and targets else 0.0

        test_results = {
            "accuracy": accuracy,
            "total": len(sample_records),
            "correct": sum(1 for r in sample_records if r["is_correct"]),
            "samples": sample_records,
        }
        error_log = {"errors": errors, "accuracy": accuracy}

        with open(os.path.join(paths["resolved_save_path"], "test_results.json"), "w", encoding="utf-8") as f:
            json.dump({"test_results": test_results, "error_log": error_log}, f, indent=2)

        config_payload = dict(config)
        config_payload.update(
            {
                "run_subdir": paths["run_subdir"],
                "resolved_save_path": paths["resolved_save_path"],
                "dynamic_cheatsheet": {
                    "approach_name": self.dc_config.approach_name,
                    "generator_prompt_path": str(self.dc_config.generator_prompt_path),
                    "cheatsheet_prompt_path": str(self.dc_config.cheatsheet_prompt_path)
                    if self.dc_config.cheatsheet_prompt_path
                    else None,
                },
            }
        )

        with open(os.path.join(paths["resolved_save_path"], "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)

        final_cheatsheet_path = os.path.join(paths["resolved_save_path"], "final_cheatsheet.txt")
        with open(final_cheatsheet_path, "w", encoding="utf-8") as f:
            f.write(state.cheatsheet_text)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Results saved to: {paths['resolved_save_path']}")
        print(f"{'='*60}\n")

        return test_results

    # ==========================================================
    # Prompt specialization (BizBench) - unchanged
    # ==========================================================
    def _format_input(self, sample: Dict[str, Any], index: int, question_override: Optional[str] = None) -> str:
        raw_question = question_override if question_override is not None else sample.get("question", "")
        raw_context = sample.get("context", "")
        question = (raw_question or "").strip()
        context = (raw_context or "").strip()
        parts = [f"Question #{index + 1}:\n{question}"]
        if context:
            parts.append(f"\nContext:\n{context}")
        return "\n".join(parts).strip()

    def _select_prompts_for_task(self, task_name: str, base_generator_prompt: str, base_cheatsheet_prompt: str) -> Tuple[str, str]:
        if task_name.lower() == "formulaeval":
            generator_prompt = (
                "You are a code completion assistant. The function/method signature is provided above.\n"
                "Fill in ONLY the missing body lines with correct indentation.\n"
                "Do NOT repeat the class definition or the signature.\n"
                "Do NOT return JSON or analysis; only return one ```python``` code block containing the body lines.\n"
                "If the signature is present, just give the indented body lines; if only the body is missing, fill it directly.\n"
                "Ensure the code is executable and self-contained (no external files/paths/imports beyond stdlib if needed).\n\n"
                "Question:\n[[QUESTION]]"
            )
            cheatsheet_prompt = base_cheatsheet_prompt
            return generator_prompt, cheatsheet_prompt

        return base_generator_prompt, base_cheatsheet_prompt

    def _process_sample(
        self,
        sample: Dict[str, Any],
        idx: int,
        state: CheatsheetState,
        data_processor,
        paths: Dict[str, str],
        strip_task_instruction_fn,
        update_state: bool = True,
        log_history: bool = True,
        verbose: bool = False,
        call_prefix: Optional[str] = None,
        retrieval_corpus: Optional[List[str]] = None,
        retrieval_embeddings: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        input_txt = self._format_input(sample, idx)
        stripped_question = strip_task_instruction_fn(sample.get("question", ""))
        input_txt_for_cheatsheet = self._format_input(sample, idx, question_override=stripped_question)
        target = sample.get("target")

        original_input_corpus = state.input_history + [input_txt]
        original_input_embeddings = None
        if retrieval_corpus and retrieval_embeddings:
            idx_in_retrieval = self.retrieval_index.get(input_txt)
            if idx_in_retrieval is None or idx_in_retrieval >= len(retrieval_embeddings):
                idx_in_retrieval = len(retrieval_embeddings) - 1
            permutation = [i for i in range(len(retrieval_corpus)) if i != idx_in_retrieval] + [idx_in_retrieval]
            prev_inputs = [retrieval_corpus[i] for i in permutation]
            prev_embeddings = [retrieval_embeddings[i] for i in permutation]
            original_input_corpus = prev_inputs
            original_input_embeddings = prev_embeddings
            generator_outputs_so_far = [self.retrieval_outputs.get(i, "") for i in permutation]
        else:
            generator_outputs_so_far = state.output_history

        output_dict = self.language_model.advanced_generate(
            approach_name=self.dc_config.approach_name,
            input_txt=input_txt,
            cheatsheet=state.cheatsheet_text,
            generator_template=self.generator_prompt_for_run,
            cheatsheet_template=self.cheatsheet_prompt_for_run,
            cheatsheet_question=input_txt_for_cheatsheet,
            temperature=self.dc_config.temperature,
            max_tokens=self.max_tokens,
            max_num_rounds=self.dc_config.max_num_rounds,
            allow_code_execution=self.dc_config.allow_code_execution,
            code_execution_flag="EXECUTE CODE!",
            add_previous_answers_to_cheatsheet=self.dc_config.add_previous_answers,
            original_input_corpus=original_input_corpus,
            original_input_embeddings=original_input_embeddings,
            generator_outputs_so_far=generator_outputs_so_far,
            retrieve_top_k=self.dc_config.retrieve_top_k,
            log_dir=paths["log_dir"],
            call_prefix=call_prefix or f"sample_{idx}",
        )

        final_answer = output_dict.get("final_answer")
        final_cheatsheet = output_dict.get("final_cheatsheet") or state.cheatsheet_text
        is_correct = data_processor.answer_is_correct(final_answer, target)

        if update_state:
            state.update_cheatsheet(final_cheatsheet)
            state.append_example(
                input_txt=input_txt,
                generator_output=output_dict.get("final_output", ""),
                generator_answer=final_answer or "",
            )
            if retrieval_corpus and retrieval_embeddings:
                idx_in_retrieval = self.retrieval_index.get(input_txt)
                if idx_in_retrieval is not None:
                    self.retrieval_outputs[idx_in_retrieval] = final_answer or ""

        if log_history and update_state:
            history_payload = {
                "index": idx,
                "input": input_txt,
                "steps": output_dict.get("steps"),
                "final_cheatsheet": final_cheatsheet,
            }
            self._append_history(paths["cheatsheet_history_path"], history_payload)

        if verbose:
            print(f"[Sample {idx + 1}] Correct: {is_correct} | Final Answer: {final_answer}")

        return {
            "index": idx,
            "input": input_txt,
            "target": target,
            "final_answer": final_answer,
            "final_output": output_dict.get("final_output"),
            "cheatsheet": final_cheatsheet,
            "is_correct": is_correct,
        }

    # ==========================================================
    # Unified run router
    # ==========================================================
    def run(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        task_name = str(config.get("task_name", getattr(data_processor, "task_name", ""))).lower()

        # EDT routing
        if "edt" in task_name:
            return self.run_edt(
                mode=mode,
                test_samples=test_samples or [],
                data_processor=data_processor,
                config=config,
            )

        # Otherwise keep original behavior
        return self.run_bizbench(
            mode=mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
