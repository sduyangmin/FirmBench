import ast
import contextlib
import faulthandler
import io
import math
import multiprocessing
import os
import platform
import json
import random
import re
import signal
import tempfile
from typing import Any, Dict, List, Optional, Tuple


LETTER_OPTIONS = ["A", "B", "C", "D", "E", "F"]


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL data for StructuredReasoning tasks.

    Args:
        data_path: Path to JSON Lines file.

    Returns:
        List of parsed JSON objects.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    samples: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {data_path}")
    return samples

def parse_finer_context(all_context: str) -> Tuple[str, str]:
    """
    Parse FINER style context into (input_text, instruction/question).
    Expected format: "Instruction: ...\\nInput: ...\\nAnswer:"
    """
    if "Input: " in all_context and "Instruction: " in all_context:
        instruction_part = all_context.split("Input: ")[0].strip()
        instruction_part = instruction_part.split("Instruction: ")[1].strip()
        remaining = all_context.split("Input: ", 1)[1]
        input_text = remaining.split("Answer:", 1)[0].strip()
        return input_text, instruction_part
    return "", all_context

def parse_formula_context(all_context: str) -> Tuple[str, str]:
    """
    Parse formula task context into (context, question).
    Expected format: "... Question: \"...\". Answer:"
    """
    if "Question: " in all_context and ". Answer:" in all_context:
        _, question_part = all_context.split("Question: ", 1)
        question_text = question_part.split(". Answer:", 1)[0].strip()
        if question_text.startswith('"') and question_text.endswith('"'):
            question_text = question_text[1:-1]
        question_text += (
            " Your answer should be a plain floating point number, "
            "round to the nearest hundredth if necessary. Do the necessary conversions, "
            "for example 5 million should be 5000000.0."
        )
        return "", question_text
    return "", all_context


class TimeoutException(Exception):
    """Custom timeout exception for sandboxed execution."""


class WriteOnlyStringIO(io.StringIO):
    """StringIO that raises when read to avoid leaking stdout/stderr."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root: str):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    Disable dangerous functionality for untrusted code execution.
    """
    import resource
    import shutil

    if maximum_memory_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    dangerous_os_attrs = [
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "getcwd",
        "chdir",
    ]
    for attr in dangerous_os_attrs:
        if hasattr(os, attr):
            setattr(os, attr, None)

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    import sys

    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def unsafe_execute(code: str, result_container, timeout: float):
    import shutil

    with create_tempdir():
        original_rmtree = shutil.rmtree
        original_rmdir = os.rmdir
        original_chdir = os.chdir

        reliability_guard()

        return_val = None
        failure_reason = None
        try:
            exec_globals: Dict[str, Any] = {}
            with swallow_io():
                with time_limit(timeout):
                    block = ast.parse(code, mode="exec")
                    last_expr = ast.Expression(block.body.pop().value)
                    fake_globals: Dict[str, Any] = {}
                    fake_locals = fake_globals

                    exec(compile(block, "<string>", mode="exec"), fake_globals, fake_locals)
                    return_val = eval(
                        compile(last_expr, "<string>", mode="eval"),
                        fake_globals,
                        fake_locals,
                    )
        except TimeoutException:
            failure_reason = "timeout"
        except BaseException as exc:  # noqa: BLE001
            failure_reason = str(exc)

        result_container.append(return_val)
        result_container.append(failure_reason)

        shutil.rmtree = original_rmtree
        os.rmdir = original_rmdir
        os.chdir = original_chdir


def exec_python(code: str, timeout: float = 1.0) -> Dict[str, Any]:
    """
    Execute python code snippet and capture the last expression result.
    """
    ctx = multiprocessing.get_context("fork")
    manager = multiprocessing.Manager()
    result = manager.list()

    process = ctx.Process(target=unsafe_execute, args=(code, result, timeout))
    process.start()
    process.join(timeout + 1)
    if process.is_alive():
        process.kill()

    if not result:
        result.append(None)
        result.append("timeout")

    if len(result) == 2:
        return {"return_val": result[0], "failure_reason": result[1]}

    return {"return_val": None, "failure_reason": "unknown"}


class DataProcessor:
    """
    Task-specific processor for StructuredReasoning datasets.
    """

    PROGRAM_SYNTHESIS = "program_synthesis"
    QUANTITY_EXTRACTION = "quantity_extraction"
    MULTIPLE_CHOICE = "multiple_choice"
    FORMULA_EVAL = "formula_eval"
    SPECIAL_FIN = "special_fin"

    TASK_CATEGORY = {
        "FinCode": PROGRAM_SYNTHESIS,
        "CodeFinQA": PROGRAM_SYNTHESIS,
        "CodeTAT-QA": PROGRAM_SYNTHESIS,
        "SEC-NUM": QUANTITY_EXTRACTION,
        "TAT-QA": QUANTITY_EXTRACTION,
        "ConvFinQA": QUANTITY_EXTRACTION,
        "FinKnow": MULTIPLE_CHOICE,
        "FormulaEval": FORMULA_EVAL,
        "finer": SPECIAL_FIN,
        "formula": SPECIAL_FIN,
    }

    NUMERIC_TOLERANCE = 0.01  # 1%
    ABSOLUTE_EPS = 1e-6
    SPECIAL_FIN_INSTRUCTIONS = {
        "finer": (
            "INSTRUCTION: YOU ARE A FINANCIAL SENTIMENT AND KEY-FACT EXTRACTION ASSISTANT. "
            "READ THE FOLLOWING INSTRUCTION AND CONTEXT, THEN OUTPUT THE REQUESTED VALUES "
            "EXACTLY AS THEY APPEAR (COMMA-SEPARATED IF MULTIPLE ARE REQUIRED)."
        ),
        "formula": (
            "INSTRUCTION: YOU ARE A FINANCIAL CALCULATION ASSISTANT. SOLVE THE QUESTION CAREFULLY "
            "AND RESPOND WITH A PLAIN FLOATING POINT NUMBER (ROUND TO THE NEAREST HUNDREDTH "
            "AND EXPAND ANY UNITS, E.G., 5 MILLION -> 5000000.0)."
        ),
    }

    TASK_INSTRUCTIONS = {
        PROGRAM_SYNTHESIS: (
            "INSTRUCTION: YOU ARE A FINANCIAL CODING ASSISTANT. "
            "READ THE PROBLEM (AND ANY CONTEXT). "
            "RESPOND WITH A PYTHON PROGRAM ENCLOSED IN ```python``` FENCES. "
            "ENSURE THE FINAL LINE OF THE PROGRAM EVALUATES TO THE NUMERIC ANSWER. "
            "AFTER THE CODE BLOCK, ALSO PROVIDE THE NUMERIC RESULT IN THE FORM [[VALUE]]. "
            "DO NOT INCLUDE EXTRA COMMENTARY. "
            "STRICTLY FOLLOW THE PROVIDED REFERENCE PROGRAM/FORMULA: WHEN THE TASK ASKS FOR CHANGE/DIFFERENCE, "
            "COMPUTE THE RAW DIFFERENCE (NEW - OLD) IN THE SAME UNITS/PERCENTAGE POINTS UNLESS THE REFERENCE PROGRAM "
            "EXPLICITLY USES A PERCENTAGE-CHANGE FORMULA."
        ),
        QUANTITY_EXTRACTION: (
            "INSTRUCTION: YOU ARE A FINANCIAL INFORMATION EXTRACTION ASSISTANT. "
            "GIVEN THE QUESTION AND CONTEXT, EXTRACT THE REQUESTED NUMERIC QUANTITY VERBATIM FROM THE TEXT. "
            "ANSWER ONLY WITH [[EXTRACTED_SPAN]] (NO EXPLANATION OR ADDITIONAL TEXT)."
        ),
        MULTIPLE_CHOICE: (
            "INSTRUCTION: YOU ARE A FINANCIAL MULTIPLE-CHOICE EXPERT. "
            "CHOOSE THE SINGLE BEST OPTION (A, B, C, ...). "
            "ANSWER ONLY WITH THE OPTION IN THE FORMAT [[A]], [[B]], ETC."
        ),
        FORMULA_EVAL: (
            "INSTRUCTION: YOU ARE A CODE COMPLETION ASSISTANT. "
            "THE FUNCTION OR METHOD SIGNATURE IS ALREADY PROVIDED ABOVE. "
            "FILL IN ONLY THE MISSING BODY LINES (DO NOT REPEAT THE SIGNATURE OR CLASS DEFINITION). "
            "IF THE BODY IS A SINGLE RETURN, OUTPUT ONLY THAT RETURN STATEMENT. "
            "RETURN YOUR COMPLETION INSIDE A ```python``` BLOCK WITH PROPER INDENTATION AND NO EXTRA COMMENTARY."
        ),
        SPECIAL_FIN: "",  # handled separately via SPECIAL_FIN_INSTRUCTIONS
    }
    FORMULA_NUM_TESTS = 4
    FORMULA_MAX_ATTEMPTS = 5
    FORMULA_REL_TOL = 1e-4
    FORMULA_ABS_TOL = 1e-4
    FORMULA_RATE_KEYWORDS = ("rate", "growth", "yield", "return")
    FORMULA_INT_HINTS = ("count", "num", "payments", "years", "period")

    def __init__(self, task_name: str):
        self.task_name = task_name
        if task_name not in self.TASK_CATEGORY:
            raise ValueError(f"Unsupported task: {task_name}")
        self.category = self.TASK_CATEGORY[task_name]

    # ----------------------------------------------------------------------
    # Data processing
    # ----------------------------------------------------------------------
    def process_task_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for sample in raw_data:
            handler = getattr(
                self,
                f"_process_{self.category}_sample",
                self._process_generic_sample,
            )
            processed.append(handler(sample))
        return processed

    def _process_program_synthesis_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        context_parts: List[str] = []
        if sample.get("context"):
            context_parts.append(str(sample["context"]).strip())
        if sample.get("program"):
            context_parts.append(f"Reference Program:\n{sample['program']}")
        context = "\n\n".join([part for part in context_parts if part]).strip()
        question = self._prepend_instruction(
            self.category,
            sample.get("question", "").strip(),
        )
        target = str(sample.get("answer", "")).strip()

        return {
            "context": context,
            "question": question,
            "target": target,
            "others": {
                "context_type": sample.get("context_type"),
                "task": sample.get("task", self.task_name),
            },
        }

    def _process_quantity_extraction_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "context": sample.get("context", ""),
            "question": self._prepend_instruction(
                self.category,
                sample.get("question", ""),
            ),
            "target": str(sample.get("answer", "")).strip(),
            "others": {
                "context_type": sample.get("context_type"),
                "task": sample.get("task", self.task_name),
            },
        }

    def _process_multiple_choice_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question_text = sample.get("question", "").strip()
        options = sample.get("options") or []
        formatted_options = "\n".join(
            f"{LETTER_OPTIONS[idx]}. {opt}" for idx, opt in enumerate(options)
        )
        instruction = self._prepend_instruction(self.category, "")
        question = "\n\n".join(
            part for part in [instruction, f"{question_text}\nOptions:\n{formatted_options}".strip()] if part
        )

        raw_answer = str(sample.get("answer", "")).strip()
        target_label = self._resolve_choice_label(raw_answer, options)

        target = {
            "label": target_label,
            "options": options,
        }

        return {
            "context": sample.get("context", ""),
            "question": question,
            "target": target,
            "others": {
                "context_type": sample.get("context_type"),
                "task": sample.get("task", self.task_name),
            },
        }

    def _process_formula_eval_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raw_question = str(sample.get("question", "")).strip()
        question = self._prepend_instruction(self.category, raw_question)
        answer = sample.get("answer", "")
        return {
            "context": "",
            "question": question,
            "target": {
                "prompt": raw_question,
                "completion": str(answer),
            },
            "others": {
                "context_type": sample.get("context_type"),
                "task": sample.get("task", self.task_name),
            },
        }

    def _process_special_fin_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.task_name == "finer":
            input_text, question = parse_finer_context(sample.get("context", ""))
            target = sample.get("target") or sample.get("answer", "")
            return {
                "context": input_text,
                "question": question,
                "target": str(target),
                "others": {
                    "original_context": sample.get("context", ""),
                    "task": self.task_name,
                },
            }
        if self.task_name == "formula":
            context, question = parse_formula_context(sample.get("context", ""))
            target = sample.get("target") or sample.get("answer", "")
            return {
                "context": context,
                "question": question,
                "target": str(target),
                "others": {
                    "original_context": sample.get("context", ""),
                    "task": self.task_name,
                },
            }
        return self._process_generic_sample(sample)

    def _process_generic_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "context": sample.get("context", ""),
            "question": sample.get("question", ""),
            "target": sample.get("answer", ""),
            "others": sample,
        }

    # ----------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------
    def answer_is_correct(self, predicted: str, ground_truth: Any) -> bool:
        if predicted is None:
            return False

        if self.category == self.PROGRAM_SYNTHESIS:
            return self._check_program_synthesis(predicted, ground_truth)
        if self.category == self.QUANTITY_EXTRACTION:
            return self._check_quantity_extraction(predicted, ground_truth)
        if self.category == self.MULTIPLE_CHOICE:
            return self._check_multiple_choice(predicted, ground_truth)
        if self.category == self.FORMULA_EVAL:
            return self._check_formula_eval(predicted, ground_truth)
        if self.category == self.SPECIAL_FIN:
            if self.task_name == "finer":
                return self._finer_answer_is_correct(predicted, ground_truth)
            if self.task_name == "formula":
                return self._formula_answer_is_correct(predicted, ground_truth)

        return str(predicted).strip() == str(ground_truth).strip()

    def evaluate_accuracy(self, predictions: List[str], ground_truths: List[Any]) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        total = len(predictions)
        correct = sum(
            1 for pred, truth in zip(predictions, ground_truths)
            if self.answer_is_correct(pred, truth)
        )
        if self.category == self.SPECIAL_FIN and self.task_name == "finer":
            return self._evaluate_finer_accuracy(predictions, ground_truths)
        if self.category == self.SPECIAL_FIN and self.task_name == "formula":
            return self._evaluate_formula_accuracy(predictions, ground_truths)
        return correct / total if total else 0.0

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    CODE_BLOCK_REGEX = re.compile(r"```(?:\w+)?\s*\n(.*?)(?:```)", re.DOTALL)

    def _check_program_synthesis(self, predicted: str, ground_truth: str) -> bool:
        pred_val = self._parse_numeric_prediction(predicted)
        true_val = self._extract_first_numeric(ground_truth)

        if true_val is None:
            return self._normalize_basic(predicted) == self._normalize_basic(ground_truth)

        if pred_val is None:
            return False

        if abs(true_val) < self.ABSOLUTE_EPS:
            return abs(pred_val - true_val) <= self.NUMERIC_TOLERANCE

        return abs(pred_val - true_val) / abs(true_val) <= self.NUMERIC_TOLERANCE

    def _check_quantity_extraction(self, predicted: str, ground_truth: str) -> bool:
        pred_val = self._parse_numeric_prediction(predicted)
        true_val = self._extract_first_numeric(ground_truth)

        if pred_val is not None and true_val is not None:
            return math.isclose(pred_val, true_val, rel_tol=0.0, abs_tol=self.ABSOLUTE_EPS)

        return self._normalize_span(predicted) == self._normalize_span(ground_truth)

    def _check_multiple_choice(self, predicted: str, target: Dict[str, Any]) -> bool:
        options = target.get("options") or []
        gold_label = target.get("label")
        pred_label = self._parse_choice_label(predicted, options)
        return pred_label == gold_label

    def _check_formula_eval(self, predicted: str, ground_truth: Any) -> bool:
        if isinstance(ground_truth, dict) and ground_truth.get("prompt") is not None:
            prompt_code = ground_truth.get("prompt", "")
            true_completion = ground_truth.get("completion", "")
            reference_program = self._assemble_formula_program(prompt_code, true_completion)
            target_info = self._extract_formula_target(reference_program)
            if not target_info:
                return self._normalize_basic(predicted) == self._normalize_basic(true_completion)

            # 对于类似 DynamicCheatsheet 返回的 JSON/结构化输出，尝试先抽取其中的代码段
            cleaned_predicted = predicted
            try:
                text = self._decode_prediction_text(str(predicted))
                # 如果输出看起来像 JSON，并包含代码相关字段，则优先使用其中内容
                if any(k in text for k in ('"solution"', "'solution'", '"completion"', "'completion'", '"response"', "'response'")):
                    import re as _re

                    m = _re.search(r"\{.*\}", text, flags=_re.DOTALL)
                    if m:
                        obj = json.loads(m.group(0))
                        if isinstance(obj, dict):
                            for key in ["solution", "completion", "response"]:
                                if key in obj:
                                    cleaned_predicted = str(obj[key])
                                    break
                            else:
                                cleaned_predicted = text
                else:
                    cleaned_predicted = text
            except Exception:
                cleaned_predicted = predicted

            prediction_program = self._prepare_formula_prediction(
                cleaned_predicted,
                prompt_code,
                target_info=target_info,
            )
            if not reference_program or not prediction_program:
                return False

            seed = abs(hash(prompt_code)) % (2 ** 32)
            for attempt in range(self.FORMULA_MAX_ATTEMPTS):
                cases = self._generate_formula_eval_cases(target_info, seed + attempt)
                harness = self._build_formula_harness(target_info, cases)

                ref_outputs, ref_error = self._execute_formula_program(reference_program, harness)
                if ref_error or ref_outputs is None:
                    continue

                pred_outputs, pred_error = self._execute_formula_program(prediction_program, harness)
                if pred_error or pred_outputs is None:
                    return False

                return self._compare_formula_outputs(pred_outputs, ref_outputs)

            return False

        return self._normalize_basic(predicted) == self._normalize_basic(ground_truth)

    # ------------------------------------------------------------------
    # Special finance task helpers (finer / formula)
    # ------------------------------------------------------------------
    def _finer_answer_is_correct(
        self,
        predicted: str,
        ground_truth: str,
        return_counts: bool = False
    ) -> bool:
        pred = [val.lower().strip() for val in predicted.split(",")]
        label = [val.lower().strip() for val in ground_truth.split(",")]
        correct = 0

        if len(pred) != len(label):
            if len(pred) > len(label):
                pred = pred[:len(label)]
            else:
                pred += [""] * (len(label) - len(pred))

        for prediction, truth in zip(pred, label):
            try:
                truth_val = eval(truth)
                pred_val = eval(prediction.replace(",", "").replace("$", ""))
                if pred_val == truth_val:
                    correct += 1
                    continue
            except Exception:
                pass
            if prediction == truth:
                correct += 1

        if return_counts:
            return correct, len(pred)
        return correct == len(pred) and len(pred) > 0

    def _formula_answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        更灵活地比较 formula 题目的数值答案。

        支持以下形式的预测输出：
        - 纯数字字符串，例如 "40.0"、"35,835.68"
        - 含有解释文本但最后包含正确数字
        - 含有 JSON/代码块并在字段如 "final_answer" 中给出数值
        """
        if predicted is None:
            return False

        # 先做基本清洗
        pred_text = self._decode_prediction_text(str(predicted))
        gt_text = (ground_truth or "").strip()

        # 尝试解析标准数值形式的 ground truth
        try:
            gt_val = float(gt_text.replace(",", ""))
        except Exception:
            gt_val = None

        # 1) 尝试直接把整个预测当作数字解析
        try:
            pred_val_direct = float(pred_text.replace(",", ""))
            if gt_val is not None:
                return pred_val_direct == gt_val
        except Exception:
            pass

        # 2) 尝试从 JSON/代码块中解析 "final_answer" 字段
        if '"final_answer"' in pred_text or "'final_answer'" in pred_text:
            try:
                import re as _re
                m = _re.search(r"\{.*\}", pred_text, flags=_re.DOTALL)
                if m:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict) and "final_answer" in obj:
                        fa = obj["final_answer"]
                        # final_answer 可能本身就是数值
                        if isinstance(fa, (int, float)):
                            if gt_val is not None:
                                return float(fa) == gt_val
                        else:
                            fa_str = str(fa)
                            try:
                                fa_val = float(fa_str.replace(",", ""))
                                if gt_val is not None:
                                    return fa_val == gt_val
                            except Exception:
                                # 回退到字符串比较
                                if fa_str.strip() == gt_text:
                                    return True
            except Exception:
                pass

        # 3) 通用：用正则提取预测中的所有数字，优先使用最后一个
        try:
            import re as _re

            numbers = _re.findall(
                r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", pred_text.replace(",", "")
            )
            if numbers:
                last_num = numbers[-1]
                if gt_val is not None:
                    try:
                        pred_val = float(last_num)
                        return pred_val == gt_val
                    except Exception:
                        pass
                # ground_truth 也是字符串时，允许与最后一个数字直接比对
                if last_num == gt_text or last_num == gt_text.replace(",", ""):
                    return True
        except Exception:
            pass

        # 4) 最后退回到原始的字符串比较
        return pred_text.strip() == gt_text

    def _evaluate_finer_accuracy(self, predictions: List[str], targets: List[str]) -> float:
        if len(predictions) != len(targets):
            raise ValueError("Predictions and ground truths must have same length")

        correct = 0
        total = 0
        for pred, truth in zip(predictions, targets):
            c, t = self._finer_answer_is_correct(pred, truth, return_counts=True)
            correct += c
            total += t
        return correct / total if total else 0.0

    def _evaluate_formula_accuracy(self, predictions: List[str], targets: List[str]) -> float:
        if len(predictions) != len(targets):
            raise ValueError("Predictions and ground truths must have same length")
        correct = sum(1 for pred, truth in zip(predictions, targets)
                      if self._formula_answer_is_correct(pred, truth))
        return correct / len(predictions) if predictions else 0.0

    def _assemble_formula_program(self, prompt: str, completion: str) -> str:
        prompt_block = prompt or ""
        completion_block = completion or ""
        if prompt_block and not prompt_block.endswith("\n"):
            prompt_block = f"{prompt_block}\n"
        program = f"{prompt_block}{completion_block}"
        program = self._append_trailing_newline(program.rstrip())
        return self._ensure_dataclass_import(program)

    def _ensure_dataclass_import(self, code: str) -> str:
        if "@dataclass" in code and "from dataclasses import dataclass" not in code:
            return f"from dataclasses import dataclass\n\n{code}"
        return code

    def _prepare_formula_prediction(
        self,
        predicted: str,
        prompt: str,
        target_info: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        if not predicted:
            return None

        predicted = self._decode_prediction_text(predicted)

        code_block = self._extract_code_block(predicted)
        if code_block:
            normalized_block = code_block.strip("\n")
            if self._looks_like_complete_program(normalized_block):
                return self._append_trailing_newline(self._ensure_dataclass_import(normalized_block))
            formatted_block = self._ensure_formula_body_indent(normalized_block, target_info)
            return self._assemble_formula_program(prompt, formatted_block)

        stripped = predicted.strip()
        if not stripped:
            return None

        if self._looks_like_complete_program(stripped):
            return self._append_trailing_newline(self._ensure_dataclass_import(stripped))

        snippet = predicted.strip("\n")
        snippet = self._ensure_formula_body_indent(snippet, target_info)
        return self._assemble_formula_program(prompt, snippet)

    def _extract_formula_target(self, code: str) -> Optional[Dict[str, Any]]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        target_node: Optional[ast.FunctionDef] = None
        target_parent: Optional[ast.ClassDef] = None
        class_fields: Dict[str, List[Tuple[str, Optional[str]]]] = {}

        def recurse(node: ast.AST, current_class: Optional[ast.ClassDef] = None):
            nonlocal target_node, target_parent
            if isinstance(node, ast.ClassDef):
                fields: List[Tuple[str, Optional[str]]] = []
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        fields.append((stmt.target.id, self._annotation_to_str(stmt.annotation)))
                class_fields[node.name] = fields
                for stmt in node.body:
                    recurse(stmt, node)
                return

            if isinstance(node, ast.FunctionDef):
                target_node = node
                target_parent = current_class
                for stmt in node.body:
                    recurse(stmt, current_class)
                return

            for child in ast.iter_child_nodes(node):
                recurse(child, current_class)

        recurse(tree)

        if not target_node:
            return None

        arg_specs = [(arg.arg, self._annotation_to_str(arg.annotation)) for arg in target_node.args.args]

        return {
            "func_name": target_node.name,
            "arg_specs": arg_specs,
            "is_method": target_parent is not None,
            "class_name": target_parent.name if target_parent else None,
            "class_fields": class_fields.get(target_parent.name, []) if target_parent else [],
        }

    @staticmethod
    def _annotation_to_str(annotation: Optional[ast.AST]) -> Optional[str]:
        if annotation is None:
            return None
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Attribute):
            return annotation.attr
        if isinstance(annotation, ast.Subscript):
            return DataProcessor._annotation_to_str(annotation.value)
        return None

    def _generate_formula_eval_cases(self, target_info: Dict[str, Any], seed: int) -> List[Any]:
        rng = random.Random(seed)
        cases: List[Any] = []
        call_args = self._call_arg_specs(target_info)

        for _ in range(self.FORMULA_NUM_TESTS):
            if target_info["is_method"]:
                fields = {
                    name: self._sample_formula_value(name, annotation, rng)
                    for name, annotation in target_info.get("class_fields", [])
                }
                args = {
                    name: self._sample_formula_value(name, annotation, rng)
                    for name, annotation in call_args
                }
                cases.append({"fields": fields, "args": args})
            else:
                args = {
                    name: self._sample_formula_value(name, annotation, rng)
                    for name, annotation in call_args
                }
                cases.append(args)
        return cases

    def _call_arg_specs(self, target_info: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
        specs = target_info.get("arg_specs", [])
        if target_info["is_method"] and specs and specs[0][0] == "self":
            return specs[1:]
        return specs

    def _sample_formula_value(self, name: str, annotation: Optional[str], rng: random.Random) -> Any:
        lname = name.lower()

        if annotation == "int" or lname in {"n", "n_payments"} or any(hint in lname for hint in self.FORMULA_INT_HINTS):
            return max(1, rng.randint(1, 12))

        if any(keyword in lname for keyword in self.FORMULA_RATE_KEYWORDS):
            return round(rng.uniform(0.01, 0.4), 4)

        if "share" in lname or "price" in lname or "value" in lname or "income" in lname:
            return round(rng.uniform(10.0, 1000.0), 4)

        if "dividend" in lname or "cash" in lname:
            return round(rng.uniform(1.0, 500.0), 4)

        value = rng.uniform(25.0, 750.0)
        if rng.random() < 0.3:
            value *= -1
        if abs(value) < 1e-3:
            value = 1.0
        return round(value, 4)

    def _build_formula_harness(self, target_info: Dict[str, Any], cases: List[Any]) -> str:
        cases_literal = repr(cases)
        if target_info["is_method"]:
            class_name = target_info["class_name"]
            func_name = target_info["func_name"]
            return (
                f"__FORMULA_TEST_CASES__ = {cases_literal}\n\n"
                "def __run_formula_eval_tests():\n"
                "    results = []\n"
                "    for case in __FORMULA_TEST_CASES__:\n"
                f"        instance = {class_name}(**case['fields'])\n"
                f"        result = instance.{func_name}(**case['args'])\n"
                "        results.append(result)\n"
                "    return results\n\n"
                "__run_formula_eval_tests()\n"
            )

        func_name = target_info["func_name"]
        return (
            f"__FORMULA_TEST_CASES__ = {cases_literal}\n\n"
            "def __run_formula_eval_tests():\n"
            "    results = []\n"
            "    for args in __FORMULA_TEST_CASES__:\n"
            f"        result = {func_name}(**args)\n"
            "        results.append(result)\n"
            "    return results\n\n"
            "__run_formula_eval_tests()\n"
        )

    def _execute_formula_program(self, program_code: str, harness: str) -> Tuple[Optional[Any], Optional[str]]:
        code = f"{program_code.rstrip()}\n\n{harness}"
        result = exec_python(code, timeout=2.0)
        return result.get("return_val"), result.get("failure_reason")

    def _compare_formula_outputs(self, predicted: Any, expected: Any) -> bool:
        if isinstance(expected, list) and isinstance(predicted, list):
            if len(predicted) != len(expected):
                return False
            return all(self._compare_formula_value(p, e) for p, e in zip(predicted, expected))
        return self._compare_formula_value(predicted, expected)

    def _compare_formula_value(self, predicted: Any, expected: Any) -> bool:
        if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
            return math.isclose(
                float(predicted),
                float(expected),
                rel_tol=self.FORMULA_REL_TOL,
                abs_tol=self.FORMULA_ABS_TOL,
            )

        if isinstance(expected, (list, tuple)) and isinstance(predicted, (list, tuple)):
            if len(expected) != len(predicted):
                return False
            return all(self._compare_formula_value(p, e) for p, e in zip(predicted, expected))

        if isinstance(expected, dict) and isinstance(predicted, dict):
            if expected.keys() != predicted.keys():
                return False
            return all(self._compare_formula_value(predicted[key], expected[key]) for key in expected)

        return self._normalize_basic(str(predicted)) == self._normalize_basic(str(expected))

    @staticmethod
    def _looks_like_complete_program(code: str) -> bool:
        stripped = (code or "").lstrip()
        return stripped.startswith("class ") or stripped.startswith("def ") or "@dataclass" in stripped

    @staticmethod
    def _append_trailing_newline(code: str) -> str:
        if not code.endswith("\n"):
            return f"{code}\n"
        return code

    def _ensure_formula_body_indent(
        self,
        code: str,
        target_info: Optional[Dict[str, Any]]
    ) -> str:
        if not code or not target_info:
            return code

        base_indent = 8 if target_info.get("is_method") else 4
        lines = code.splitlines()
        meaningful = [line for line in lines if line.strip()]
        if not meaningful:
            return code

        min_leading = min(self._leading_whitespace_width(line) for line in meaningful)
        formatted_lines: List[str] = []
        for line in lines:
            stripped = line.lstrip(" \t")
            if not stripped:
                formatted_lines.append("")
                continue
            current_leading = self._leading_whitespace_width(line)
            relative = max(0, current_leading - min_leading)
            adjusted_indent = " " * (base_indent + relative)
            formatted_lines.append(f"{adjusted_indent}{stripped}")
        return "\n".join(formatted_lines)

    @staticmethod
    def _leading_whitespace_width(line: str) -> int:
        width = 0
        for ch in line:
            if ch == " ":
                width += 1
            elif ch == "\t":
                width += 4
            else:
                break
        return width

    @staticmethod
    def _decode_prediction_text(text: str) -> str:
        if not text:
            return ""
        # Handle common escaped newline/tab sequences that sometimes appear verbatim
        decoded = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        # Remove surrounding quotes occasionally returned by models
        if decoded.startswith('"') and decoded.endswith('"'):
            decoded = decoded[1:-1]
        return decoded

    @staticmethod
    def _normalize_basic(text: str) -> str:
        return (text or "").strip().lower()

    @staticmethod
    def _normalize_span(text: str) -> str:
        cleaned = (text or "").strip().lower()
        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.replace("$", "")
        cleaned = cleaned.replace("%", "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    @staticmethod
    def _clean_numeric_string(text: str) -> Optional[str]:
        if text is None:
            return None

        if isinstance(text, (int, float)):
            return str(text)

        extracted = re.findall(r"[-+]?\d[\d,\.]*", text)
        if not extracted:
            return None
        number = extracted[0]
        number = number.replace(",", "")
        return number

    def _extract_first_numeric(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        cleaned = self._clean_numeric_string(str(value))
        if cleaned is None:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _prepend_instruction(self, category: str, question: str) -> str:
        if category == self.SPECIAL_FIN:
            instruction = self.SPECIAL_FIN_INSTRUCTIONS.get(self.task_name, "")
        else:
            instruction = self.TASK_INSTRUCTIONS.get(category, "")
        instruction = instruction.strip()
        if not instruction:
            return question

        if question:
            return f"{instruction}\n\n{question}"
        return instruction

    def _extract_code_block(self, text: str) -> Optional[str]:
        if not text:
            return None
        matches = self.CODE_BLOCK_REGEX.findall(text)
        if not matches:
            return None
        return matches[-1].strip()

    def _try_execute_python(self, text: str) -> Optional[float]:
        code = self._extract_code_block(text)
        if not code:
            return None
        result = exec_python(code)
        return_val = result.get("return_val")
        failure_reason = result.get("failure_reason")
        if failure_reason:
            return None
        if isinstance(return_val, (int, float)):
            return float(return_val)
        try:
            return float(return_val)
        except (TypeError, ValueError):
            return None

    def _parse_numeric_prediction(self, predicted: str) -> Optional[float]:
        if not predicted:
            return None

        if "```" in predicted:
            executed_value = self._try_execute_python(predicted)
            if executed_value is not None:
                return executed_value

        brace_value = self._brace_extract_numeric(predicted)
        if brace_value is not None:
            return brace_value

        return self._extract_first_numeric(predicted)

    def _brace_extract_value(self, text: str) -> Optional[str]:
        if not text:
            return None
        matches = re.findall(r"\[\[(.*)\]\]", text)
        if len(matches) != 1:
            return None
        value = matches[0].strip()
        for token in [",", "$", "%"]:
            value = value.replace(token, "")
        return value.strip()

    def _brace_extract_numeric(self, text: str) -> Optional[float]:
        raw_value = self._brace_extract_value(text)
        if raw_value is None:
            return None

        upper_value = raw_value.upper()
        if upper_value in LETTER_OPTIONS:
            return float(LETTER_OPTIONS.index(upper_value))

        if self._is_float(raw_value):
            return float(raw_value)

        return None

    def _parse_choice_label(self, predicted: str, options: List[str]) -> str:
        brace_numeric = self._brace_extract_numeric(predicted)
        if brace_numeric is not None:
            idx = int(round(brace_numeric))
            if 0 <= idx < len(options):
                return LETTER_OPTIONS[idx]

        return self._resolve_choice_label(predicted, options)

    @staticmethod
    def _is_float(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _resolve_choice_label(answer: str, options: List[str]) -> str:
        if not answer:
            return ""
        normalized = answer.strip()
        upper = normalized.upper()

        if upper in LETTER_OPTIONS[: len(options)]:
            return upper

        if normalized.isdigit():
            idx = int(normalized)
            if 0 <= idx < len(options):
                return LETTER_OPTIONS[idx]

        for idx, option in enumerate(options):
            if normalized.lower() == option.strip().lower():
                return LETTER_OPTIONS[idx]

        return upper

