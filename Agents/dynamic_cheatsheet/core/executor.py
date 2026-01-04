"""
Utilities to execute Python code blocks emitted by the Generator during
Dynamic Cheatsheet runs. Execution is sandboxed by reusing the secure
runtime from bizbench.data_processor.
"""

from __future__ import annotations

from textwrap import indent

from StructuredReasoning.data_processor import exec_python


def _run_in_sandbox(code: str, timeout: float = 2.0) -> str:
    """
    Execute arbitrary Python code inside the BizBench sandbox while capturing
    stdout. The return value mirrors the message style of the original
    Dynamic Cheatsheet executor.
    """
    instrumented = "\n".join(
        [
            "import io",
            "import contextlib",
            "_ds_output_stream = io.StringIO()",
            "with contextlib.redirect_stdout(_ds_output_stream):",
            indent(code, "    "),
            "_ds_output_stream.getvalue()",
        ]
    )

    result = exec_python(instrumented, timeout=timeout)
    failure = result.get("failure_reason")
    if failure and failure != "timeout":
        return f"Error in execution: {failure}"
    if failure == "timeout":
        return "Execution took too long, aborting..."

    captured_output = result.get("return_val")
    if captured_output is None or str(captured_output).strip() == "":
        return (
            "(No output was generated. Add a print statement to inspect "
            "intermediate values.)"
        )
    return str(captured_output).strip()


def extract_and_run_python_code(txt: str, timeout: float = 2.0) -> str | None:
    """
    Extract the last ```python``` block from `txt`, ensure a visible print
    statement exists, execute it securely, and return the captured output.
    """

    def extract_code(input_str: str) -> str:
        try:
            return input_str.split("```python", 1)[1].split("```", 1)[0].strip()
        except IndexError:
            raise ValueError("No valid Python code block found.") from None

    def ensure_print_statement(code: str) -> str:
        lines = code.splitlines()
        if not lines:
            return code
        last_line = lines[-1].rstrip()
        if (
            last_line
            and not last_line.startswith(("print(", "#"))
            and "return" not in last_line
        ):
            lines[-1] = f"print({last_line})"
        return "\n".join(lines)

    if "```python" not in txt:
        return None

    code_block = extract_code(txt)
    code_with_print = ensure_print_statement(code_block)
    python_output = _run_in_sandbox(code_with_print, timeout=timeout)
    return f"Output of the Python code above:\n```\n{python_output}\n```"














