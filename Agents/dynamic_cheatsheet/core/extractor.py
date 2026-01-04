"""
Helper functions to extract answers, cheatsheets, and evaluation summaries
from model responses.
"""


def extract_answer(response: str) -> str:
    """
    Extract the final answer from the model response.
    Mirrors the logic used in the original Dynamic Cheatsheet repo to keep
    compatibility with the expected FINAL ANSWER format.
    """
    if "<answer>" in response:
        try:
            txt = response.split("<answer>")[-1].strip()
            txt = txt.split("</answer>")[0].strip()
            return txt
        except Exception:
            return "No final answer found"
    if "FINAL ANSWER" not in response:
        return "No final answer found"
    try:
        response = response.split("FINAL ANSWER")[-1].strip()
        if response and response[0] == ":":
            response = response[1:].strip()

        idx_1 = response.find("'''")
        idx_2 = response.find("```")
        if min(idx_1, idx_2) != -1:
            if idx_1 != -1 and (idx_1 < idx_2 or idx_2 == -1):
                response = response.split("'''")[1].strip()
            else:
                response = response.split("```")[1].strip()
        else:
            if idx_1 == -1 and idx_2 != -1:
                response = response.split("```")[1].strip()
            elif idx_2 == -1 and idx_1 != -1:
                response = response.split("'''")[1].strip()

        first_line = response.split("\n")[0].strip().lower()
        if first_line == "python":
            response = "\n".join(response.split("\n")[1:]).strip()
        return response
    except Exception:
        return "No final answer found"


def extract_cheatsheet(response: str, old_cheatsheet: str) -> str:
    """
    Extract the cheatsheet block from curator outputs.
    """
    response = response.strip()
    if "<cheatsheet>" in response:
        try:
            txt = response.split("<cheatsheet>")[1].strip()
            txt = txt.split("</cheatsheet>")[0].strip()
            return txt
        except Exception:
            return old_cheatsheet
    return old_cheatsheet


def extract_solution(
    response: str,
    header: str = "SOLUTION EVALUATION:",
    error_message: str = "No solution evaluation found",
) -> str:
    """
    Extract solution-evaluation snippets if present.
    """
    response = response.strip()
    try:
        txt = response.split(header)[1]
        try:
            txt = txt.split("'''")[1].strip()
        except Exception:
            return txt.strip()
    except Exception:
        return error_message
    return txt














