# utils/consulting_tools.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from dotenv import load_dotenv
# ============================================================
# 配置与特殊标记
# ============================================================

# 面试官结束本次面试的标记（尽量不与自然语言冲突）
INTERVIEW_END_TOKEN = "<|CASE_INTERVIEW_END|>"

load_dotenv()
# LLM 模型：默认固定为 gpt-5（可用环境变量覆盖）
CONSULTING_LLM_MODEL = os.getenv("CONSULTING_LLM_MODEL", "deepseek-v3")

OPENAI_BASE_URL = "http://35.220.164.252:3888/v1/"
API_KEY = os.getenv('OPENAI_API_KEY', '')

# 你可以用 consulting/case_dataset.py 中的系统 prompt 替换下面这两个
INTERVIEWER_SYSTEM_PROMPT = f"""
You are the interviewer in a consulting-style case interview with an LLM candidate.

Each turn you receive:
1) These instructions.
2) The full text of ONE case (problem/background, any sections such as
   "Information to be provided if requested", "If asked for market information",
   "Hints", "Key questions", "Analysis", "Possible recommendations / approaches",
   "Solution", "Case wrap-up", "Interviewer notes", etc.).
3) The chat history so far.

Your job is ONLY to produce the next interviewer message.

1. What you may reveal
- The problem statement, scenario description, and general background are information
  the candidate is allowed to know.
- Sections like "Information to be provided if requested", "If asked for ...",
  "Further information", "Hints" or similar are GATED facts.
- Sections like "Key questions", "Possible recommendations / approaches", "Analysis",
  "Solution", "Case wrap-up", "Interviewer notes" are INTERNAL GUIDANCE ONLY.

Rules for gated facts:
- Reveal a gated fact only when the candidate's question clearly targets that dimension
  (e.g., market size, growth, costs, customers, competition, operations, risks).
- Reveal one logical piece at a time, not the whole section at once.
- Never quote or expose "Solution", "Analysis", "Case wrap-up",
  "Possible recommendations / approaches" or "Interviewer notes" directly.
  Use them only to decide what to probe and which facts matter.

Do NOT invent or assume new facts beyond the case. If the candidate asks for
information that is not in the case and not covered by any gated section, say that the
case does not provide that detail and invite them to proceed with reasonable assumptions
or move to another relevant angle.

2. How to run the interview

First turn:
- Briefly set up the situation in your own words (1–3 sentences).
- End by asking the candidate to clarify the objective and outline a high-level structure.

Later turns:
- Read the latest candidate answer and the history.
- Answer their concrete questions using only allowed and already-unlocked information
  (plus any newly unlocked gated facts).
- Ask ONE focused follow-up at a time, pushing them toward structured business
  reasoning (e.g., profitability, market / customer / competition, operations, risks).
- Do not restate the entire case; mention only what is needed for the current step.

Pacing and depth:
- Use the case length hint (e.g. "Short 15 Minutes", "Medium 30 Minutes",
  "Long 45 Minutes") and the conversation so far to manage depth:
  * Short cases: aim for at least 3–4 candidate answers before closing.
  * Medium cases: aim for 5–7 candidate answers.
  * Long cases: aim for 7–10 candidate answers with deeper quantitative or conceptual work.
- If the candidate keeps asking for more data without analyzing, gently redirect them to:
  (a) summarize what they know, and (b) propose a structure or hypothesis BEFORE you give more data.
- If the candidate is stuck, you may give a small hint or suggest one missing dimension,
  but do NOT present a full framework or full solution.

3. Ending the case

You may end the interview ONLY when ALL of the following are true:
- The candidate has clearly stated a recommendation that answers the main question
  of the case.
- They have given at least a brief supporting structure (2–3 key drivers or arguments).
- You have given short, high-level feedback and, if appropriate, added one or two
  important missing points.

Ending protocol:
- NEVER end the interview in your very first message.
- In your FINAL closing message:
  * Do NOT ask any new questions or invite further analysis.
  * Optionally give concise feedback and highlight key drivers.
  * Append the exact token {INTERVIEW_END_TOKEN} as the VERY LAST characters.
- In all earlier messages you MUST NOT output {INTERVIEW_END_TOKEN}.

4. Style and constraints

- Speak as a professional human interviewer: concise, neutral, business-like.
- Ask at most one or a small cluster of closely related questions per turn.
- Never mention "case text", "sections", "gated information", "solutions", or any
  internal labels; to the candidate you are simply an interviewer.
- Use only information from the case text and the chat history; do not bring in
  outside knowledge.
- Keep each interviewer message concise: at most 500 words in each turn. Do not write long essays.

Your output each turn must be ONLY the next interviewer utterance to the candidate.
"""


JUDGE_SYSTEM_PROMPT = """
You are a senior consulting interviewer evaluating the performance of a CANDIDATE
in a case interview.

You will receive:
- case_text: the full written case (problem, background, solution, etc.);
- transcript_text: the complete dialogue between INTERVIEWER and CANDIDATE,
  in chronological order. Each line clearly indicates who is speaking.

Your job is to assess ONLY the CANDIDATE, not the interviewer.

Evaluate the candidate along FOUR dimensions plus an overall score:

1) structure (0–10)
   - How well does the candidate understand and restate the problem and objective?
   - Do they propose a clear, logical, and MECE-enough structure or approach early on?
   - Do they use hypothesis-driven thinking and adjust their structure as new information appears?

2) quant (0–10)
   - Does the candidate ask for the right type of information or data when needed?
   - Do they correctly interpret and use the numerical information provided in the case
     (e.g., doing rough calculations, sanity checks, comparisons)?
   - Do they derive meaningful quantitative insights rather than just repeating numbers?

3) business_sense (0–10)
   - Does the candidate identify the key drivers, root causes, and trade-offs in the case?
   - Are their conclusions and recommendations commercially reasonable and consistent
     with the information given?
   - Do they recognize important risks/uncertainties and, when appropriate, suggest
     sensible next steps or mitigations?

4) communication (0–10)
   - Is the candidate’s communication clear, concise, and well-structured?
   - Do they signpost their thinking (e.g., “first/second/third”) without being verbose?
   - Do they interact professionally with the interviewer, responding to questions,
     picking up on hints, and keeping a natural case-interview flow?

In addition, provide:

5) overall (0–10)
   - Your holistic judgment of the candidate’s performance on this case.
   - This is NOT just an arithmetic average; it reflects whether you would be
     comfortable recommending this candidate for a consulting role.

Scoring guidelines (be strict and well-calibrated across many cases):
- 0–2: very weak (almost no useful contribution or completely off-track).
- 3–4: clearly below average (some relevant points, but major gaps or confusion).
- 5–6: average candidate (generally reasonable but shallow, incomplete, or inconsistent).
- 7: above average (solid performance with notable but fixable weaknesses).
- 8: very strong (consultant-level performance with only minor issues).
- 9–10: truly exceptional (outstanding on almost all dimensions; reserve for rare cases).

Additional rules:
- If the candidate barely speaks, never proposes a clear structure, or never gives a
  concrete recommendation, most scores should be in the 0–3 range.
- Do NOT reward verbosity alone; reward clear, structured, business-relevant thinking.
- Penalize hallucinated facts that contradict or go beyond the case_text.

Output format:
Return ONLY a single valid JSON object with this exact schema:
{
  "structure": float,        # 0–10
  "quant": float,            # 0–10
  "business_sense": float,   # 0–10
  "communication": float,    # 0–10
  "overall": float,          # 0–10
  "feedback": string         # short textual feedback / summary (within 400 words)
}
No extra text before or after the JSON.
""".strip()


# ============================================================
# 基础 LLM 封装
# ============================================================

def _build_openai_client() -> OpenAI:
    
    if OPENAI_BASE_URL:
        return OpenAI(api_key=API_KEY, base_url=OPENAI_BASE_URL)

    return OpenAI()


def _chat_text(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 8192,
) -> str:
    resp = client.chat.completions.create(
        model=CONSULTING_LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort="low",
    )
    return resp.choices[0].message.content or ""


def _chat_json(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=CONSULTING_LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        print(content)
        response_cleaned = content.strip()
        # Try to find JSON content if wrapped in other text
        if not response_cleaned.startswith('{'):
            start_idx = response_cleaned.find('{')
            if start_idx != -1:
                response_cleaned = response_cleaned[start_idx:]
        if not response_cleaned.endswith('}'):
            end_idx = response_cleaned.rfind('}')
            if end_idx != -1:
                response_cleaned = response_cleaned[:end_idx + 1]
        return json.loads(response_cleaned)
    except Exception:
        return {
            "structure": 0.0,
            "quant": 0.0,
            "business_sense": 0.0,
            "communication": 0.0,
            "overall": 0.0,
            "feedback": f"JSON parse error. Raw content: {content[:5000]}",
        }


# ============================================================
# 面试官（interviewer）& 单 case 运行
# ============================================================

def _build_interviewer_messages(
    case_text: str,
    opening: str,
    history: List[Dict[str, str]],
    remaining_turns: int,
) -> List[Dict[str, str]]:
    # history 是 [{role: interviewer/candidate, content: str}, ...]
    history_lines = [
        f"{h.get('role', 'unknown').upper()}: {h.get('content', '')}"
        for h in history
    ]
    history_block = "\n".join(history_lines) if history_lines else "(no dialogue yet)"

    user_content = (
        "You are running a consulting case interview.\n\n"
        "=== CASE TEXT (only you can see) ===\n"
        f"{case_text}\n\n"
        "=== OPENING / INITIAL DESCRIPTION ===\n"
        f"{opening}\n\n"
        "=== DIALOGUE SO FAR ===\n"
        f"{history_block}\n\n"
        "Now produce your NEXT TURN as the interviewer.\n"
        f"- Maximum remaining turns (including this one) is {remaining_turns}.\n"
        f"- Keep this turn concise: no more than 500 words.\n"
        f"- If this is the first turn, briefly introduce the case and ask the candidate "
        f"to structure the problem.\n"
        f"- If you believe you have seen enough and want to end the interview, "
        f"you MUST include the exact token {INTERVIEW_END_TOKEN} in your reply, "
        f"then give a short closing remark.\n"
        "Do NOT mention that you can see the full case text."
    )
    # {"role": "system", "content": INTERVIEWER_STYLE_HINT},
    return [
        {"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _run_single_interview(
    client: OpenAI,
    agent: Any,
    case: Dict[str, Any],
    max_turns: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    单个 case 的完整面试流程：
    - LLM 担任 interviewer（可访问 case_text）；
    - Agent 担任 candidate（只能看到对话 history）；
    - 返回 (history, judge_scores)：
        history: [{role: "interviewer"/"candidate", content: str}, ...]
        judge_scores: evaluator LLM 输出的 JSON。
    """
    case_id = str(case.get("case_id", "unknown_case"))
    case_text = case.get("case_text", "")
    opening = case.get("opening", "") or "The interviewer briefly describes the client's situation and objective."

    # 通知 Agent：case 开始
    if hasattr(agent, "on_case_start"):
        agent.on_case_start(case_id)

    history: List[Dict[str, str]] = []
    turns_used = 0
    ended_by_interviewer = False

    while turns_used < max_turns:
        remaining = max_turns - turns_used

        # interviewer 说话
        interviewer_msgs = _build_interviewer_messages(
            case_text=case_text,
            opening=opening,
            history=history,
            remaining_turns=remaining,
        )
        interviewer_reply = _chat_text(client, interviewer_msgs, temperature=0.3)
        if not interviewer_reply:
            interviewer_reply = f"{INTERVIEW_END_TOKEN} Thank you, let's stop here."

        history.append({"role": "interviewer", "content": interviewer_reply})
        turns_used += 1

        # 检查是否结束
        if INTERVIEW_END_TOKEN in interviewer_reply:
            ended_by_interviewer = True
            break

        if turns_used >= max_turns:
            break

        # candidate 回复：接口为 reply(case_id, history)
        if hasattr(agent, "reply"):
            candidate_reply = agent.reply(case_id, history)
        else:
            candidate_reply = "(Agent.reply() not implemented.)"
        history.append({"role": "candidate", "content": candidate_reply})
        turns_used += 1

    # case 结束：通知 Agent 做总结/记忆更新
    if hasattr(agent, "on_case_end"):
        try:
            agent.on_case_end(case_id, case_text, history)
        except TypeError:
            # 如果签名不完全匹配，退回只传 case_id
            agent.on_case_end(case_id)

    # 为了日志可读性，把结束 token 从文本中拿掉（只保留在元信息里）
    cleaned_history: List[Dict[str, str]] = []
    for turn in history:
        content = turn.get("content", "")
        if INTERVIEW_END_TOKEN in content:
            content = content.replace(INTERVIEW_END_TOKEN, "").strip()
        cleaned_history.append({**turn, "content": content})
    history = cleaned_history

    # 评测者：对整段对话打分
    transcript_lines = [
        f"{h.get('role', 'unknown').upper()}: {h.get('content', '')}"
        for h in history
    ]
    transcript_text = "\n".join(transcript_lines)

    judge_messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You are now given a consulting case and the full dialogue between interviewer and candidate.\n\n"
                "=== CASE TEXT ===\n"
                f"{case_text}\n\n"
                "=== DIALOGUE TRANSCRIPT ===\n"
                f"{transcript_text}\n\n"
                "Evaluate the CANDIDATE's performance only (do not evaluate the interviewer), "
                "strictly following the scoring instructions. Return ONLY the JSON object."
            ),
        }
    ]
    judge_scores = _chat_json(client, judge_messages, temperature=0.1)
    judge_scores.setdefault("ended_by_interviewer", ended_by_interviewer)
    judge_scores.setdefault("turns_used", turns_used)

    return history, judge_scores


# ============================================================
# 评测主入口：evaluate_consulting_set
# ============================================================

def evaluate_consulting_set(
    *,
    agent: Any,
    test_samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    log_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Consulting 数据集评测入口（与 BizBench / BeerGame 接口形式保持一致）：

    - 由 AMemAgent.run_consulting(...) 调用；
    - 数据集/划分名称从 config 中读取（可选）；
    - 只在 log_dir 下写：
        - consulting_interview_states.json ：所有 case 的对话 + 打分；
    - 返回：
        results: 全局指标（不含 per-case transcript）
        error_log: 失败 case 信息（目前如果运行中抛异常会记录）
    """
    os.makedirs(log_dir, exist_ok=True)
    client = _build_openai_client()

    dataset_name = config.get("dataset_name", config.get("task_name", "Consulting"))
    split_name = config.get("split_name", "test")
    max_turns = int(config.get("consulting_max_turns", 12))

    interview_states: List[Dict[str, Any]] = []
    per_case_scores: List[Dict[str, Any]] = []
    error_log: Dict[str, Any] = {"failed_cases": []}

    for idx, case in enumerate(test_samples):
        case_id = str(case.get("case_id", f"case_{idx:04d}"))
        print(f"[Consulting] Running case {idx+1}/{len(test_samples)}: {case_id}")

        try:
            history, judge_scores = _run_single_interview(
                client=client,
                agent=agent,
                case=case,
                max_turns=max_turns,
            )
        except Exception as e:
            error_log["failed_cases"].append(
                {"case_index": idx, "case_id": case_id, "error": repr(e)}
            )
            continue

        interview_states.append(
            {
                "case_index": idx,
                "case_id": case_id,
                "meta": case.get("meta", {}),
                "history": history,
                "judge_scores": judge_scores,
            }
        )
        per_case_scores.append({"case_id": case_id, **judge_scores})

    # 聚合统计
    def _avg(field: str) -> float:
        vals = []
        for s in per_case_scores:
            v = s.get(field)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "split": split_name,
        "num_cases": len(test_samples),
        "num_finished": len(per_case_scores),
        "num_failed": len(error_log["failed_cases"]),
        "metrics": {
            "structure": _avg("structure"),
            "quant": _avg("quant"),
            "business_sense": _avg("business_sense"),
            "communication": _avg("communication"),
            "overall": _avg("overall"),
        },
    }

    # 写全局对话记录文件
    states_path = Path(log_dir) / "consulting_interview_states.json"
    with states_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "split": split_name,
                "num_cases": len(test_samples),
                "cases": interview_states,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 返回给 AMemAgent.run_consulting，由它来写 test_results.json / run_config.json
    results = summary
    return results, error_log

# ============================================================
# Consulting: agent-side helpers (model-agnostic prompt/query) + run helpers
# ============================================================

def consulting_build_candidate_query(*, case_id: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Build a compact decision state for the candidate agent:
    - query: for memory retrieval
    - turns: number of candidate turns so far
    - last_interviewer_msg: last interviewer utterance
    - transcript_text: plain text transcript (role: content)
    """
    turns = sum(1 for h in history if h.get("role") == "candidate")

    last_interviewer_msg = ""
    for h in reversed(history):
        if h.get("role") == "interviewer":
            last_interviewer_msg = h.get("content", "") or ""
            break

    transcript_lines = [f"{h.get('role','unknown')}: {h.get('content','')}" for h in history]
    transcript_text = "\n".join(transcript_lines) if transcript_lines else "[no previous dialogue]"

    query = (
        f"consulting case={case_id} "
        f"candidate_turns={turns} "
        f"last_interviewer={last_interviewer_msg}"
    )

    return {
        "case_id": case_id,
        "turns": turns,
        "last_interviewer_msg": last_interviewer_msg,
        "transcript_text": transcript_text,
        "query": query,
    }


_CANDIDATE_SYSTEM_PROMPT = (
    "You are the CANDIDATE in a consulting-style case interview.\n"
    "You DO NOT see the internal case text, only the dialogue history.\n"
    "Behave like a top-tier consulting candidate:\n"
    "- structure the problem explicitly when appropriate,\n"
    "- reason in a hypothesis-driven way,\n"
    "- use simple quantitative checks when possible,\n"
    "- communicate clearly and concisely.\n\n"
    "Respond ONLY with what you would say next as the candidate.\n"
    "Wrap your answer in a JSON object of the form:\n"
    "  {\"reply\": \"<your answer>\"}\n"
    "Do not include any other fields."
)

def consulting_render_candidate_prompt(
    *,
    case_id: str,
    state: Dict[str, Any],
    retrieved: str = "",
) -> Tuple[str, str]:
    """
    Render model-agnostic (system, user) prompts for candidate reply generation.
    """
    transcript_text = state.get("transcript_text", "") or "[no previous dialogue]"
    last_interviewer_msg = state.get("last_interviewer_msg", "") or ""

    user_parts = [
        f"Current case ID: {case_id}",
        "",
        "Dialogue so far (Interviewer / Candidate):",
        transcript_text,
        "",
        "Interviewer just said:",
        last_interviewer_msg or "[no interviewer message found]",
    ]
    if retrieved:
        user_parts += [
            "",
            "Some of your previous notes or remembered information:",
            retrieved,
        ]
    user_parts += [
        "",
        'Now respond with your next candidate message, wrapped in JSON as {"reply": "..."}.',
    ]
    return _CANDIDATE_SYSTEM_PROMPT, "\n".join(user_parts)


def consulting_extract_candidate_reply(data: Dict[str, Any]) -> str:
    """
    Extract candidate reply from LLM JSON. Provide a robust fallback.
    """
    reply = data.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        reply = (
            "Let me first structure the key issues and then outline a "
            "hypothesis-driven approach to address the client's problem."
        )
    return reply.strip()


def consulting_format_memory_note(*, case_id: str, state: Dict[str, Any], reply: str) -> str:
    turns = int(state.get("turns", 0) or 0)
    last_interviewer_msg = state.get("last_interviewer_msg", "") or ""
    return (
        f"[CONSULTING][case_id={case_id}][turn={turns}] "
        f"Interviewer: {last_interviewer_msg}\n"
        f"Candidate: {reply}"
    )


def consulting_prepare_run(
    *,
    mode: str,
    test_samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    allowed_modes: set,
    agent_method: str,
) -> Dict[str, Any]:
    """
    Prepare run directories and lightweight run context for consulting evaluation.
    """
    if mode not in allowed_modes:
        raise ValueError(f"{agent_method.upper()} agent only supports modes {allowed_modes}, got '{mode}'")
    if not test_samples:
        raise ValueError("Consulting requires non-empty test_samples")

    save_dir = config.get("save_dir", "results")
    task_name = config.get("task_name", "Consulting")

    run_subdir = f"{task_name}/{agent_method}/{mode}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    resolved_save_path = os.path.join(save_dir, run_subdir)
    os.makedirs(resolved_save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{agent_method.upper()} - CONSULTING EVALUATION")
    print(f"{'='*60}")
    print(f"Cases: {len(test_samples)}")
    print(f"Save dir: {resolved_save_path}")
    print(f"{'='*60}\n")

    return {
        "task_name": task_name,
        "save_dir": save_dir,
        "run_subdir": run_subdir,
        "resolved_save_path": resolved_save_path,
    }


def consulting_evaluate_run(
    *,
    agent: Any,
    test_samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute consulting evaluation.
    """
    return evaluate_consulting_set(
        agent=agent,
        test_samples=test_samples,
        config=config,
        log_dir=ctx["resolved_save_path"],
    )


def consulting_save_run(
    *,
    results: Dict[str, Any],
    error_log: Dict[str, Any],
    config: Dict[str, Any],
    ctx: Dict[str, Any],
) -> None:
    """
    Save run artifacts (test_results.json, run_config.json) in a consistent format.
    """
    resolved_save_path = ctx["resolved_save_path"]

    with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump({"test_results": results, "error_log": error_log}, f, indent=2, ensure_ascii=False)

    config_payload = dict(config)
    config_payload["run_subdir"] = ctx["run_subdir"]
    config_payload["resolved_save_path"] = resolved_save_path

    with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"{ctx.get('task_name','Consulting')} - RUN COMPLETE")
    print(f"{'='*60}")
    print(
        f"Num cases: {results.get('num_cases')}, "
        f"finished: {results.get('num_finished')}, "
        f"failed: {results.get('num_failed')}"
    )
    print(f"Metrics: {results.get('metrics')}")
    print(f"Results saved to: {resolved_save_path}")
    print(f"{'='*60}\n")
