# case_dataset.py
"""
Casebook 面试数据集 + 固定面试官环境 + InterviewState 定义。
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

from utils.llm import timed_llm_call
from utils.tools import initialize_clients


# ============ 面试官配置 ============

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

INTERVIEWER_MODEL = os.getenv("INTERVIEWER_MODEL", "gpt-5")
INTERVIEWER_PROVIDER = os.getenv("INTERVIEWER_PROVIDER", "openai")
INTERVIEWER_END_TOKEN = "<<<INTERVIEW_OVER>>>"

_interviewer_client, _, _ = initialize_clients(INTERVIEWER_PROVIDER)

# =======================
# 评测用 Judge 模型配置
# =======================

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5")
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "openai")
_judge_client, _, _ = initialize_clients(JUDGE_PROVIDER)

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


# ============ 数据结构 ============

@dataclass
class CaseItem:
    """单个 case 的基础信息：ID + 完整文本"""
    case_id: str
    case_text: str


@dataclass
class InterviewState:
    """
    一场面试的完整状态（用于保存 / 评测），内部可以包含 case_text。
    """
    case_id: str
    case_index: int
    case_text: str
    turns: int
    done: bool
    history: List[Dict[str, str]]  # [{"speaker": "interviewer"/"candidate", "text": "..."}]


@dataclass
class CandidateStateView:
    """
    暴露给面试者模型的只读视图：去掉 case_text，防止作弊。
    """
    case_id: str
    case_index: int
    turns: int
    done: bool
    history: List[Dict[str, str]]


def make_candidate_state_view(state: InterviewState) -> CandidateStateView:
    """从 InterviewState 构造面试者可见的视图（不含 case_text）"""
    return CandidateStateView(
        case_id=state.case_id,
        case_index=state.case_index,
        turns=state.turns,
        done=state.done,
        history=list(state.history),
    )


# ============ 工具函数 ============

def history_to_transcript(history: List[Dict[str, str]]) -> str:
    """把内部 history 转成简单对话文本"""
    lines = []
    for turn in history:
        speaker = "Interviewer" if turn["speaker"] == "interviewer" else "Candidate"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def call_judge_llm(
    case_text: str,
    history: List[Dict[str, str]],
    model: str = JUDGE_MODEL,
) -> Dict:
    """
    调用 LLM Judge，对一场面试给出 JSON 格式评分。
    输入: case 原始文案 + 完整 history
    输出: Python dict（解析后的 JSON），如果解析失败则返回一个退化结果
    """
    transcript = history_to_transcript(history)
    user_prompt = (
        "Below is a consulting-style case and the full interview transcript.\n\n"
        "===== CASE TEXT =====\n"
        f"{case_text}\n\n"
        "===== INTERVIEW TRANSCRIPT =====\n"
        f"{transcript}\n\n"
        "Please evaluate the candidate strictly following your instructions and output JSON only."
    )

    resp, _ = timed_llm_call(
        client=_judge_client,
        api_provider=JUDGE_PROVIDER,
        model=model,
        prompt="",
        role="judge",
        call_id="judge",
        max_tokens=512,
        use_json_mode=False,
        temperature=0.2,  # 评测尽量稳定
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = resp.strip()

    # 解析 JSON，防止偶尔不合法
    try:
        result = json.loads(raw)
        if isinstance(result, dict) and "scores" in result:
            return result
    except Exception:
        pass

    # 如果解析失败，给一个兜底结构，方便下游代码不崩
    return {
        "scores": {
            "understanding": 0,
            "structuring": 0,
            "analysis": 0,
            "quant_reasoning": 0,
            "creativity": 0,
            "communication": 0,
            "overall": 0,
        },
        "comment": f"LLM judge output could not be parsed as valid JSON. Raw output was:\n{raw}",
    }


def call_interviewer_llm(
    system_prompt: str,
    user_content: str,
    model: str = INTERVIEWER_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """调用固定的面试官 LLM（统一走 utils.llm）。"""
    resp, _ = timed_llm_call(
        client=_interviewer_client,
        api_provider=INTERVIEWER_PROVIDER,
        model=model,
        prompt="",
        role="interviewer",
        call_id="interviewer",
        max_tokens=max_tokens,
        use_json_mode=False,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.strip()


def save_interview_states_to_json(states: List[InterviewState], filepath: str):
    """将多个 InterviewState 保存成一个 json 文件（列表形式）"""
    data = [asdict(s) for s in states]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_interview_states_from_json(filepath: str) -> List[InterviewState]:
    """从 json 文件中恢复多个 InterviewState"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [InterviewState(**d) for d in data]


# ============ Casebook 面试环境 ============

class CasebookInterviewDataset:
    """
    封装 casebook + 固定面试官的交互环境。

    - __len__     -> case 数量
    - iter_cases  -> (index, case_id)
    - start_interview(case_index) -> (InterviewState, interviewer_msg)
    - step(state, candidate_reply) -> (InterviewState, interviewer_msg 或 None)
    """

    def __init__(
        self,
        cases: List[CaseItem],
        interviewer_system_prompt: str = INTERVIEWER_SYSTEM_PROMPT,
        interviewer_model: str = INTERVIEWER_MODEL,
        max_turns: int = 10,
    ):
        self.cases = cases
        self.interviewer_system_prompt = interviewer_system_prompt
        self.interviewer_model = interviewer_model
        self.max_turns = max_turns

    # ---- 数据集接口 ----

    def __len__(self) -> int:
        return len(self.cases)

    def iter_cases(self):
        """遍历所有 case 的 (index, case_id)"""
        for i, c in enumerate(self.cases):
            yield i, c.case_id

    def get_case(self, index: int) -> CaseItem:
        return self.cases[index]

    # ---- 面试流程 ----

    def _build_interviewer_prompt(
        self,
        case: CaseItem,
        history: List[Dict[str, str]],
    ) -> str:
        transcript = history_to_transcript(history)
        return (
            "You are the interviewer in this case interview.\n\n"
            "CASE TEXT:\n"
            f"{case.case_text}\n\n"
            "CHAT HISTORY:\n"
            f"{transcript if transcript else '[no previous dialogue]'}\n\n"
            "Now, as the interviewer, produce ONLY your next message to the candidate, "
            "following your system instructions. Do not include any explanations about what you are doing."
        )

    def start_interview(self, case_index: int) -> Tuple[InterviewState, str]:
        """开始某一道 case 的面试"""
        case = self.cases[case_index]
        history: List[Dict[str, str]] = []

        user_prompt = self._build_interviewer_prompt(case, history)
        interviewer_msg = call_interviewer_llm(
            system_prompt=self.interviewer_system_prompt,
            user_content=user_prompt,
            model=self.interviewer_model,
        )
        history.append({"speaker": "interviewer", "text": interviewer_msg})

        state = InterviewState(
            case_id=case.case_id,
            case_index=case_index,
            case_text=case.case_text,
            turns=1,
            done=False,
            history=history,
        )

        if INTERVIEWER_END_TOKEN in interviewer_msg or state.turns >= self.max_turns:
            state.done = True

        return state, interviewer_msg

    def step(
        self,
        state: InterviewState,
        candidate_reply: str,
    ) -> Tuple[InterviewState, Optional[str]]:
        """
        单步对话：
          输入: 当前 state + 候选人回答
          输出: 更新后的 state + 下一句 interviewer_msg（结束后为 None）
        """
        if state.done:
            return state, None

        case = self.cases[state.case_index]

        # 记录候选人回答
        state.history.append({"speaker": "candidate", "text": candidate_reply})

        # 检查轮数
        if state.turns >= self.max_turns:
            state.done = True
            return state, None

        # 面试官下一句
        user_prompt = self._build_interviewer_prompt(case, state.history)
        interviewer_msg = call_interviewer_llm(
            system_prompt=self.interviewer_system_prompt,
            user_content=user_prompt,
            model=self.interviewer_model,
        )
        state.history.append({"speaker": "interviewer", "text": interviewer_msg})
        state.turns += 1

        if INTERVIEWER_END_TOKEN in interviewer_msg or state.turns >= self.max_turns:
            state.done = True

        return state, interviewer_msg


def evaluate_interview_with_llm(state: InterviewState) -> Dict:
    """
    使用 LLM Judge 对单个 InterviewState 进行评价打分。

    返回结构示例：
    {
      "case_id": "...",
      "case_index": 0,
      "scores": {...},
      "comment": "...",
    }
    """
    result = call_judge_llm(
        case_text=state.case_text,
        history=state.history,
    )
    # 附加 case 信息
    result_with_meta = {
        "case_id": state.case_id,
        "case_index": state.case_index,
        "scores": result.get("scores", {}),
        "comment": result.get("comment", ""),
    }
    return result_with_meta
