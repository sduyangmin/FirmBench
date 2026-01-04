from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from utils.llm import timed_llm_call
from utils.playbook_utils import extract_json_from_text
from utils.tools import initialize_clients


@dataclass(frozen=True)
class AoaRouterConfig:
    api_provider: str
    model: str
    max_tokens: int = 1024
    temperature: float = 0.0


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def route_one(
    experience_json: Dict[str, Any],
    sample: Dict[str, Any],
    cfg: AoaRouterConfig,
    prompt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    让 LLM 根据 experience（从历史结果提炼）为当前 sample 选择 agent。
    约定 sample 可包含：task_name/capability/difficulty_bucket/question/context 等。
    """
    prompt_path = prompt_path or (Path(__file__).parent / "prompts" / "router.md")
    system_prompt = _read_text(prompt_path)
    user_input = json.dumps(
        {"experience_json": experience_json, "sample": sample},
        ensure_ascii=False,
    )

    client, _, _ = initialize_clients(cfg.api_provider)
    raw, _info = timed_llm_call(
        client=client,
        api_provider=cfg.api_provider,
        model=cfg.model,
        prompt=user_input,  # 用于日志与异常记录
        role="aoa_router",
        call_id="aoa_router",
        max_tokens=cfg.max_tokens,
        log_dir=None,
        use_json_mode=True,
        temperature=cfg.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    out = extract_json_from_text(raw)
    if out is None:
        raise ValueError(f"AOA Router 输出无法解析为 JSON。原始输出前 800 字:\n{raw[:800]}")
    return out


