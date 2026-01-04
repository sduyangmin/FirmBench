from __future__ import annotations

import json
import math
import random
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import numpy as np

from utils.llm import timed_llm_call
from utils.tools import extract_answer
from StructuredReasoning.data_processor import DataProcessor

from .config import GepaConfig
from .prompts import (
    INITIAL_CANDIDATES_GENERATION_PROMPT,
    MERGING_PROMPT,
    META_PROMPT,
    SEED_PROMPT,
)


def _parse_candidate_list(text: str) -> List[str]:
    """解析 JSON 结构的候选 prompt 列表，退化为按行拆分。"""
    if not text:
        return []

    cleaned = text.strip()
    # 去掉 markdown fenced code
    if cleaned.startswith("```"):
        parts = cleaned.splitlines()
        if parts:
            parts = parts[1:]
        if parts and parts[-1].startswith("```"):
            parts = parts[:-1]
        cleaned = "\n".join(parts).strip()

    # 优先 JSON
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "list_candidate_prompts" in data:
            lst = data["list_candidate_prompts"]
            if isinstance(lst, list):
                return [
                    (item.get("candidate_prompt", "") if isinstance(item, dict) else "").strip()
                    for item in lst
                    if isinstance(item, dict) and item.get("candidate_prompt")
                ]
    except Exception:
        pass

    # 退化：按行拆
    candidates = [line.strip("- \t") for line in cleaned.splitlines() if line.strip()]
    return candidates


def _eval_metric(pred: Any, gt: Any) -> float:
    """Retained for potential fallback; not used when accuracy-based eval is enabled."""
    pred_str = "" if pred is None else str(pred)
    gt_str = "" if gt is None else str(gt)
    return round(SequenceMatcher(None, pred_str, gt_str).ratio(), 3)


def _pick_prediction(sample: Dict[str, Any], resp: Any, task_name: str, data_processor=None) -> str:
    """
    对代码补全类任务优先提取 code block；否则回退 extract_answer。
    """
    if task_name == "FormulaEval":
        try:
            if data_processor and hasattr(data_processor, "_extract_code_block"):
                code_block = data_processor._extract_code_block(str(resp))
            else:
                processor = DataProcessor(task_name)
                code_block = processor._extract_code_block(str(resp))
            if code_block:
                return code_block
        except Exception:
            pass
    pred = extract_answer(resp)
    if pred == "No final answer found":
        pred = resp
    return pred


def _build_query(sample: Dict[str, Any], prompt: str) -> str:
    context = sample.get("context", "")
    question = sample.get("question", "")
    query = question if not context else f"{context}\n\n{question}"
    return f"{prompt}\n\nQuery: {query}\nAnswer:"


def generate_initial_candidates(
    reflection_client,
    api_provider: str,
    reflection_model: str,
    cfg: GepaConfig,
    Dpareto: List[Dict[str, str]],
) -> List[str]:
    inputs_parts = []
    for sample in Dpareto:
        xi, gt = sample.get("question", ""), sample.get("target", "")
        inputs_parts.append(f"Query: {xi}\nAnswer: {gt}")
    inputs = "\n".join(inputs_parts) + "\n"

    prompt_tpl = INITIAL_CANDIDATES_GENERATION_PROMPT
    seed_prompt = cfg.seed_prompt or SEED_PROMPT

    def _one(idx: int) -> str:
        call_id = f"gepa_init_{idx}"
        rendered = prompt_tpl.format(
            seed_prompt=seed_prompt,
            inputs=inputs,
            num_new_prompts=1,
        )
        resp, _ = timed_llm_call(
            reflection_client,
            api_provider=api_provider,
            model=reflection_model,
            prompt=rendered,
            role="reflector",
            call_id=call_id,
            max_tokens=cfg.max_tokens,
            use_json_mode=cfg.use_json_mode,
            temperature=cfg.reflection_temperature,
        )
        candidates = _parse_candidate_list(resp)
        return candidates[0] if candidates else ""

    with ThreadPoolExecutor(max_workers=min(cfg.max_workers, cfg.num_initial)) as ex:
        futures = {ex.submit(_one, i): i for i in range(cfg.num_initial)}
        results = []
        for fut in as_completed(futures):
            cand = fut.result()
            if cand and cand not in results:
                results.append(cand)
    return results


def select_candidate(P: List[Dict[str, Any]], S: np.ndarray) -> Tuple[List[int], List[float]]:
    """与原实现等价：Pareto 过滤 + 频次分布。"""
    num_tasks = S.shape[1]
    num_candidates = S.shape[0]

    if num_candidates == 1:
        return [0], [1.0]

    s_star = np.max(S, axis=0)
    P_star = [set() for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_candidates):
            if S[j, i] == s_star[i]:
                P_star[i].add(j)

    C = set().union(*P_star)
    D = set()
    C_list = list(C)
    while True:
        dominated_found = False
        for idx1 in range(len(C_list)):
            if C_list[idx1] in D:
                continue
            is_dominated = False
            for idx2 in range(len(C_list)):
                if idx1 == idx2 or C_list[idx2] in D:
                    continue
                all_leq = all(S[C_list[idx1]][k] <= S[C_list[idx2]][k] for k in range(num_tasks))
                any_lt = any(S[C_list[idx1]][k] < S[C_list[idx2]][k] for k in range(num_tasks))
                if all_leq and any_lt:
                    is_dominated = True
                    break
            if is_dominated:
                D.add(C_list[idx1])
                dominated_found = True
        if not dominated_found:
            break

    hat_P_star = [p_set - D for p_set in P_star]
    freq = {}
    for p_set in hat_P_star:
        for k in p_set:
            freq[k] = freq.get(k, 0) + 1
    hat_C = list(set(k for p_set in hat_P_star for k in p_set))
    if not hat_C:
        raise ValueError("No candidates after Pareto filtering.")
    probs = [freq.get(k, 0) for k in hat_C]
    total = sum(probs)
    probs = [p / total for p in probs]
    return hat_C, probs


@dataclass
class GepaResult:
    best_prompt: str
    best_candidate: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]


def run_gepa(
    generator_client,
    reflection_client,
    api_provider: str,
    target_model: str,
    reflection_model: str,
    cfg: GepaConfig,
    Dpareto: List[Dict[str, Any]],
    Dfeedback: List[Dict[str, Any]],
    initial_candidates: List[str] | None = None,
    use_accuracy: bool = False,
    data_processor=None,
    debug: bool = False,
) -> GepaResult:
    if not Dpareto:
        raise ValueError("Dpareto 为空")
    if not Dfeedback:
        raise ValueError("Dfeedback 为空")

    if initial_candidates is not None:
        candidates_list = [c for c in initial_candidates if c]
    else:
        candidates_list = generate_initial_candidates(
            reflection_client, api_provider, reflection_model, cfg, Dpareto
        )
        if not candidates_list:
            raise ValueError("初始候选生成失败")

    candidates: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates_list):
        candidates.append(
            {"id": idx, "parent_id": -1, "prompt": cand, "scores": [0.0] * len(Dpareto), "mean_score": 0.0}
        )
    S = np.zeros((len(candidates), len(Dpareto)), dtype=float)

    # 评估初始候选
    def _log_debug(prefix: str, resp: Any, pred: Any, gt: Any) -> None:
        if not debug:
            return

        def _trunc(x):
            s = str(x)
            return s if len(s) <= 400 else s[:400] + "...(truncated)"

        print(f"[GEPA][debug] {prefix} resp={_trunc(resp)} pred={_trunc(pred)} gt={_trunc(gt)}")

    def _eval_candidate(j: int, candidate: Dict[str, Any]) -> None:
        local_scores = []
        prompt = candidate["prompt"]
        for h, sample in enumerate(Dpareto):
            contents = _build_query(sample, prompt)
            resp, _ = timed_llm_call(
                generator_client,
                api_provider=api_provider,
                model=target_model,
                prompt=contents,
                role="generator",
                call_id=f"gepa_init_eval_{j}_{h}",
                max_tokens=cfg.max_tokens,
                use_json_mode=False,
                temperature=cfg.target_temperature,
            )
            pred = _pick_prediction(sample, resp, data_processor.task_name if data_processor else "", data_processor)
            if use_accuracy and data_processor is not None:
                score = 1.0 if data_processor.answer_is_correct(pred, sample.get("target", "")) else 0.0
            else:
                score = _eval_metric(pred, sample.get("target", ""))
            if debug and use_accuracy and score == 0.0 and h < 2:
                _log_debug(f"init_eval j={j} h={h}", resp, pred, sample.get("target", ""))
            local_scores.append(score)
        candidate["scores"] = local_scores
        candidate["mean_score"] = float(np.mean(local_scores))
        S[j, :] = np.array(local_scores, dtype=float)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        list(ex.map(lambda pair: _eval_candidate(*pair), enumerate(candidates)))

    budget_left = cfg.budget - len(candidates) * len(Dpareto)
    trace: List[Dict[str, Any]] = []

    # 主循环
    while budget_left > 0:
        mini_size = min(cfg.mini_batch_size, len(Dfeedback))
        batch_indices = random.sample(range(len(Dfeedback)), mini_size)
        Dmini = [Dfeedback[i] for i in batch_indices]

        # 选择策略
        if random.random() <= cfg.exploit_prob:
            selected_ids, probs = select_candidate(candidates, S)
            if len(selected_ids) > 1:
                if random.random() <= cfg.merge_prob:
                    strategy = "exploit-normal"
                    selected_id = random.choices(selected_ids, weights=probs, k=1)[0]
                    selected = candidates[selected_id]
                else:
                    strategy = "exploit-merge"
                    merging_text = ""
                    parents = []
                    for cid in selected_ids:
                        cand = candidates[cid]
                        parents.append(cand["id"])
                        merging_text += f"Candidate {cand['id']} >> {cand['prompt']}\n"
                    merge_prompt = MERGING_PROMPT.format(candidates=merging_text)
                    resp, _ = timed_llm_call(
                        reflection_client,
                        api_provider=api_provider,
                        model=reflection_model,
                        prompt=merge_prompt,
                        role="reflector",
                        call_id="gepa_merge",
                        max_tokens=cfg.max_tokens,
                        use_json_mode=cfg.use_json_mode,
                        temperature=cfg.reflection_temperature,
                    )
                    merged_candidates = _parse_candidate_list(resp)
                    merged_prompt = merged_candidates[0] if merged_candidates else ""
                    selected = {
                        "id": len(candidates),
                        "parent_id": tuple(parents),
                        "prompt": merged_prompt,
                        "scores": [0.0] * len(Dpareto),
                        "mean_score": 0.0,
                    }
                    candidates.append(selected)
                    # 扩展 S
                    S = np.vstack([S, np.zeros((1, len(Dpareto)))])
            else:
                strategy = "exploit-max"
                selected = candidates[selected_ids[0]]
        else:
            strategy = "explore"
            selected = random.choice(candidates)

        # 在 mini-batch 上评估 selected
        def _eval_sample(sample: Dict[str, Any], call_idx: int) -> Tuple[float, str]:
            contents = _build_query(sample, selected["prompt"])
            resp, _ = timed_llm_call(
                generator_client,
                api_provider=api_provider,
                model=target_model,
                prompt=contents,
                role="generator",
                call_id=f"gepa_fb_{selected['id']}_{call_idx}",
                max_tokens=cfg.max_tokens,
                use_json_mode=False,
                temperature=cfg.target_temperature,
            )
            pred = _pick_prediction(sample, resp, data_processor.task_name if data_processor else "", data_processor)
            if use_accuracy and data_processor is not None:
                score = 1.0 if data_processor.answer_is_correct(pred, sample.get("target", "")) else 0.0
            else:
                score = _eval_metric(pred, sample.get("target", ""))
            if debug and use_accuracy and score == 0.0 and call_idx < 2:
                _log_debug(f"mini_eval sel={selected['id']} idx={call_idx}", resp, pred, sample.get("target", ""))
            return score, pred

        mini_scores = []
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futures = {ex.submit(_eval_sample, sample, i): i for i, sample in enumerate(Dmini)}
            for fut in as_completed(futures):
                s, _ = fut.result()
                mini_scores.append(s)

        budget_left -= len(mini_scores)
        mean_feedback = float(np.mean(mini_scores)) if mini_scores else 0.0

        # 反思生成新 prompt
        overall_feedback = "\n".join(
            f"Query: {_build_query(sample, '')}\nScore: {score}"
            for sample, score in zip(Dmini, mini_scores)
        )
        meta_contents = META_PROMPT.format(
            candidate=selected["prompt"],
            inputs_outputs_feedback=overall_feedback,
        )
        resp, _ = timed_llm_call(
            reflection_client,
            api_provider=api_provider,
            model=reflection_model,
            prompt=meta_contents,
            role="reflector",
            call_id=f"gepa_reflect_{selected['id']}",
            max_tokens=cfg.max_tokens,
            use_json_mode=cfg.use_json_mode,
            temperature=cfg.reflection_temperature,
        )
        new_candidates = _parse_candidate_list(resp)
        if not new_candidates:
            trace.append(
                {
                    "step": len(trace),
                    "selected": selected["id"],
                    "strategy": strategy,
                    "accepted": False,
                    "reason": "no_reflection_candidate",
                }
            )
            continue
        new_prompt = new_candidates[0]

        # 再在 mini-batch 评估新 prompt
        def _eval_new(sample: Dict[str, Any], call_idx: int) -> float:
            contents = _build_query(sample, new_prompt)
            resp, _ = timed_llm_call(
                generator_client,
                api_provider=api_provider,
                model=target_model,
                prompt=contents,
                role="generator",
                call_id=f"gepa_new_{len(candidates)}_{call_idx}",
                max_tokens=cfg.max_tokens,
                use_json_mode=False,
                temperature=cfg.target_temperature,
            )
            pred = _pick_prediction(sample, resp, data_processor.task_name if data_processor else "", data_processor)
            if use_accuracy and data_processor is not None:
                score = 1.0 if data_processor.answer_is_correct(pred, sample.get("target", "")) else 0.0
            else:
                score = _eval_metric(pred, sample.get("target", ""))
            if debug and use_accuracy and score == 0.0 and call_idx < 2:
                _log_debug(f"new_eval cand={len(candidates)} idx={call_idx}", resp, pred, sample.get("target", ""))
            return score

        new_scores = []
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futures = {ex.submit(_eval_new, sample, i): i for i, sample in enumerate(Dmini)}
            for fut in as_completed(futures):
                new_scores.append(fut.result())

        budget_left -= len(new_scores)
        mean_new = float(np.mean(new_scores)) if new_scores else 0.0

        accepted = mean_new >= mean_feedback
        if accepted and budget_left > 0:
            # 全量 Dpareto 评估
            pareto_scores = []
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
                futures = {
                    ex.submit(_eval_new, sample, i + 10000): i for i, sample in enumerate(Dpareto)
                }
                for fut in as_completed(futures):
                    pareto_scores.append(fut.result())
            budget_left -= len(pareto_scores)
            mean_pareto = float(np.mean(pareto_scores)) if pareto_scores else 0.0
            new_cand = {
                "id": len(candidates),
                "parent_id": selected["id"],
                "prompt": new_prompt,
                "scores": pareto_scores,
                "mean_score": mean_pareto,
            }
            candidates.append(new_cand)
            S = np.vstack([S, np.array(pareto_scores, dtype=float)])
            trace.append(
                {
                    "step": len(trace),
                    "selected": selected["id"],
                    "strategy": strategy,
                    "accepted": True,
                    "candidate_id": new_cand["id"],
                    "parent_id": new_cand["parent_id"],
                    "mean_feedback": mean_feedback,
                    "mean_new": mean_new,
                    "mean_pareto": mean_pareto,
                }
            )
        else:
            trace.append(
                {
                    "step": len(trace),
                    "selected": selected["id"],
                    "strategy": strategy,
                    "accepted": False,
                    "mean_feedback": mean_feedback,
                    "mean_new": mean_new,
                }
            )

        if budget_left <= 0:
            break

    best_idx = int(np.argmax([c["mean_score"] for c in candidates]))
    best_candidate = candidates[best_idx]
    return GepaResult(
        best_prompt=best_candidate["prompt"],
        best_candidate=best_candidate,
        candidates=candidates,
        trace=trace,
    )



