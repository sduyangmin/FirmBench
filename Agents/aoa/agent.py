from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.llm import timed_llm_call
from utils.playbook_utils import extract_json_from_text
from utils.tools import extract_answer, initialize_clients

# Underlying single-sample generators (all expose .generate(question, playbook, context, ...)->(resp, bullet_ids, call_info))
from Agents.cot.generator import ChainOfThoughtGenerator
from Agents.amem.generator import AMemBizBenchGenerator
from Agents.self_refine.generator import SelfRefineGenerator
from Agents.reflexion.generator import ReflexionGenerator
from Agents.debate.generator import DebateGenerator, DebateConfig
from Agents.discussion.generator import DiscussionGenerator, DiscussionConfig


@dataclass(frozen=True)
class AOAConfig:
    """
    AOA online router configuration.
    """

    router_model: str
    router_temperature: float = 0.0
    router_max_tokens: int = 512
    # ACE-style: only send a small, relevant subset of experience to router to avoid context blow-up.
    router_experience_compact: bool = True
    router_experience_max_bullets: int = 120
    router_experience_bullet_max_chars: int = 160
    router_experience_diagnosis_max_chars: int = 80
    candidates: Tuple[str, ...] = (
        "cot",
        "amem",
        "self_refine",
        "reflexion",
        "debate",
        "discussion",
        # "dynamic_cheatsheet",  # supported via flag, see below
    )
    include_dynamic_cheatsheet: bool = False


def _canon_agent_name(s: str) -> str:
    s = (s or "").strip()
    alias = {
        "self-refine": "self_refine",
        "dc": "dynamic_cheatsheet",
    }
    return alias.get(s, s)


def _extract_features(question: str, context: str) -> Dict[str, Any]:
    """
    Cheap, deterministic routing features (no leakage).
    """
    import re

    q = question or ""
    c = context or ""
    text = f"{q}\n{c}"
    nums = re.findall(r"-?\\d+(?:\\.\\d+)?", text)
    has_code = ("```" in text) or ("def " in text) or ("import " in text) or ("<code>" in text.lower())
    has_table = "|" in c and ("\n|" in c or "| " in c)
    # rough row estimate: count lines containing '|'
    table_rows = 0
    if has_table:
        table_rows = sum(1 for line in c.splitlines() if "|" in line)
    return {
        "question_chars": len(q),
        "context_chars": len(c),
        "num_numbers": len(nums),
        "has_code": bool(has_code),
        "has_table": bool(has_table),
        "table_row_estimate": table_rows,
    }

def _get_routing_policy_from_experience(exp: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(exp.get("routing_policy"), dict):
        return exp["routing_policy"]  # type: ignore[return-value]
    if isinstance(exp.get("meta_experience"), dict) and isinstance(exp["meta_experience"].get("routing_policy"), dict):
        return exp["meta_experience"]["routing_policy"]  # type: ignore[return-value]
    return {}

def _get_meta_findings_from_experience(exp: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(exp.get("meta_findings"), list):
        return [x for x in exp.get("meta_findings") if isinstance(x, dict)]  # type: ignore[list-item]
    if isinstance(exp.get("meta_experience"), dict) and isinstance(exp["meta_experience"].get("meta_findings"), list):
        return [x for x in exp["meta_experience"].get("meta_findings") if isinstance(x, dict)]  # type: ignore[list-item]
    return []

def _compact_text(s: Any, max_chars: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."

def _retrieve_bullets_for_sample(
    exp: Dict[str, Any],
    *,
    task_name: str,
    capability: str,
    difficulty_bucket: str,
    max_bullets: int,
    bullet_max_chars: int,
    diagnosis_max_chars: int,
) -> List[Dict[str, Any]]:
    if max_bullets <= 0:
        return []
    bullets = exp.get("experience_bullets")
    if not isinstance(bullets, list):
        return []

    def score(b: Dict[str, Any]) -> int:
        tags = b.get("tags") or {}
        if not isinstance(tags, dict):
            tags = {}
        s = 0
        if str(tags.get("task_name") or "").strip() == str(task_name or "").strip():
            s += 3
        if str(tags.get("capability") or "").strip() == str(capability or "").strip():
            s += 2
        db = str(tags.get("difficulty_bucket") or "").strip()
        if difficulty_bucket and db == difficulty_bucket:
            s += 1
        return s

    scored: List[tuple[int, int, Dict[str, Any]]] = []
    for i, b in enumerate(bullets):
        if isinstance(b, dict):
            scored.append((score(b), i, b))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    out: List[Dict[str, Any]] = []
    for s, _i, b in scored:
        if len(out) >= max_bullets:
            break
        tags = b.get("tags") or {}
        if not isinstance(tags, dict):
            tags = {}
        out.append(
            {
                "score": int(s),
                "bullet": _compact_text(b.get("bullet"), bullet_max_chars),
                "diagnosis": _compact_text(b.get("diagnosis"), diagnosis_max_chars),
                "outcome": b.get("outcome"),
                "tags": {
                    "agent_method": tags.get("agent_method"),
                    "task_name": tags.get("task_name"),
                    "capability": tags.get("capability"),
                    "difficulty_bucket": tags.get("difficulty_bucket"),
                },
            }
        )
    return out

def _compact_experience_for_router(
    exp: Dict[str, Any],
    *,
    task_name: str,
    capability: str,
    difficulty_bucket: str,
    max_bullets: int,
    bullet_max_chars: int,
    diagnosis_max_chars: int,
) -> Dict[str, Any]:
    policy = _get_routing_policy_from_experience(exp)
    findings = _get_meta_findings_from_experience(exp)
    retrieved = _retrieve_bullets_for_sample(
        exp,
        task_name=task_name,
        capability=capability,
        difficulty_bucket=difficulty_bucket,
        max_bullets=max_bullets,
        bullet_max_chars=bullet_max_chars,
        diagnosis_max_chars=diagnosis_max_chars,
    )
    out: Dict[str, Any] = {"routing_policy": policy, "meta_findings": findings}
    if retrieved:
        out["retrieved_bullets"] = retrieved
    return out


def _load_labels_jsonl(path: Optional[str], task_name: str) -> Dict[int, Dict[str, Any]]:
    """
    Load LLM reclassify classifications.jsonl and FILTER by task_name.

    Important:
    - The repo commonly uses a single classifications.jsonl that mixes multiple tasks.
    - Each line contains fields like: {task, index, parsed:{capability,difficulty_score,...}, ...}
    - StructuredReasoning runner evaluates one task at a time, so we filter by task and
      return index-aligned labels for that task only.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    task_name = str(task_name or "").strip()
    out: Dict[int, Dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if task_name and str(obj.get("task") or "").strip() != task_name:
                continue
            idx = obj.get("index")
            parsed = obj.get("parsed") or {}
            if idx is None:
                continue
            try:
                i = int(idx)
            except Exception:
                continue
            out[i] = {
                "capability": parsed.get("capability"),
                "difficulty_score": parsed.get("difficulty_score"),
                "reasoning": parsed.get("reasoning"),
            }
    return out


class AOARouter:
    def __init__(
        self,
        api_provider: str,
        model: str,
        max_tokens: int,
        temperature: float,
        experience_json: Dict[str, Any],
        experience_compact: bool = True,
        experience_max_bullets: int = 120,
        experience_bullet_max_chars: int = 160,
        experience_diagnosis_max_chars: int = 80,
    ):
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.experience_json = experience_json
        self.experience_compact = bool(experience_compact)
        self.experience_max_bullets = int(experience_max_bullets)
        self.experience_bullet_max_chars = int(experience_bullet_max_chars)
        self.experience_diagnosis_max_chars = int(experience_diagnosis_max_chars)
        self.client, _, _ = initialize_clients(api_provider)
        self.prompt_path = Path(__file__).parent / "prompts" / "router.md"
        self.system_prompt = self.prompt_path.read_text(encoding="utf-8")

    def choose(
        self,
        *,
        task_name: str,
        sample_index: int,
        question: str,
        context: str,
        labels: Dict[str, Any],
        features: Dict[str, Any],
        candidates: List[str],
        log_dir: Optional[str],
    ) -> Dict[str, Any]:
        exp_for_router = self.experience_json
        if self.experience_compact:
            exp_for_router = _compact_experience_for_router(
                self.experience_json,
                task_name=task_name,
                capability=str(labels.get("capability") or ""),
                difficulty_bucket=str(labels.get("difficulty_bucket") or "NA"),
                max_bullets=max(0, int(self.experience_max_bullets)),
                bullet_max_chars=int(self.experience_bullet_max_chars),
                diagnosis_max_chars=int(self.experience_diagnosis_max_chars),
            )
        payload = {
            "experience_json": exp_for_router,
            "candidates": candidates,
            "sample": {
                "task_name": task_name,
                "index": sample_index,
                "labels": labels,
                "features": features,
                "question": question,
                "context": context,
            },
        }
        user_input = json.dumps(payload, ensure_ascii=False)

        raw, info = timed_llm_call(
            client=self.client,
            api_provider=self.api_provider,
            model=self.model,
            prompt=user_input,
            role="aoa_router",
            call_id=f"aoa_router_{task_name}_{sample_index}",
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=True,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        obj = extract_json_from_text(raw) or {}
        if not isinstance(obj, dict):
            obj = {}
        obj["_raw"] = raw
        obj["_call_info"] = info
        return obj


class AOAAgent:
    """
    True online Agent-of-Agents:
    - For each sample: call router LLM (content + features + labels) -> choose underlying agent method
    - Delegate generation to that agent's single-sample generator
    - Evaluate correctness via the given DataProcessor

    Supports modes: online / eval_only (no training loop).
    """

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        agent_method: str = "aoa",
        aoa_config: Optional[AOAConfig] = None,
        experience_path: Optional[str] = None,
    ):
        self.agent_method = agent_method
        self.api_provider = api_provider
        self.generator_model = generator_model
        self.max_tokens = max_tokens
        self.cfg = aoa_config or AOAConfig(router_model=generator_model)

        exp_path = Path(experience_path) if experience_path else (Path("results") / "StructuredReasoning_run" / "aoa_mode" / "experience" / "experience.latest.json")
        if not exp_path.exists():
            raise FileNotFoundError(f"AOA experience_json not found: {exp_path}")
        self.experience_json = json.loads(exp_path.read_text(encoding="utf-8"))

        # Router uses its own model (often same as generator_model, but configurable)
        self.router = AOARouter(
            api_provider=api_provider,
            model=self.cfg.router_model,
            max_tokens=self.cfg.router_max_tokens,
            temperature=self.cfg.router_temperature,
            experience_json=self.experience_json,
            experience_compact=self.cfg.router_experience_compact,
            experience_max_bullets=self.cfg.router_experience_max_bullets,
            experience_bullet_max_chars=self.cfg.router_experience_bullet_max_chars,
            experience_diagnosis_max_chars=self.cfg.router_experience_diagnosis_max_chars,
        )

        # Underlying generators
        self._init_underlying()

    def _init_underlying(self) -> None:
        # shared client for AMem
        client, _, _ = initialize_clients(self.api_provider)
        self.generators: Dict[str, Any] = {
            "cot": ChainOfThoughtGenerator(self.api_provider, self.generator_model, self.max_tokens),
            "amem": AMemBizBenchGenerator(client=client, api_provider=self.api_provider, model=self.generator_model, max_tokens=self.max_tokens),
            "self_refine": SelfRefineGenerator(
                api_provider=self.api_provider,
                model_name=self.generator_model,
                max_tokens=self.max_tokens,
                refine_rounds=2,
                initial_temperature=0.0,
                feedback_temperature=0.2,
            ),
            "reflexion": ReflexionGenerator(
                api_provider=self.api_provider,
                model_name=self.generator_model,
                max_tokens=self.max_tokens,
                reflexion_rounds=2,
                initial_temperature=0.0,
                reflect_temperature=0.2,
            ),
            "debate": DebateGenerator(
                api_provider=self.api_provider,
                model_name=self.generator_model,
                max_tokens=self.max_tokens,
                debate_config=DebateConfig(rounds=1),
            ),
            "discussion": DiscussionGenerator(
                api_provider=self.api_provider,
                model_name=self.generator_model,
                max_tokens=self.max_tokens,
                discussion_config=DiscussionConfig(num_experts=3, rounds=1),
            ),
        }

        if self.cfg.include_dynamic_cheatsheet:
            from Agents.dynamic_cheatsheet.agent import DynamicCheatsheetAgent, DynamicCheatsheetConfig
            from Agents.dynamic_cheatsheet.core.state import CheatsheetState

            dc_agent = DynamicCheatsheetAgent(
                api_provider=self.api_provider,
                generator_model=self.generator_model,
                max_tokens=self.max_tokens,
                agent_method="dynamic_cheatsheet",
                dc_config=DynamicCheatsheetConfig(),
            )
            # Minimal state for online incremental usage
            dc_state = CheatsheetState(cheatsheet_text="(empty)")

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

            self.generators["dynamic_cheatsheet"] = {
                "agent": dc_agent,
                "state": dc_state,
                "strip_fn": strip_task_instruction,
            }

    def _prepare_dirs(self, task_name: str, mode: str, save_dir: str) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name, self.agent_method, mode, timestamp)
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)
        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        return {
            "timestamp": timestamp,
            "run_subdir": run_subdir,
            "resolved_save_path": resolved_save_path,
            "log_dir": log_dir,
        }

    def _call_underlying(
        self,
        method: str,
        sample: Dict[str, Any],
        idx: int,
        data_processor,
        log_dir: str,
        task_name: str,
        use_json_mode: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        method = _canon_agent_name(method)
        question = sample.get("question", "")
        context = sample.get("context", "") or ""

        if method == "dynamic_cheatsheet":
            bundle = self.generators.get("dynamic_cheatsheet")
            if not bundle:
                raise ValueError("dynamic_cheatsheet not enabled in AOAConfig.")
            dc_agent = bundle["agent"]
            state = bundle["state"]
            strip_fn = bundle["strip_fn"]
            # mimic dynamic_cheatsheet._process_sample minimal paths
            paths = {"log_dir": log_dir, "cheatsheet_history_path": os.path.join(log_dir, "dc_history.jsonl")}
            # ensure prompts are selected
            dc_agent.generator_prompt_for_run, dc_agent.cheatsheet_prompt_for_run = dc_agent._select_prompts_for_task(
                task_name.lower(), dc_agent.generator_prompt, dc_agent.cheatsheet_prompt
            )
            out = dc_agent._process_sample(
                sample=sample,
                idx=idx,
                state=state,
                data_processor=data_processor,
                paths=paths,
                strip_task_instruction_fn=strip_fn,
                update_state=True,
                log_history=True,
                verbose=False,
                call_prefix=f"aoa_dc_{idx}",
            )
            # dynamic_cheatsheet already gives final_answer
            resp = json.dumps({"reasoning": "(dynamic_cheatsheet)", "final_answer": out.get("final_answer")}, ensure_ascii=False)
            return resp, {"dc": out}

        gen = self.generators.get(method)
        if gen is None:
            raise ValueError(f"Unknown/unsupported underlying agent: {method}")

        resp, _bullet_ids, call_info = gen.generate(
            question=question,
            playbook="",
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"aoa_{method}_{idx}",
            log_dir=log_dir,
        )
        return resp, call_info

    def run(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        _ = (train_samples, val_samples)
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"{self.agent_method} only supports {self.SUPPORTED_MODES}, got {mode}")

        task_name = config.get("task_name", "unknown_task")
        save_dir = config.get("save_dir")
        if not save_dir:
            raise ValueError("Configuration missing 'save_dir'.")

        paths = self._prepare_dirs(task_name, mode, save_dir)
        resolved_save_path = paths["resolved_save_path"]
        log_dir = paths["log_dir"]

        # Optional: labels jsonl path (capability/difficulty), to help router
        labels_jsonl_path = config.get("aoa_labels_jsonl", "") or ""
        labels_by_idx = _load_labels_jsonl(labels_jsonl_path, task_name=task_name)

        # Candidate set
        candidates = [_canon_agent_name(x) for x in self.cfg.candidates]
        if self.cfg.include_dynamic_cheatsheet and "dynamic_cheatsheet" not in candidates:
            candidates.append("dynamic_cheatsheet")

        use_json_mode = bool(config.get("json_mode", False))

        # Evaluate sequentially (router + delegate are costly; keep it deterministic)
        correct = 0
        total = 0
        errors: List[Dict[str, Any]] = []
        sample_logs: List[Dict[str, Any]] = []

        for i, sample in enumerate(test_samples):
            question = sample.get("question", "")
            context = sample.get("context", "") or ""
            target = sample.get("target")
            feats = _extract_features(question, context)
            labels = labels_by_idx.get(i, {})

            router_out = self.router.choose(
                task_name=task_name,
                sample_index=i,
                question=question,
                context=context,
                labels=labels,
                features=feats,
                candidates=candidates,
                log_dir=log_dir,
            )
            chosen = _canon_agent_name(str(router_out.get("chosen_agent") or router_out.get("agent") or ""))
            if chosen not in candidates:
                # fallback: if router outputs unknown agent, use first candidate
                chosen = candidates[0]

            try:
                resp, call_info = self._call_underlying(
                    method=chosen,
                    sample=sample,
                    idx=i,
                    data_processor=data_processor,
                    log_dir=log_dir,
                    task_name=task_name,
                    use_json_mode=use_json_mode,
                )
                final_answer = extract_answer(resp)
                is_correct = data_processor.answer_is_correct(final_answer, target)
            except Exception as e:
                final_answer = "ERROR"
                is_correct = False
                call_info = {"error": f"{type(e).__name__}: {str(e)}"}

            total += 1
            correct += (1 if is_correct else 0)
            if not is_correct:
                errors.append(
                    {
                        "index": i,
                        "chosen_agent": chosen,
                        "final_answer": final_answer,
                        "target": target,
                    }
                )

            sample_logs.append(
                {
                    "index": i,
                    "chosen_agent": chosen,
                    "router": {
                        "chosen_agent": router_out.get("chosen_agent"),
                        "reason": router_out.get("reason"),
                        "used_rule": router_out.get("used_rule"),
                        "confidence": router_out.get("confidence"),
                    },
                    "labels": labels,
                    "features": feats,
                    "final_answer": final_answer,
                    "target": target,
                    "is_correct": bool(is_correct),
                }
            )

        acc = (correct / total) if total else 0.0
        test_results = {
            "correct": correct,
            "total": total,
            "accuracy": acc,
            "errors": errors,
        }
        payload = {
            "test_results": test_results,
            "error_log": {"errors": errors, "accuracy": acc},
            "aoa": {
                "experience_path": str(config.get("aoa_experience_path") or ""),
                "labels_jsonl": labels_jsonl_path,
                "candidates": candidates,
            },
            "per_sample": sample_logs,
        }

        # Save config + results
        with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({"config": config, "run_subdir": paths["run_subdir"]}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[AOA] Finished {task_name} mode={mode} acc={acc:.4f} (N={total}) saved={resolved_save_path}")
        return test_results


