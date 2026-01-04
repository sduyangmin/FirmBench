"""
ACE (Agent-Curator-Environment) System
Main orchestrator class for training and testing with playbook-based learning.

This module coordinates three agents:
- Generator: Produces answers using playbook knowledge
- Reflector: Analyzes outputs and tags bullets
- Curator: Updates the playbook based on feedback
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from utils.consulting_tools import evaluate_consulting_set
from utils.seriousgame_tools import (
    beergame_prepare_run,
    beergame_evaluate_run,
    beergame_save_run,
    beergame_base_rule_order,
    beergame_render_prompt,
    beergame_extract_order_and_note,
    edt_prepare_run,
    edt_evaluate_run,
    edt_save_run,
    render_edt_prompt,
    build_edt_decision_context,
    normalize_edt_schema,
)

from .core import Generator, Reflector, Curator, BulletpointAnalyzer
from utils.playbook_utils import *
from utils.logger import *
from utils.tools import *


class ACE:
    """
    Main ACE system orchestrator.
    
    Manages the training loop where:
    1. Generator produces answers using playbook
    2. Reflector analyzes answers and tags bullets
    3. Curator updates playbook based on feedback
    
    """
    
    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        max_tokens: int = 4096,
        initial_playbook: Optional[str] = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90,
        agent_method: str = "ace",
    ):
        """
        Initialize the ACE system.
        
        Args:
            api_provider: API provider for LLM calls
            generator_model: Model name for generator
            reflector_model: Model name for reflector
            curator_model: Model name for curator
            max_tokens: Maximum tokens for LLM calls
            initial_playbook: Initial playbook content (optional)
            use_bulletpoint_analyzer: Whether to use bulletpoint analyzer for deduplication
            bulletpoint_analyzer_threshold: Similarity threshold for bulletpoint analyzer (0-1)
        """
        # Initialize API clients
        generator_client, reflector_client, curator_client = initialize_clients(api_provider)

        # Initialize the three agents
        self.generator = Generator(generator_client, api_provider, generator_model, max_tokens)
        self.reflector = Reflector(reflector_client, api_provider, reflector_model, max_tokens)
        self.curator = Curator(curator_client, api_provider, curator_model, max_tokens)
        
        # Initialize bulletpoint analyzer if requested and available
        self.use_bulletpoint_analyzer = use_bulletpoint_analyzer
        self.bulletpoint_analyzer_threshold = bulletpoint_analyzer_threshold
        
        if use_bulletpoint_analyzer:
            self.bulletpoint_analyzer = BulletpointAnalyzer(
                curator_client, 
                curator_model, 
                max_tokens
            )
            print(f"✓ BulletpointAnalyzer initialized (threshold={bulletpoint_analyzer_threshold})")
        else:
            self.bulletpoint_analyzer = None
        
        # Store configuration
        self.generator_client = generator_client
        self.reflector_client = reflector_client
        self.curator_client = curator_client
        self.max_tokens = max_tokens
        self.temperature = 0.7
        self.max_order_qty = 5000
        self.SUPPORTED_MODES = {"online", "eval_only"}

        # Consulting-specific runtime state
        self.consulting_mode: Optional[str] = None
        self._current_case_id: Optional[str] = None
        self.consulting_cases_since_curate: int = 0
        # how many cases between curator updates (will be configurable later)
        self.consulting_curator_frequency: int = 1
        # buffer of recent reflections (to be used in later steps)
        self.consulting_pending_reflections: List[str] = []
        # log dir for consulting LLM calls
        self._consulting_log_dir: Optional[str] = None
        # cached config for consulting reflection/curation
        self._consulting_token_budget: int = 80000
        self._consulting_total_samples: int = 0
        
        # Initialize playbook
        self.agent_method = agent_method
        if initial_playbook:
            self.playbook = initial_playbook
        else:
            self.playbook = self._initialize_empty_playbook()
        
        self.best_playbook = self.playbook
        # Track global bullet ID
        self.next_global_id = 1
        # EDT-specific online learning bookkeeping
        self._edt_episode_counter: int = 0         # 当前已经见过多少个 EDT episode
        self._edt_total_episodes: int = 0          # 总的 episode 数（在 run_edt 里设置）
        self._edt_config_params: Dict[str, Any] = {}  # 复用 _extract_config_params 的结果，用于 curator

    # ==========================================================
    # Consulting support (lightweight, no playbook updates)
    # ==========================================================

    def _call_llm_json(self, system: str, user: str) -> Dict[str, Any]:
        """
        Lightweight JSON helper for consulting chat turns.
        """
        import json as _json

        resp = self.generator_client.chat.completions.create(
            model=self.generator.model,
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
        """Hook called at the beginning of each consulting case."""
        self._current_case_id = case_id

    def on_case_end(
            self,
            case_id: str,
            case_text: str,
            history: List[Dict[str, str]],
    ) -> None:
        """Hook called at the end of each consulting case.

        In online mode this triggers reflection + (periodic) curation so that
        the consulting playbook can evolve over cases.
        """
        self._current_case_id = None

        # Only update playbook in online mode
        if self.consulting_mode != "online":
            return

        log_dir = getattr(self, "_consulting_log_dir", None) or "."

        # ----- 构造给 Reflector 的输入 -----
        # 1) 用 case_text 作为 question（可认为是 case 描述）
        question_for_reflection = case_text or f"Consulting case {case_id}"

        # 2) 整个对话作为 reasoning_trace
        transcript_lines = [
            f"{h.get('role', 'unknown')}: {h.get('content', '')}"
            for h in history
        ]
        reasoning_trace = "\n".join(transcript_lines) or "[no dialogue]"

        # 3) 找最后一条 candidate/assistant 消息，作为 predicted_answer
        predicted_answer = ""
        for h in reversed(history):
            role = h.get("role")
            if role in ("candidate", "assistant"):
                predicted_answer = h.get("content", "")
                break
        if not predicted_answer:
            predicted_answer = "[no candidate answer captured]"

        environment_feedback = (
            "Consulting case finished. Extract better structures, heuristics, "
            "and common mistakes for future cases."
        )

        # 对于 consulting，目前不追踪 per-turn bullet usage
        playbook_bullets: List[str] = []

        # ----- 调 Reflector 生成反思内容 -----
        reflection_content, bullet_tags, _ = self.reflector.reflect(
            question=question_for_reflection,
            reasoning_trace=reasoning_trace,
            predicted_answer=predicted_answer,
            ground_truth=None,
            environment_feedback=environment_feedback,
            bullets_used=playbook_bullets,
            use_ground_truth=False,
            use_json_mode=False,
            call_id=f"consult_case_{case_id}_reflect",
            log_dir=log_dir,
        )

        if reflection_content and isinstance(reflection_content, str):
            self.consulting_pending_reflections.append(reflection_content)

        # 更新 bullet 使用计数（和 StructuredReasoning 一致）
        if bullet_tags:
            self.playbook = update_bullet_counts(self.playbook, bullet_tags)

        self.consulting_cases_since_curate += 1

        # ----- 根据频率调用 Curator -----
        freq = max(int(getattr(self, "consulting_curator_frequency", 1)), 1)
        if self.consulting_cases_since_curate >= freq:
            merged_reflection = "\n\n".join(self.consulting_pending_reflections)
            stats = get_playbook_stats(self.playbook)

            token_budget = getattr(self, "_consulting_token_budget", 80000)
            total_samples = getattr(
                self, "_consulting_total_samples", self.consulting_cases_since_curate
            )

            self.playbook, self.next_global_id, operations, _ = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=merged_reflection,
                question_context=case_text or "",
                current_step=self.consulting_cases_since_curate,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                use_ground_truth=False,
                use_json_mode=False,
                call_id=f"consult_curate_case_{case_id}",
                log_dir=log_dir,
                next_global_id=self.next_global_id,
            )

            # 可选：再跑一下 BulletpointAnalyzer
            if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
                print(
                    f"  Running BulletpointAnalyzer "
                    f"(threshold={self.bulletpoint_analyzer_threshold})..."
                )
                self.playbook = self.bulletpoint_analyzer.analyze(
                    playbook=self.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True,
                )

            # 重置 case 计数和缓存
            self.consulting_cases_since_curate = 0
            self.consulting_pending_reflections = []

    def _render_consulting_candidate_prompt(
        self,
        case_id: str,
        history: List[Dict[str, str]],
    ) -> Tuple[str, str]:
        """Build (question, context) for Generator in consulting.

        question: high-level task description + latest interviewer question
        context: full dialogue transcript so far
        """
        # Last interviewer message
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

        question_parts = [
            f"[CONSULTING CASE] case_id={case_id}",
            "",
            "You are the CANDIDATE in a consulting-style case interview.",
            "Act like a top-tier consulting candidate: structured, hypothesis-driven,",
            "quantitative when possible, and clearly communicate your thinking.",
            "",
            "The interviewer has just said:",
            last_interviewer_msg or "[no interviewer message found]",
            "",
            "Respond with what you would say next as the candidate in this dialogue.",
        ]
        question = "\n".join(question_parts)

        context = "Dialogue so far:\n" + transcript_text
        return question, context

    def reply(self, case_id: str, history: List[Dict[str, str]]) -> str:
        """Consulting turn: generate next candidate reply using full Generator.

        This version conditions on the current playbook and uses the same
        Generator pipeline as StructuredReasoning, so consulting can
        benefit from accumulated playbook knowledge. Adds a lightweight
        self-reflection pass (no ground truth) to improve the answer.
        """
        # Build question/context for Generator
        question, context = self._render_consulting_candidate_prompt(case_id, history)

        # Decide log dir for this call
        log_dir = getattr(self, "_consulting_log_dir", None) or "."

        # 初答：不启用 JSON mode，只要自然语言回答
        gen_response, bullet_ids, call_info = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=False,
            call_id=f"consult_case_{case_id}_turn_{len(history)}",
            log_dir=log_dir,
        )

        # 轻量自评：无 ground truth，使用 reflector 做自我批改
        reflection_content = ""
        try:
            question_for_reflection = self._strip_task_instruction(question)
            reflection_content, _, _ = self.reflector.reflect(
                question=question_for_reflection,
                reasoning_trace=gen_response,
                predicted_answer=gen_response,
                ground_truth=None,
                environment_feedback="Check correctness, structure, and quantitative rigor.",
                bullets_used="(none)",
                use_ground_truth=False,
                use_json_mode=False,
                call_id=f"consult_case_{case_id}_turn_{len(history)}_reflect",
                log_dir=log_dir,
            )
        except Exception:
            reflection_content = ""

        # 若有反思内容，按反思重写一遍
        final_response = gen_response
        if reflection_content:
            try:
                final_response, _, _ = self.generator.generate(
                    question=question,
                    playbook=self.playbook,
                    context=context,
                    reflection=reflection_content,
                    use_json_mode=False,
                    call_id=f"consult_case_{case_id}_turn_{len(history)}_rewrite",
                    log_dir=log_dir,
                )
            except Exception:
                final_response = gen_response

        reply = (final_response or "").strip()
        if not reply:
            reply = (
                "Let me outline the key drivers, propose a hypothesis, and the first "
                "analyses I'd run to validate it."
            )
        return reply

    def run_consulting(
            self,
            mode: str,
            test_samples: List[Dict[str, Any]],
            data_processor: Any,
            config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if mode not in ["online", "eval_only"]:
            raise ValueError(f"Consulting mode must be online/eval_only, got '{mode}'")
        if not test_samples:
            raise ValueError("Consulting requires non-empty test_samples")

        # Record consulting mode & reset per-run state
        self.consulting_mode = mode
        self.consulting_cases_since_curate = 0
        self.consulting_pending_reflections = []
        self._consulting_total_samples = len(test_samples)

        # 从 config 提取通用参数，主要是 curator 频率和 token_budget
        config_params = self._extract_config_params(config)
        self.consulting_curator_frequency = config_params["curator_frequency"]
        self._consulting_token_budget = config_params["token_budget"]

        save_dir = config.get("save_dir", "results")
        task_name = config.get("task_name", "Consulting")
        run_subdir = (
            f"{task_name}/{self.agent_method}/{mode}/"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        # Dedicated log dir for consulting LLM calls
        self._consulting_log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(self._consulting_log_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"{self.agent_method.upper()} - CONSULTING EVALUATION")
        print(f"{'=' * 60}")
        print(f"Cases: {len(test_samples)}")
        print(f"Save dir: {resolved_save_path}")
        print(f"{'=' * 60}\n")

        # Save initial playbook snapshot before any consulting interaction
        try:
            with open(
                    os.path.join(resolved_save_path, "initial_playbook.md"),
                    "w",
                    encoding="utf-8",
            ) as f:
                f.write(self.playbook)
        except Exception:
            pass

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

        # Save final playbook after this consulting run (may differ in online mode)
        try:
            with open(
                    os.path.join(resolved_save_path, "final_playbook.md"),
                    "w",
                    encoding="utf-8",
            ) as f:
                f.write(self.playbook)
        except Exception:
            pass

        config_payload = dict(config or {})
        config_payload.update(
            {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
            }
        )
        with open(
                os.path.join(resolved_save_path, "run_config.json"),
                "w",
                encoding="utf-8",
        ) as f:
            json.dump(config_payload, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"{self.agent_method.upper()} - CONSULTING RUN COMPLETE")
        print(f"{'=' * 60}")
        print(f"Num cases: {results.get('num_cases')}, "
              f"finished: {results.get('num_finished')}, "
              f"failed: {results.get('num_failed')}")
        print(f"Metrics: {results.get('metrics')}")
        print(f"Results saved to: {resolved_save_path}")
        print(f"{'=' * 60}\n")

        return results

    # ==========================================================
    # BeerGame support (SeriousGame) - policy + run wrapper
    # ==========================================================

    def _decide_order_qty(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """
        Single-step BeerGame decision.

        This implementation:
        - uses task-specific helpers from utils.seriousgame_tools
        - uses ACE workflow: Analyze (generator) -> Critique (reflector) -> Enhance (generator)
        - optionally curates the playbook for future steps (persistent strategy memory)
        """
        role = str(ctx.get("role", obs.get("role", "retailer")))

        # Baseline order from simple rule
        base_order = beergame_base_rule_order(
            obs=obs,
            ctx=ctx,
            max_order_qty=getattr(self, "max_order_qty", 5000),
        )

        # Use playbook as persistent "retrieved" guidance (truncate to avoid prompt blow-up)
        playbook_text = getattr(self, "playbook", "") or ""
        retrieved = playbook_text[:6000] if isinstance(playbook_text, str) else ""

        system, user = beergame_render_prompt(
            role=role,
            obs=obs,
            retrieved=retrieved,
            base_order=base_order,
        )

        max_order_qty = int(getattr(self, "max_order_qty", 5000))

        # -----------------------
        # 1) Analyze (Generator)
        # -----------------------
        gen_user = (
            user
            + "\n\n"
            + "ACE Analyze: Propose an order quantity.\n"
              "Output JSON ONLY, exactly {\"order_qty\": <int>, \"note\": \"...\"}.\n"
              "Keep note short. No extra keys. No extra text."
        )
        js1 = self._call_llm_json(system=system, user=gen_user)
        order1, note1 = beergame_extract_order_and_note(
            js=js1,
            base_order=base_order,
            max_order_qty=max_order_qty,
        )

        # -----------------------
        # 2) Critique (Reflector)
        # -----------------------
        try:
            # BeerGame没有ground truth，这里仅做“健壮性/风险”反思
            question_for_reflection = f"[BeerGame] role={role} week={obs.get('week')} obs={json.dumps(obs, ensure_ascii=False)}"
            reasoning_trace = f"note={note1}"
            predicted_answer = json.dumps({"order_qty": order1, "note": note1}, ensure_ascii=False)
            environment_feedback = (
                "No immediate ground truth. Critique whether the order balances inventory vs backlog, "
                "avoids overreaction/bullwhip, and respects the baseline as a safe fallback."
            )
            reflection_content, bullet_tags, _ = self.reflector.reflect(
                question=question_for_reflection,
                reasoning_trace=reasoning_trace,
                predicted_answer=predicted_answer,
                ground_truth=None,
                environment_feedback=environment_feedback,
                bullets_used=[],
                use_ground_truth=False,
                use_json_mode=False,
                call_id=f"beergame_reflect_{ctx.get('scenario_id','')}_{ctx.get('episode_id','')}_w{obs.get('week','')}",
                log_dir=getattr(self, "_beergame_log_dir", None),
            )

            # Update playbook bullet stats if reflector produced tags
            if bullet_tags:
                self.playbook = update_bullet_counts(self.playbook, bullet_tags)

            # Cache reflections for curator
            pending = getattr(self, "_beergame_pending_reflections", None)
            if pending is None:
                pending = []
                setattr(self, "_beergame_pending_reflections", pending)
            if isinstance(reflection_content, str) and reflection_content.strip():
                pending.append(reflection_content.strip())

            # Optional curator: update playbook periodically
            steps_since = int(getattr(self, "_beergame_steps_since_curate", 0))
            steps_since += 1
            setattr(self, "_beergame_steps_since_curate", steps_since)
            freq = max(int(getattr(self, "beergame_curator_frequency", 5)), 1)
            if steps_since >= freq:
                merged_reflection = "\n\n".join(pending[-5:])  # cap context
                stats = get_playbook_stats(self.playbook)
                token_budget = int(getattr(self, "_beergame_token_budget", 40000))
                total_steps = int(getattr(self, "_beergame_total_steps_hint", steps_since))
                self.playbook, self.next_global_id, _ops, _ = self.curator.curate(
                    current_playbook=self.playbook,
                    recent_reflection=merged_reflection,
                    question_context=question_for_reflection,
                    current_step=steps_since,
                    total_samples=total_steps,
                    token_budget=token_budget,
                    playbook_stats=stats,
                    use_ground_truth=False,
                    use_json_mode=False,
                    call_id=f"beergame_curate_{ctx.get('scenario_id','')}_{ctx.get('episode_id','')}_w{obs.get('week','')}",
                    log_dir=getattr(self, "_beergame_log_dir", None),
                    next_global_id=self.next_global_id,
                )
                setattr(self, "_beergame_steps_since_curate", 0)
                setattr(self, "_beergame_pending_reflections", [])
        except Exception:
            reflection_content = ""

        # -----------------------
        # 3) Enhance (Generator)
        # -----------------------
        enhance_user = (
            user
            + "\n\n"
            + "ACE Enhance: You have a candidate decision and a critique. Refine the order if needed.\n"
            + f"Candidate decision JSON: {json.dumps({'order_qty': order1, 'note': note1}, ensure_ascii=False)}\n"
            + f"Critique:\n{reflection_content or '(no critique)'}\n\n"
            + "Return JSON ONLY, exactly {\"order_qty\": <int>, \"note\": \"...\"}. No extra keys."
        )
        js2 = self._call_llm_json(system=system, user=enhance_user)
        order2, note2 = beergame_extract_order_and_note(
            js=js2,
            base_order=order1,  # if enhance fails, fall back to analyzed decision first
            max_order_qty=max_order_qty,
        )

        # Final note for logging
        self._last_beergame_note = note2 or note1
        return int(order2)

    def run_beergame(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Entry point for SeriousGame / BeerGame evaluation.

        Delegates environment roll-out and logging to utils.seriousgame_tools,
        using self._decide_order_qty as the policy function.
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"BeerGame only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )
        if not test_samples:
            raise ValueError("BeerGame requires non-empty test_samples")

        # Allow config to override default max_order_qty if provided
        beergame_cfg = dict((config or {}).get("beergame", {}) or {})
        max_order = beergame_cfg.get("max_order_qty")
        if max_order is None:
            max_order = getattr(self, "max_order_qty", 5000)
        self.max_order_qty = int(max_order)

        # Setup persistent BeerGame playbook path (shared across runs)
        save_dir = str((config or {}).get("save_dir", "results"))
        pb_path = str(beergame_cfg.get("playbook_path") or os.path.join(save_dir, "ace_beergame_playbook.md"))
        self._beergame_playbook_path = pb_path
        try:
            if os.path.exists(pb_path):
                with open(pb_path, "r", encoding="utf-8") as f:
                    txt = f.read()
                if txt.strip():
                    self.playbook = txt
        except Exception:
            pass

        ctx = beergame_prepare_run(
            mode=mode,
            test_samples=test_samples,
            config=config or {},
            allowed_modes=self.SUPPORTED_MODES,
            agent_method=self.agent_method,
        )
        # expose log dir to reflector/curator for troubleshooting
        self._beergame_log_dir = ctx.get("log_dir")
        # reset counters for this run
        self.beergame_curator_frequency = int(beergame_cfg.get("curator_frequency", getattr(self, "beergame_curator_frequency", 5)))
        self._beergame_token_budget = int(beergame_cfg.get("playbook_token_budget", getattr(self, "_beergame_token_budget", 40000)))
        self._beergame_total_steps_hint = int(beergame_cfg.get("total_steps_hint", 25))
        self._beergame_steps_since_curate = 0
        self._beergame_pending_reflections = []

        # Save initial playbook snapshot in this run dir
        try:
            with open(os.path.join(ctx["resolved_save_path"], "initial_playbook.md"), "w", encoding="utf-8") as f:
                f.write(self.playbook)
        except Exception:
            pass
        results, error_log = beergame_evaluate_run(
            agent=self,
            test_samples=test_samples,
            config=config or {},
            ctx=ctx,
        )
        beergame_save_run(
            results=results,
            error_log=error_log,
            config=config or {},
            ctx=ctx,
        )

        # Save final playbook snapshot + persist global playbook
        try:
            with open(os.path.join(ctx["resolved_save_path"], "final_playbook.md"), "w", encoding="utf-8") as f:
                f.write(self.playbook)
        except Exception:
            pass
        try:
            with open(pb_path, "w", encoding="utf-8") as f:
                f.write(self.playbook)
        except Exception:
            pass
        return results

    def _record_edt_experience(
        self,
        scenario: Dict[str, Any],
        schema: Dict[str, Any],
        metrics: Dict[str, Any],
        repeat_idx: int,
    ) -> None:
        """
        由 EDT 评测侧在每个 episode 结束后调用。

        用 Reflector + Curator 把 “场景 + 决策 + 结果指标” 总结成可复用的经验，
        并写入 playbook。
        """
        # 基础信息
        scenario_id = scenario.get("scenario_id", "unknown")
        config = scenario.get("config", {})
        base_summary = config.get("base_summary") or {}

        # 文本化：给 Reflector 看的 “推理轨迹 + 环境反馈”
        from pprint import pformat

        decision_text = json.dumps(schema, ensure_ascii=False)
        metrics_text = json.dumps(metrics, ensure_ascii=False)

        reasoning_trace = (
            f"EDT scenario run #{repeat_idx} for scenario_id={scenario_id}.\n\n"
            f"Base summary (template-level):\n{pformat(base_summary, indent=2)}\n\n"
            f"Chosen schema (C, R, P):\n{decision_text}\n\n"
            f"Outcome metrics:\n{metrics_text}\n"
        )

        env_feedback = (
            "Simulation outcome summary:\n"
            f"- accumulated_earnings = {metrics.get('accumulated_earnings')}\n"
            f"- accumulated_revenue = {metrics.get('accumulated_revenue')}\n"
            f"- accumulated_expenses = {metrics.get('accumulated_expenses')}\n"
            f"- overall_profit_margin = {metrics.get('overall_profit_margin')}\n"
            f"- overall_avg_utilization = {metrics.get('overall_avg_utilization')}\n"
        )

        question_for_reflection = (
            "From this single EDT simulation run, extract concise, reusable "
            "lessons for how to choose C (consultants), R (risk level), and "
            "project windows P to maximize profit while keeping utilization "
            "healthy and avoiding obviously fragile strategies."
        )

        # 1) Reflector：把一局的结果总结成若干条经验 bullet（文本）
        reflection_content, bullet_tags, _ = self.reflector.reflect(
            question=question_for_reflection,
            reasoning_trace=reasoning_trace,
            predicted_answer=json.dumps(schema, ensure_ascii=False),
            ground_truth=None,                     # EDT 没有 ground truth，只看收益表现
            environment_feedback=env_feedback,
            bullets_used=None,
            use_ground_truth=False,
            use_json_mode=False,
            call_id=f"edt_reflect_{scenario_id}_rep{repeat_idx}",
            log_dir=None,
        )

        # 2) Curator：把刚刚这段 reflection 内容以小步更新方式纳入 playbook
        #    这里复用 _extract_config_params 提供的 curator_frequency / token_budget 等。
        config_params = self._edt_config_params or self._extract_config_params({})
        curator_frequency = config_params.get("curator_frequency", 1)
        token_budget = config_params.get("token_budget", 4096)

        # 更新计数
        self._edt_episode_counter += 1
        step = self._edt_episode_counter
        total_samples = self._edt_total_episodes or max(step, 1)

        # 只有在到达 curator_frequency 的步数时才真正跑一次 curator，避免太频繁
        if step % curator_frequency != 0:
            return

        stats = get_playbook_stats(self.playbook)

        self.playbook, self.next_global_id, _, _ = self.curator.curate(
            current_playbook=self.playbook,
            recent_reflection=reflection_content,
            question_context=reasoning_trace,
            current_step=step,
            total_samples=total_samples,
            token_budget=token_budget,
            playbook_stats=stats,
            use_ground_truth=False,
            use_json_mode=False,
            call_id=f"edt_curate_{scenario_id}_rep{repeat_idx}",
            log_dir=None,
            next_global_id=self.next_global_id,
        )

        # 如果你在 ACE 中启用了 BulletpointAnalyzer，也可以在这里再跑一轮去重：
        if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
            self.playbook = self.bulletpoint_analyzer.analyze(
                playbook=self.playbook,
                threshold=self.bulletpoint_analyzer_threshold,
                merge=True,
            )

    # ==========================================================
    # EDT support (SeriousGame) - scenario schema + run wrapper
    # ==========================================================
    def _get_edt_experience_snippet(self, max_chars: int = 2500) -> str:
        """
        从当前 playbook 中抽取与 EDT 相关的经验片段，控制长度。
        这里先用一个简单启发式：优先保留最近的内容，必要时做行级筛选。
        """
        text = getattr(self, "playbook", "") or ""
        if not text:
            return ""

        # 简单策略 1：按行筛一下含 EDT 或 consultants / risk 的行
        lines = text.splitlines()
        edt_lines = [
            ln for ln in lines
            if ("EDT" in ln)
               or ("consultant" in ln.lower())
               or ("risk" in ln.lower())
               or ("profit" in ln.lower())
               or ("utilization" in ln.lower())
        ]
        snippet = "\n".join(edt_lines).strip()

        # 如果筛出来太少，就退化为取 playbook 的尾部
        if not snippet:
            snippet = text[-max_chars:].strip()
        elif len(snippet) > max_chars:
            snippet = snippet[-max_chars:].strip()

        return snippet

    def _decide_edt_scenario_schema(
        self,
        base_summary: Dict[str, Any],
        scenario_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Single-scenario EDT design decision.

        This implementation:
        - uses utils.seriousgame_tools EDT helpers
        - does NOT touch ACE playbook; purely scenario-level policy.
        """
        ctx = build_edt_decision_context(
            base_summary=base_summary,
            scenario_meta=scenario_meta,
            max_steps_hint=(scenario_meta or {}).get("max_steps"),
        )
        # 让 seriousgame_tools 生成 system / user prompt
        system, user = render_edt_prompt(ctx)

        # 2) 从 playbook 中抽取 EDT 相关经验片段
        edt_experience = self._get_edt_experience_snippet(max_chars=2500)
        # print(f"used experience: {edt_experience}")
        if edt_experience:
            user = (
                    user
                    + "\n\n=== The experience from previous runs ===\n"
                    + edt_experience
                    + "\n"
            )

        # 3) 最后再加一句明确的输出约束，避免 LLM 输出解释
        user = (
                user
                + "\n\nIMPORTANT OUTPUT CONSTRAINTS:\n"
                + "- Output ONLY a single JSON object.\n"
                + "- The JSON object must contain ONLY keys: C, R, P.\n"
                + "- Do not wrap JSON in markdown fences.\n"
        )

        raw = self._call_llm_json(system=system, user=user)
        return normalize_edt_schema(raw, ctx)

    def run_edt(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Entry point for SeriousGame / EDT evaluation.

        Delegates scenario construction and roll-out to utils.seriousgame_tools,
        using self._decide_edt_scenario_schema as the policy for scenario design.
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"EDT only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )

        # 记录下与 curator 相关的参数，供 _record_edt_experience 使用
        self._edt_config_params = self._extract_config_params(config)
        self._edt_episode_counter = 0

        # 可以根据 edt 配置推一个 “总 episode 数”，主要是给 curator 提供 total_samples
        edt_cfg = dict(config.get("edt", {}) or {})
        repeats = int(edt_cfg.get("repeats", 1))
        self._edt_total_episodes = max(repeats * max(len(test_samples), 1), 1)

        ctx = edt_prepare_run(
            mode=mode,
            test_samples=test_samples,
            config=config or {},
            allowed_modes=self.SUPPORTED_MODES,
        )
        results, error_log = edt_evaluate_run(
            agent=self,
            test_samples=test_samples,
            config=config or {},
            ctx=ctx,
        )
        edt_save_run(
            results=results,
            error_log=error_log,
            config=config or {},
            ctx=ctx,
        )
        return results
    
    def _initialize_empty_playbook(self) -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""
    
    def _extract_config_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract common configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with extracted parameters
        """
        return {
            'num_epochs': config.get('num_epochs', 1),
            'max_num_rounds': config.get('max_num_rounds', 3),
            'curator_frequency': config.get('curator_frequency', 1),
            'eval_steps': config.get('eval_steps', 100),
            'save_steps': config.get('save_steps', 50),
            'token_budget': config.get('playbook_token_budget', 80000),
            'task_name': config.get('task_name', 'default'),
            'use_json_mode': config.get('json_mode', False),
            'no_ground_truth': config.get('no_ground_truth', False),
            'save_dir': config.get('save_dir', './results'),
            'test_workers': config.get('test_workers', 20),
            'use_bulletpoint_analyzer': config.get('use_bulletpoint_analyzer', False),
            'bulletpoint_analyzer_threshold': config.get('bulletpoint_analyzer_threshold', 0.90)
        }

    @staticmethod
    def _strip_task_instruction(question: str) -> str:
        """
        去掉 DataProcessor 注入的任务级指令前缀，避免在反思/策展阶段干扰角色输出。
        """
        try:
            from bizbench.data_processor import DataProcessor

            instruction_values = list(DataProcessor.TASK_INSTRUCTIONS.values()) + list(
                DataProcessor.SPECIAL_FIN_INSTRUCTIONS.values()
            )
            for instr in instruction_values:
                if not instr:
                    continue
                if question.startswith(instr):
                    return question[len(instr):].lstrip()
        except Exception:
            # 安全降级：若遇到导入或匹配异常，保留原问题文本
            return question

        return question
    
    def _setup_paths(self, save_dir: str, task_name: str, mode: str) -> Tuple[str, str]:
        """
        Setup logging paths and directories.
        
        Args:
            save_dir: Base path for saving results
            task_name: task name
            mode: 'offline', 'online', or 'eval_only'
            
        Returns:
            Tuple of (usage_log_path, playbook_dir)
        """
        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(task_name, self.agent_method, mode, timestamp)
        save_path = os.path.join(save_dir, run_folder)
        os.makedirs(save_path, exist_ok=True)
        log_dir = os.path.join(save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        if mode == "eval_only":
            return save_path, log_dir

        usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
        playbook_dir = os.path.join(save_path, "intermediate_playbooks")
        os.makedirs(playbook_dir, exist_ok=True)
        
        return save_path, usage_log_path, playbook_dir, log_dir
    
    def run(
        self,
        mode: str,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entrypoint for running ACE system in different modes.
        
        Args:
            mode: Run mode - 'offline', 'online', or 'eval_only'
            train_samples: Training samples (required for offline mode)
            val_samples: Validation samples (required for offline mode)
            test_samples: Test samples (required for online and eval_only modes)
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            
        Returns:
            Dictionary with results depending on the mode
        """
        # Validate inputs
        if mode not in ['offline', 'online', 'eval_only']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'offline', 'online', or 'eval_only'")
        
        if mode == 'offline' and (train_samples is None or val_samples is None):
            raise ValueError("Offline mode requires train_samples and val_samples")
        
        if mode == 'online' and test_samples is None:
            raise ValueError("Online mode requires test_samples")
        
        if mode == 'eval_only' and test_samples is None:
            raise ValueError("eval_only mode requires test_samples")
        
        # Extract configuration
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        save_dir = config_params['save_dir']
        name_lower = str(task_name).lower()

        # Consulting routing
        if "consult" in name_lower:
            return self.run_consulting(
                mode=mode,
                test_samples=test_samples or [],
                data_processor=data_processor,
                config=config or {},
            )

        # BeerGame routing
        if "beer" in name_lower:
            return self.run_beergame(
                mode=mode,
                test_samples=test_samples or [],
                data_processor=data_processor,
                config=config or {},
            )

        # EDT routing
        if "edt" in name_lower:
            return self.run_edt(
                mode=mode,
                test_samples=test_samples or [],
                data_processor=data_processor,
                config=config or {},
            )

        # Setup paths based on mode
        if mode == 'eval_only':
            save_path, log_dir = self._setup_paths(save_dir, task_name, mode)
            usage_log_path = None
            playbook_dir = None
        else:
            save_path, usage_log_path, playbook_dir, log_dir = self._setup_paths(save_dir, task_name, mode)
        
        # Save configuration
        run_subdir = os.path.relpath(save_path, save_dir)
        config_path = os.path.join(save_path, "run_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "task_name": task_name,
                "mode": mode,
                "generator_model": self.generator.model,
                "reflector_model": self.reflector.model,
                "curator_model": self.curator.model,
                "config": config,
                "run_subdir": run_subdir,
            }, f, indent=2)
        
        # Print initial banner
        print(f"\n{'='*60}")
        print(f"ACE SYSTEM - {mode.upper().replace('_', ' ')} MODE")
        print(f"{'='*60}")
        print(f"Task: {task_name}")
        if mode == 'offline':
            print(f"Train samples: {len(train_samples)}")
            print(f"Validation samples: {len(val_samples)}")
            if test_samples:
                print(f"Test samples: {len(test_samples)}")
        elif mode == 'online':
            print(f"Test samples (used for training and testing): {len(test_samples)}")
        else:  # eval_only
            print(f"Test samples: {len(test_samples)}")
        print(f"{'='*60}\n")
        
        # Execute based on mode
        results = {}
        
        if mode == 'offline':
            # OFFLINE MODE WORKFLOW
            # 1. Run initial test if test_samples provided
            if test_samples:
                print(f"\n{'='*60}")
                print(f"INITIAL TEST (before training)")
                print(f"{'='*60}\n")
                initial_test_results = self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.playbook,
                    config=config,
                    log_dir=log_dir,
                    save_path=save_path,
                    prefix="initial"
                )
                results['initial_test_results'] = initial_test_results
                print(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")
            
            # 2. Run offline training
            print(f"\n{'='*60}")
            print(f"STARTING OFFLINE TRAINING")
            print(f"{'='*60}\n")
            training_results = self._offline_train(
                train_samples=train_samples,
                val_samples=val_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir
            )
            results['training_results'] = training_results
            
            # 3. Run final test if test_samples provided
            if test_samples:
                print(f"\n{'='*60}")
                print(f"FINAL TEST (with best playbook)")
                print(f"{'='*60}\n")
                final_test_results = self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.best_playbook,
                    config=config,
                    log_dir=log_dir,
                    save_path=save_path,
                    prefix="final"
                )
                results['final_test_results'] = final_test_results
                print(f"Final Test Accuracy: {final_test_results['accuracy']:.3f}\n")
        
        elif mode == 'online':
            # ONLINE MODE WORKFLOW
            # 1. Run initial test
            print(f"\n{'='*60}")
            print(f"INITIAL TEST (before training)")
            print(f"{'='*60}\n")
            initial_test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="initial"
            )
            results['initial_test_results'] = initial_test_results
            print(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")
            
            # 2. Run online training and testing
            print(f"\n{'='*60}")
            print(f"STARTING ONLINE TRAIN AND TEST")
            print(f"{'='*60}\n")
            online_results = self._online_train_and_test(
                test_samples=test_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir
            )
            results['online_test_results'] = online_results
        
        else:  # eval_only
            # EVAL ONLY MODE WORKFLOW
            print(f"\n{'='*60}")
            print(f"RUNNING TEST")
            print(f"{'='*60}\n")
            test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="test"
            )
            results['test_results'] = test_results
        
        # Save consolidated results
        final_results_path = os.path.join(save_path, "final_results.json")
        with open(final_results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Mode: {mode.upper().replace('_', ' ')}")
        if mode == 'offline':
            print(f"Best Validation Accuracy: {results['training_results']['best_validation_accuracy']:.3f}")
            if test_samples:
                print(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
                print(f"Final Test Accuracy: {results['final_test_results']['accuracy']:.3f}")
        elif mode == 'online':
            print(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
            print(f"Final Test Accuracy: {results['online_test_results']['accuracy']:.3f}")
        else:  # eval_only
            print(f"Test Accuracy: {results['test_results']['accuracy']:.3f}")
        print(f"Results saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def _run_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook: str,
        config: Dict[str, Any],
        log_dir: str,
        save_path: str,
        prefix: str = "test"
    ) -> Dict[str, Any]:
        """
        Run testing
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to use for testing
            config: Configuration dictionary
            log_dir: Directory for detailed logs
            save_path: Path to save results
            prefix: Prefix for saved files (e.g., 'initial', 'final', 'test')
            
        Returns:
            Dictionary with test results
        """
        config_params = self._extract_config_params(config)
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']
        
        test_results, test_error_log = evaluate_test_set(
            data_processor,
            self.generator,
            playbook,
            test_samples,
            self.max_tokens,
            log_dir,
            max_workers=test_workers,
            use_json_mode=use_json_mode
        )

        # Save test results
        test_results_path = os.path.join(save_path, f"{prefix}_test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_results": test_results,
                "error_log": test_error_log,
            }, f, indent=2)
        
        return test_results
    
    def _train_single_sample(
        self,
        task_dict: Dict[str, Any],
        data_processor,
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: str,
        log_dir: str,
        config_params: Dict[str, Any],
        total_samples: int
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Train on a single sample with reflection and curation.
        
        Args:
            task_dict: Sample dictionary with question, context, target
            data_processor: Data processor for evaluation
            step_id: Identifier string for this step (e.g., "train_e_1_s_10" or "online_train_w_1_s_5")
            epoch: Current epoch number
            step: Current step number
            usage_log_path: Path for bullet usage logging
            log_dir: Path for logging directory
            config_params: Configuration parameters dictionary
            total_samples: Total number of samples in dataset
            
        Returns:
            Tuple of (pre_train_answer, post_train_answer, tracking_dict)
        """
        # Extract configuration
        max_num_rounds = config_params['max_num_rounds']
        curator_frequency = config_params['curator_frequency']
        token_budget = config_params['token_budget']
        use_json_mode = config_params['use_json_mode']
        no_ground_truth = config_params['no_ground_truth']
        
        # Extract sample data
        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        target = task_dict.get("target", "")
        question_for_reflection = self._strip_task_instruction(question)
        
        # STEP 1: Initial generation (pre-train)
        print("Generating initial answer...")
        gen_response, bullet_ids, call_info = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir
        )
        
        # Extract answer and check correctness
        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)
        pre_train_answer = final_answer
        
        print(f"Correct: {is_correct}")
        
        # Log bullet usage
        log_bullet_usage(usage_log_path, epoch, step, task_dict, bullet_ids,
                       playbook=self.playbook, is_correct=is_correct)
        
        # Track pre-train result
        tracking_dict = {
            "pre_train_result": {
                "final_answer": final_answer,
                "is_correct": is_correct,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook)
            }
        }
        
        reflection_content = "(empty)"
        
        # STEP 2: Reflection and regeneration
        if not is_correct:
            # For incorrect answers - iterate reflection rounds
            for round_num in range(max_num_rounds):
                print(f"Reflection round {round_num + 1}/{max_num_rounds}")
                
                # Get bullets for reflector
                playbook_bullets = extract_playbook_bullets(
                    self.playbook, bullet_ids
                )
                
                # Reflect on error
                reflection_content, bullet_tags, _ = self.reflector.reflect(
                    question=question_for_reflection,
                    reasoning_trace=gen_response,
                    predicted_answer=final_answer,
                    ground_truth=target if not no_ground_truth else None,
                    environment_feedback="Predicted answer does not match ground truth",
                    bullets_used=playbook_bullets,
                    use_ground_truth=not no_ground_truth,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_round_{round_num}",
                    log_dir=log_dir
                )
                
                # Update bullet counts
                if bullet_tags:
                    self.playbook = update_bullet_counts(
                        self.playbook, bullet_tags
                    )
                
                # Regenerate with reflection
                gen_response, bullet_ids, _ = self.generator.generate(
                    question=question,
                    playbook=self.playbook,
                    context=context,
                    reflection=reflection_content,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_post_reflect_round_{round_num}",
                    log_dir=log_dir
                )
                
                final_answer = extract_answer(gen_response)
                
                if data_processor.answer_is_correct(final_answer, target):
                    print(f"Corrected after reflection round {round_num + 1}!")
                    is_correct = True
                    break
        
        else:
            # For correct answers - still run reflector to tag helpful bullets
            playbook_bullets = extract_playbook_bullets(
                self.playbook, bullet_ids
            )
            
            reflection_content, bullet_tags, _ = self.reflector.reflect(
                question=question_for_reflection,
                reasoning_trace=gen_response,
                predicted_answer=final_answer,
                ground_truth=target if not no_ground_truth else None,
                environment_feedback="Predicted answer matches ground truth",
                bullets_used=playbook_bullets,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_reflect_on_correct",
                log_dir=log_dir
            )
            
            # Update bullet counts
            if bullet_tags:
                self.playbook = update_bullet_counts(
                    self.playbook, bullet_tags
                )
            
            # Log with reflection
            log_bullet_usage(usage_log_path, epoch, step, task_dict, bullet_ids,
                           playbook=self.playbook, 
                           reflection_content=reflection_content,
                           is_correct=is_correct)
        
        # STEP 3: Curator - Periodically update playbook
        if step % curator_frequency == 0:
            print(f"\n--- Running Curator at step {step} ---")
            
            stats = get_playbook_stats(self.playbook)
            
            self.playbook, self.next_global_id, operations, _ = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=reflection_content,
                question_context=context,
                current_step=step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=step_id,
                log_dir=log_dir,
                next_global_id=self.next_global_id
            )
            
            # Run bulletpoint analyzer if enabled
            if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
                print(f"  Running BulletpointAnalyzer (threshold={self.bulletpoint_analyzer_threshold})...")
                self.playbook = self.bulletpoint_analyzer.analyze(
                    playbook=self.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True
                )
        
        # STEP 4: Post-curator generation
        gen_response, _, _ = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_post_curate",
            log_dir=log_dir
        )
        
        final_answer = extract_answer(gen_response)
        post_train_answer = final_answer
        
        post_train_is_correct = data_processor.answer_is_correct(final_answer, target)
        tracking_dict["post_train_result"] = {
            "final_answer": final_answer,
            "is_correct": post_train_is_correct,
            "playbook_num_tokens": count_tokens(self.playbook),
            "playbook_length": len(self.playbook)
        }
        
        return pre_train_answer, post_train_answer, tracking_dict
    
    def _offline_train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str
    ) -> Dict[str, Any]:
        """
        Run offline training
        
        Args:
            train_samples: List of training samples
            val_samples: List of validation samples
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            
        Returns:
            Dictionary with training results
        """
        # Extract configuration using helper
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        num_epochs = config_params['num_epochs']
        eval_steps = config_params['eval_steps']
        save_steps = config_params['save_steps']
        test_workers = config_params['test_workers']
        use_json_mode = config_params['use_json_mode']
        curator_frequency = config_params['curator_frequency']
        
        # Initialize tracking
        results = []
        pre_train_post_train_results = []
        error_logs = []
        best_accuracy = 0.0
        self.best_playbook = self.playbook

        print(f"Total epochs: {num_epochs}")
        print(f"Train samples per epoch: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print(f"Curator frequency: every {curator_frequency} steps")
        print(f"Evaluation frequency: every {eval_steps} steps\n")
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []
            
            for step, task_dict in enumerate(train_samples):
                step += 1
                print(f"\n--- Step {step}/{len(train_samples)} ---")
                
                target = task_dict.get("target", "")
                
                # Use helper method for training single sample
                pre_train_answer, post_train_answer, tracking_dict = self._train_single_sample(
                    task_dict=task_dict,
                    data_processor=data_processor,
                    step_id=f"train_e_{epoch}_s_{step}",
                    epoch=epoch,
                    step=step,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=len(train_samples)
                )
                
                # Collect answers for accuracy calculation
                epoch_answers_pre_train.append(pre_train_answer)
                epoch_targets_pre_train.append(target)
                epoch_answers_post_train.append(post_train_answer)
                epoch_targets_post_train.append(target)
                
                # Track pre-train and post-train results
                pre_train_post_train_result = {
                    "epoch": epoch,
                    "step": step,
                    "target": target,
                    **tracking_dict
                }
                pre_train_post_train_results.append(pre_train_post_train_result)
                
                # Save intermediate playbook
                if step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"epoch_{epoch}_step_{step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
                
                # Periodic evaluation
                if step % eval_steps == 0:
                    print(f"\n{'='*40}")
                    print(f"EVALUATION AT EPOCH {epoch}, STEP {step}")
                    print(f"{'='*40}")
                    
                    # Compute training accuracies
                    pre_train_accuracy = data_processor.evaluate_accuracy(
                        epoch_answers_pre_train, epoch_targets_pre_train
                    )
                    post_train_accuracy = data_processor.evaluate_accuracy(
                        epoch_answers_post_train, epoch_targets_post_train
                    )
                    
                    # Validation evaluation
                    val_results = {}
                    if val_samples:
                        val_results, val_error_log = evaluate_test_set(
                            data_processor, self.generator, self.playbook, 
                            val_samples, self.max_tokens, log_dir, 
                            max_workers=test_workers, use_json_mode=use_json_mode
                        )
                    
                    result = {
                        "epoch": epoch,
                        "step": step,
                        "train_result": {
                            "pre_train_accuracy": pre_train_accuracy,
                            "post_train_accuracy": post_train_accuracy
                        },
                        "val_result": val_results,
                        "playbook_num_tokens": count_tokens(self.playbook),
                        "playbook_length": len(self.playbook),
                        "playbook_stats": get_playbook_stats(self.playbook)
                    }
                    results.append(result)
                    error_logs.append({
                        "epoch": epoch,
                        "step": step,
                        "val_results": val_results,
                        "error_log": val_error_log
                    })

                    # Track best playbook
                    if val_results:
                        acc = val_results["accuracy"]
                        if acc > best_accuracy:
                            best_accuracy = acc
                            self.best_playbook = self.playbook
                            print(f"🎉 New best accuracy: {best_accuracy:.3f}")
                    
                    # Save results
                    results_path = os.path.join(save_path, "train_results.json")
                    with open(results_path, "w") as f:
                        json.dump({
                            "best_accuracy": best_accuracy,
                            "results": results,
                        }, f, indent=2)
                    
                    error_logs_path = os.path.join(save_path, "val_results.json")
                    with open(error_logs_path, "w") as f:
                        json.dump(error_logs, f, indent=2)
            
            # End of epoch - save final playbook
            epoch_playbook_path = os.path.join(
                playbook_dir, f"epoch_{epoch}_final_playbook.txt"
            )
            with open(epoch_playbook_path, "w") as f:
                f.write(self.playbook)

        # Save training results
        results_path = os.path.join(save_path, "train_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "best_accuracy": best_accuracy,
                "results": results,
            }, f, indent=2)
        
        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)
        
        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)
        
        # Save best playbook
        best_playbook_path = os.path.join(save_path, f"best_playbook.txt")
        with open(best_playbook_path, "w") as f:
            f.write(self.best_playbook)
        
        print(f"\n{'='*60}")
        print(f"OFFLINE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_accuracy:.3f}")
        print(f"{'='*60}\n")

        return {"best_validation_accuracy": best_accuracy}

    
    def test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run testing with the playbook (backward compatibility wrapper).
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to be used for generator
            config: Configuration dictionary
            
        Returns:
            Dictionary with test results
        """
        # Temporarily set the playbook
        old_playbook = self.playbook
        self.playbook = playbook
        
        # Use the run method
        results = self.run(
            mode='eval_only',
            test_samples=test_samples,
            data_processor=data_processor,
            config=config
        )
        
        # Restore old playbook
        self.playbook = old_playbook
        
        # Return in the old format for backward compatibility
        return {
            "test_results": results['test_results'],
            "error_log": results.get('test_error_log', {}),
            "playbook": playbook
        }
    
    def _online_train_and_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str
    ) -> Dict[str, Any]:
        """
        Run online training and testing
        
        Args:
            test_samples: List of samples to train and test on
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            
        Returns:
            Dictionary with training results, test results, and final playbook
        """
        # Extract configuration using helper
        config_params = self._extract_config_params(config)
        num_epochs = config_params['num_epochs']
        
        # Validate configuration
        if num_epochs != 1:
            raise ValueError(f"online_train_and_test requires num_epochs=1, got {num_epochs}")
        
        # Extract additional parameters
        curator_frequency = config_params['curator_frequency']
        task_name = config_params['task_name']
        save_steps = config_params['save_steps']
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']
        online_eval_frequency = config.get('online_eval_frequency', 100)  # Get from config
        
        # Initialize tracking
        train_results = []
        pre_train_post_train_results = []
        
        # Test tracking - accumulate across all windows
        correct_count_sample_based = 0
        correct_count = 0
        total_count = 0
        all_test_errors = []
        window_test_results = []
        print(f"Total samples: {len(test_samples)}")
        print(f"Window size: {online_eval_frequency}")
        print(f"Number of windows: {(len(test_samples) + online_eval_frequency - 1) // online_eval_frequency}")
        print(f"Curator frequency: every {curator_frequency} steps")
        
        # Split samples into windows
        num_windows = (len(test_samples) + online_eval_frequency - 1) // online_eval_frequency
        
        epoch = 1  # Always 1 epoch
        global_step = 0
        
        for window_idx in range(num_windows):
            start_idx = window_idx * online_eval_frequency
            end_idx = min((window_idx + 1) * online_eval_frequency, len(test_samples))
            window_samples = test_samples[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"WINDOW {window_idx + 1}/{num_windows}")
            print(f"Samples {start_idx} to {end_idx - 1}")
            print(f"{'='*60}")
            
            # =================================================================
            # STEP 1: TEST on window with current playbook (before training)
            # =================================================================
            print(f"\n--- Testing window {window_idx + 1} with current playbook ---")
            
            # Use evaluate_test_set for parallel evaluation
            window_test_results_dict, window_test_error_log = evaluate_test_set(
                data_processor,
                self.generator,
                self.playbook,
                window_samples,
                self.max_tokens,
                log_dir,
                max_workers=test_workers,
                use_json_mode=use_json_mode
            )
            
            # Extract results
            window_accuracy = window_test_results_dict['accuracy']
            window_correct = window_test_results_dict['correct']
            window_total = window_test_results_dict['total']
            correct_count_sample_based += window_correct
            correct_count += window_accuracy * window_total
            total_count += window_total
            
            # Add errors with window and global index information
            for error in window_test_error_log['errors']:
                all_test_errors.append({
                    "window": window_idx + 1,
                    "global_index": start_idx + error['index'],
                    "prediction": error['prediction'],
                    "ground_truth": error['ground_truth']
                })
            
            window_test_results.append({
                "window": window_idx + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "window_accuracy": window_accuracy,
                "window_correct": window_correct,
                "window_total": window_total
            })
            
            # Calculate cumulative test accuracy so far
            cumulative_test_accuracy = correct_count / total_count
            
            print(f"Window {window_idx + 1} test accuracy: {window_accuracy:.3f}")
            print(f"Cumulative test accuracy so far: {cumulative_test_accuracy:.3f} "
                  f"({total_count} samples)")
            
            # =================================================================
            # STEP 2: TRAIN on window (same as offline_train)
            # =================================================================
            print(f"\n--- Training on window {window_idx + 1} ---")
            
            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []
            
            for local_step, task_dict in enumerate(window_samples):
                global_step += 1
                local_step += 1
                
                print(f"\n--- Window {window_idx + 1}, Step {local_step}/{len(window_samples)} "
                      f"(Global step {global_step}) ---")
                
                target = task_dict.get("target", "")
                
                # Use helper method for training single sample
                pre_train_answer, post_train_answer, tracking_dict = self._train_single_sample(
                    task_dict=task_dict,
                    data_processor=data_processor,
                    step_id=f"online_train_s_{global_step}",
                    epoch=epoch,
                    step=global_step,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=len(test_samples)
                )
                
                # Collect answers for accuracy calculation
                epoch_answers_pre_train.append(pre_train_answer)
                epoch_targets_pre_train.append(target)
                epoch_answers_post_train.append(post_train_answer)
                epoch_targets_post_train.append(target)
                
                # Track pre-train and post-train results
                pre_train_post_train_result = {
                    "window": window_idx + 1,
                    "global_step": global_step,
                    "target": target,
                    **tracking_dict
                }
                pre_train_post_train_results.append(pre_train_post_train_result)
                
                # Save intermediate playbook
                if global_step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"step_{global_step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
            
            # End of window - compute training accuracies for this window
            pre_train_accuracy = data_processor.evaluate_accuracy(
                epoch_answers_pre_train, epoch_targets_pre_train
            )
            post_train_accuracy = data_processor.evaluate_accuracy(
                epoch_answers_post_train, epoch_targets_post_train
            )
            
            window_train_result = {
                "window": window_idx + 1,
                "global_step": global_step,
                "train_result": {
                    "pre_train_accuracy": pre_train_accuracy,
                    "post_train_accuracy": post_train_accuracy
                },
                "cumulative_test_accuracy": cumulative_test_accuracy,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook),
                "playbook_stats": get_playbook_stats(self.playbook)
            }
            train_results.append(window_train_result)
            
            print(f"\nWindow {window_idx + 1} training complete:")
            print(f"  Pre-train accuracy: {pre_train_accuracy:.3f}")
            print(f"  Post-train accuracy: {post_train_accuracy:.3f}")
            
            # Save window playbook
            window_playbook_path = os.path.join(
                playbook_dir, f"window_{window_idx + 1}_final_playbook.txt"
            )
            with open(window_playbook_path, "w") as f:
                f.write(self.playbook)
        
        # All windows complete
        print(f"\n{'='*60}")
        print(f"ONLINE TRAIN AND TEST COMPLETE")
        print(f"{'='*60}")
        
        # Calculate final cumulative test accuracy
        assert total_count == len(test_samples)
        final_test_accuracy = correct_count / total_count
        
        test_results = {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
            "window_results": window_test_results
        }
        
        test_error_log = {
            "accuracy": final_test_accuracy,
            "errors": all_test_errors
        }

        # Save test results
        test_results_path = os.path.join(save_path, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_accuracy": final_test_accuracy,
                "test_results": test_results,
                "test_error_log": test_error_log
            }, f, indent=2)
        
        # Save training results (per window)
        train_results_path = os.path.join(save_path, "train_results.json")
        with open(train_results_path, "w") as f:
            json.dump({"train_results": train_results}, f, indent=2)
        
        # Save pre-train/post-train results
        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)
        
        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)
        
        print(f"\n{'='*60}")
        print(f"ONLINE TRAINING AND TESTING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Test Accuracy: {final_test_accuracy:.3f}")
        print(f"{'='*60}\n")
        
        return {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
        }