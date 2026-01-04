# Agent Extension Specification (Detailed Version)

This document is designed to help developers who are **new to this project** quickly understand the process for extending Agents. It covers the overall architecture, interface conventions, sample code, script integration, and a validation checklist to ensure new methods can be implemented efficiently.

---

## 1. Global Architecture Overview

```
CLI (StructuredReasoning/run.py)
├── Parse command line arguments (--agent_method / --task_name / --mode / ...)
├── Load configuration + Data → DataProcessor
└── Dispatch to Agents/ subpackages based on agent_method

Agents/
├── ace/  → Full ACE process: Generator → Reflector → Curator
└── cot/  → Lightweight Baseline: Single Generator + Unified Prompt

DataProcessor
└── Responsible for sample preprocessing, answer determination, and accuracy calculation

Results
└── <save_path>/<task>/<agent>/<mode>/<timestamp>/...
```

### Process Comparison (ACE vs. CoT)

| Step | ACE | CoT Baseline |
| --- | --- | --- |
| Prompt | Different prompts for three roles (see `Agents/ace/prompts/`) | Single prompt, JSON output (`Agents/cot/generator.py`) |
| Feedback Loop | Generate → Reflect → Curate | No feedback, single-pass generation |
| Supported Modes | offline / online / eval_only | online / eval_only (extensible) |
| Results | `final_results.json`, `train_results.json`, etc. | `test_results.json` + `run_config.json` |

---

## 2. Quick Start Checklist

1. Read this document to confirm the required modes and functionalities.
2. Create a new subpackage under `Agents/`, following the pattern of `ace` or `cot`.
3. Implement core components & the Agent class, adhering to the unified interface.
4. Modify `StructuredReasoning/run.py` to map `--agent_method` to your new Agent.
5. Update corresponding shell scripts (if necessary).
6. Perform a smoke test to confirm that directory structure and result output are normal.
7. Document usage in the README or docs.

---

## 3. Directory Structure and Responsibilities

```
Agents/
├── __init__.py                       # Unified export
├── ace/
│   ├── ace.py                        # ACE System main class
│   ├── core/                         # Generator / Reflector / Curator implementation
│   └── prompts/                      # Prompt templates for three roles
└── cot/
    ├── agent.py                      # CoT Agent (run() logic)
    └── generator.py                  # Prompt builder + timed_llm_call
```

- **StructuredReasoning Entry Point**: `StructuredReasoning/run.py` selects the Agent via `--agent_method`; ACE is the default, and CoT serves as a baseline.
- **Result Directory Standard**: All Agents write to `save_dir/<task>/<agent>/<mode>/<timestamp>/`, and record the `run_subdir` in `run_config.json` for tracking.

---

## 4. Unified Interface and Code Template

### 4.1 Agent Constructor Signature

```python
class YourAgent:
    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        agent_method: str,
        **kwargs,
    ):
        self.agent_method = agent_method
        self.generator = ...  # Initialize model client
```

> If Reflector/Curator are needed, initialize them here sequentially.

### 4.2 run() Method Skeleton

```python
def run(
    self,
    mode: str,
    train_samples: Optional[List[Dict[str, Any]]],
    val_samples: Optional[List[Dict[str, Any]]],
    test_samples: Optional[List[Dict[str, Any]]],
    data_processor,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    supported = {"online", "eval_only"}  # Adjust as needed
    if mode not in supported:
        raise ValueError(f"{self.agent_method} only supports {supported}")

    task_name = config["task_name"]
    save_dir = config["save_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_subdir = os.path.join(task_name, self.agent_method, mode, timestamp)
    resolved_save_path = os.path.join(save_dir, run_subdir)
    os.makedirs(resolved_save_path, exist_ok=True)

    # Execute main logic (evaluation / training)
    results, error_log = evaluate_test_set(...)

    # Write configuration and results
    with open(os.path.join(resolved_save_path, "run_config.json"), "w") as f:
        json.dump({"config": config, "run_subdir": run_subdir}, f, indent=2)

    with open(os.path.join(resolved_save_path, "test_results.json"), "w") as f:
        json.dump({"test_results": results, "error_log": error_log}, f, indent=2)

    return results
```

### 4.3 Prompt Example (CoT)

```python
prompt_lines = [
    "You are a finance-domain reasoning assistant...",
    "Return JSON with keys reasoning / bullet_ids / final_answer",
    "- reasoning: ...",
    "- bullet_ids: [] when no external knowledge source is referenced",
    "- final_answer: ...",
    "",
    "Question:",
    question_text,
    "",
    "Context:",
    context_text,
    "",
    "Respond strictly in JSON.",
]
```

> Unified output formats can be directly parsed by `utils.extract_answer()`.

---

## 5. StructuredReasoning Entry Point Integration

### 5.1 Command Line Arguments

`StructuredReasoning/run.py` must include the new method in `--agent_method`, for example:

```python
parser.add_argument(
    "--agent_method",
    choices=["ace", "cot", "your_agent"],
    ...
)
```

### 5.2 main() Branching

```python
if args.agent_method == "ace":
    ace_system = ACE(..., agent_method=args.agent_method)
    ace_system.run(...)
elif args.agent_method == "cot":
    cot_agent = ChainOfThoughtAgent(...)
    cot_agent.run(...)
else:
    new_agent = YourAgent(..., agent_method=args.agent_method)
    new_agent.run(...)
```

### 5.3 Shell Scripts

Scripts in `run_scripts/StructuredReasoning/...` are parameterized. New agents can follow this template:

```bash
BENCHMARK_MODULE="StructuredReasoning.run"
BENCHMARK_NAME="StructuredReasoning"
AGENT_METHOD="your_agent"
TASK_NAME="CodeFinQA"
MODE="online"
CONFIG_PATH="StructuredReasoning/data/task_config.json"
SAVE_PATH="results/${BENCHMARK_NAME}_run"
LOG_NAME="${BENCHMARK_NAME}_run_${TASK_NAME}_${AGENT_METHOD}_${MODE}.log"

nohup python -m "${BENCHMARK_MODULE}" \
  --agent_method "${AGENT_METHOD}" \
  --task_name "${TASK_NAME}" \
  --mode "${MODE}" \
  --config_path "${CONFIG_PATH}" \
  --save_path "${SAVE_PATH}" \
  "$@" \
  > "${LOG_NAME}" 2>&1 &
```

---

## 6. DataProcessor and Tool Reuse

- **DataProcessor Responsibilities** (located in `StructuredReasoning/data_processor.py`):
  - `process_task_data`: Converts raw samples into standard fields (context/question/target).
  - `answer_is_correct(pred, target)`: Determines if model output is correct.
  - `evaluate_accuracy(answers, targets)`: Calculates accuracy.
- **Evaluation Utilities**: `utils.evaluate_test_set()` calls `generator.generate()`, parses the output, and submits it to DataProcessor for scoring. All Agents are encouraged to reuse this function for consistent `test_results` formatting.
- **Model Output**: Ensure that the `final_answer` field in the prompt can be extracted by `utils.extract_answer()`; if using a custom format, update the parsing logic accordingly.

---

## 7. Important Considerations

1. **Mode Support**: If `offline` is not supported, explicitly `raise ValueError` to avoid silent failures.
2. **API Client**: It is recommended to use `utils.initialize_clients(api_provider)` for initialization to ensure consistent retry/logging policies with ACE.
3. **Logging and Errors**:
   - ACE uses `logger.py` for fine-grained call logging; lightweight Agents should at least print key events (startup, sample count, accuracy).
   - `timed_llm_call()` has built-in retries and problem request recording.
4. **Result Naming**:
   - Must generate `run_config.json` (with `run_subdir`), `test_results.json`, and if applicable, `final_results.json`, `train_results.json`, etc.
5. **Prompt Constraints**: When no external memory/knowledge is available, clearly state this in the prompt to avoid hallucinations (e.g., CoT `bullet_ids` should always be empty).
6. **Directory Structure**: Every new Agent must follow the `<task>/<agent>/<mode>/<timestamp>` pattern so that result management tools can automatically identify them.

---

## 8. Mode Descriptions (offline / online / eval_only)

- **offline**: A one-time train-then-evaluate process. First, use `train` data to complete all training/fine-tuning, then use `test` data for a single evaluation. The test set does not participate in training.
- **online**: A windowed "test-then-train" cycle. The `test` samples are split into windows (e.g., every N samples or by time slices). Each window is first evaluated using the current model, then immediately used for incremental training/fine-tuning. When moving to the next window, the model accumulates training results from previous windows, maintaining chronological order.
- **eval_only**: Pure evaluation mode. No training is performed; evaluation is conducted using the entire `test` dataset.

---

## 9. Validation Checklist

1. **Command Line Execution**  
   ```bash
   python -m StructuredReasoning.run \
     --agent_method your_agent \
     --task_name FinCode \
     --mode online \
     --save_path results/StructuredReasoning_run \
     --config_path StructuredReasoning/data/task_config.json
   ```
2. **Directory Check**  
   - Verify `results/StructuredReasoning_run/FinCode/your_agent/online/<timestamp>/` exists.
   - Check `run_config.json` for recorded `run_subdir` and correct configuration.
3. **Result Files**  
   - `test_results.json` / `final_results.json` (if applicable) are correctly formatted and readable.
   - Log directories like `detailed_llm_logs/` are created.
4. **Log Output**  
   - Console/log files contain key information such as startup messages, sample counts, and accuracy.
5. **Glue Scripts**  
   - If corresponding shell scripts exist, ensure `nohup` output points to the correct log file.
6. **Assertions**  
   - If an Agent does not support a mode, verify that it throws a friendly error message when run.

Following these steps confirms that the new Agent method is fully integrated into the current StructuredReasoning framework.

---

## 10. Reference Examples

- **ACE**: `Agents/ace/ace.py` (Multi-role collaboration, multi-stage generation, and reflection); suitable for Agents requiring complex memory management.
- **CoT**: `Agents/cot/agent.py` + `generator.py` (Single Prompt Baseline); suitable as a minimal implementation template.

Between these two, you can implement solutions of any complexity as long as they adhere to the conventions in this document for rapid integration into existing CLI, storage, and logging systems.
