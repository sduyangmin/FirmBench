#!/usr/bin/env python3
import re
import json
from typing import Dict, Any, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from a LaTeX \\boxed{...} expression."""
    pattern = r"\\boxed\{"
    match = re.search(pattern, text)
    if not match:
        return None
    start = match.end() - 1
    brace_count = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]
        i += 1
    return None


def extract_answer(response: str) -> str:
    """Extract final answer from model response."""
    try:
        parsed = json.loads(response)
        return str(parsed.get('final_answer', 'No final answer found'))
    except Exception:
        # Fallbacks
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            return matches[-1]
        matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
        if matches:
            return matches[-1]
        matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
        if matches:
            return matches[-1]
        matches = re.findall(r"[\"']final_answer[\"']\s*:\s*([^,}]+)", response)
        if matches:
            ans = matches[-1].strip()
            ans = re.sub(r"[,}]*$", "", ans)
            return ans
        final_answer_pattern = r"[Tt]he final answer is:?\s*\$?\\\\boxed\{"
        match = re.search(final_answer_pattern, response)
        if match:
            remaining_text = response[match.start():]
            boxed_content = extract_boxed_content(remaining_text)
            if boxed_content:
                return boxed_content
        matches = re.findall(r"[Tt]he final answer is:?\s*([^\n.]+)", response)
        if matches:
            ans = matches[-1].strip()
            ans = re.sub(r"^\$?\\\\boxed\{([^}]+)\}\$?$", r"\1", ans)
            ans = ans.replace('$', '').strip()
            if ans:
                return ans
        return 'No final answer found'


enc = tiktoken.get_encoding('cl100k_base')


def count_tokens(prompt: str) -> int:
    return len(enc.encode(prompt))


def evaluate_single_test_sample(args_tuple, data_processor) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Evaluate a single test sample (generation + correctness). No memory write-back here."""
    (i, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode) = args_tuple
    try:
        context = task_dict.get('context', '')
        question = task_dict.get('question', '')
        target = task_dict.get('target', '')
        call_id = f"test_eval_{i}"

        gen_response, bullet_ids, call_info = generator.generate(
            question=question,
            playbook=playbook,
            context=context,
            reflection='(empty)',
            use_json_mode=use_json_mode,
            call_id=call_id,
            log_dir=log_dir,
        )

        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)

        return {
            'index': i,
            'question': question,
            'context': context,
            'final_answer': final_answer,
            'target': target,
            'is_correct': is_correct,
            'call_id': call_id,
            'success': True,
        }, None
    except Exception as e:
        return None, f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}"


def evaluate_test_set(
    agent,
    data_processor,
    generator,
    playbook,
    test_samples: List[Dict[str, Any]],
    *,
    max_tokens: int = 4096,
    log_dir: Optional[str] = None,
    max_workers: int = 20,
    use_json_mode: bool = False,
    mode: str = 'eval_only',
    batch_size: int = 20,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Batch-wise online evaluation for StructuredReasoning.

    - In 'eval_only' mode: evaluate all samples, do NOT write to memory.
    - In 'online' mode: evaluate in batches; after each batch completes, write the batch's
      (question, context, prediction, ground_truth, correctness) into agent memory.

    This avoids within-batch GT leakage while still enabling inter-batch adaptation.
    """
    if mode not in {'online', 'eval_only'}:
        raise ValueError(f"mode must be 'online' or 'eval_only', got: {mode}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got: {batch_size}")

    total_n = len(test_samples)
    print(f"\n{'=' * 40}")
    print(f"EVALUATING TEST SET - {total_n} samples, {max_workers} workers, batch_size={batch_size}, mode={mode}")
    print(f"{'=' * 40}")

    results_accum = {
        'correct': 0,
        'total': 0,
        'no_answer': 0,
        'answers': [],
        'targets': [],
        'errors': [],
    }

    def eval_wrapper(args_tuple):
        return evaluate_single_test_sample(args_tuple, data_processor)

    # process in batches (deterministic by index order)
    for batch_start in range(0, total_n, batch_size):
        batch_end = min(total_n, batch_start + batch_size)
        batch_samples = test_samples[batch_start:batch_end]
        args_list = [
            (batch_start + j, sample, generator, playbook, max_tokens, log_dir, use_json_mode)
            for j, sample in enumerate(batch_samples)
        ]

        batch_successful: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {executor.submit(eval_wrapper, args): args for args in args_list}
            for future in as_completed(future_to_args):
                result, error = future.result()
                if error:
                    print(error)
                    continue
                if result and result.get('success'):
                    batch_successful.append(result)

        # sort to keep stable commit order and stable logs
        batch_successful.sort(key=lambda r: r['index'])

        # update metrics
        for r in batch_successful:
            results_accum['correct'] += (1 if r['is_correct'] else 0)
            results_accum['total'] += 1
            results_accum['answers'].append(r['final_answer'])
            results_accum['targets'].append(r['target'])
            if r['final_answer'] == 'No final answer found':
                results_accum['no_answer'] += 1
            if not r['is_correct']:
                results_accum['errors'].append({
                    'index': r['index'],
                    'prediction': r['final_answer'],
                    'ground_truth': r['target'],
                })

        # commit this batch into memory (online only)
        if mode == 'online' and hasattr(agent, 'add_memory'):
            for r in batch_successful:
                try:
                    agent.add_memory(
                        question=r.get('question', ''),
                        context=r.get('context', ''),
                        response=r.get('final_answer', ''),
                        target=r.get('target', ''),
                        is_correct=bool(r.get('is_correct', False)),
                        call_id=r.get('call_id', f"test_eval_{r.get('index', -1)}"),
                    )
                except Exception as e:
                    print(f"[WARN] memory write-back failed for index={r.get('index')}: {type(e).__name__}: {e}")

        curr_acc = results_accum['correct'] / results_accum['total'] if results_accum['total'] else 0.0
        print(f"Batch {batch_start//batch_size + 1}: {batch_start}-{batch_end-1} done. Running acc={curr_acc:.3f}")

    # final accuracy
    if results_accum['answers'] and results_accum['targets']:
        accuracy = data_processor.evaluate_accuracy(results_accum['answers'], results_accum['targets'])
        final_results = {
            'accuracy': accuracy,
            'correct': results_accum['correct'],
            'total': results_accum['total'],
            'no_answer': results_accum['no_answer'],
        }
        error_logs = {'accuracy': accuracy, 'errors': results_accum['errors']}
        print(f"\nFinal Accuracy: {accuracy:.3f} ({results_accum['correct']}/{results_accum['total']})")
    else:
        final_results = {'accuracy': 0.0, 'correct': 0, 'total': 0, 'no_answer': 0}
        error_logs = {}
        print("\nNo valid results!")

    return final_results, error_logs
