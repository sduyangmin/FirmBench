import ast
import json
import math
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Domain mapping（按照用户确认的分组，FormulaEval 仅在 Knowledge understand）
DOMAIN_MAP: Dict[str, List[str]] = {
    "Finance Reasoning": ["FinCode", "CodeFinQA", "CodeTAT-QA", "formula"],
    "Span extraction": ["ConvFinQA", "SEC-NUM", "TAT-QA"],
    "Knowledge understand": ["finer", "FinKnow", "FormulaEval"],
}

# 便于通过 task_name 反查 domain
TASK_TO_DOMAIN: Dict[str, str] = {
    task: domain for domain, tasks in DOMAIN_MAP.items() for task in tasks
}


def _count_numbers(text: str) -> int:
    return len(re.findall(r"-?\d+(?:\.\d+)?", text))


def _safe_ast_complexity(code: str) -> int:
    if not code:
        return 0
    try:
        tree = ast.parse(code)
        op_nodes = (
            ast.BinOp,
            ast.BoolOp,
            ast.Compare,
            ast.Call,
            ast.If,
            ast.For,
            ast.While,
            ast.FunctionDef,
            ast.Assign,
            ast.Return,
        )
        return sum(1 for n in ast.walk(tree) if isinstance(n, op_nodes))
    except Exception:
        # 退化为行数
        return len(code.splitlines())


def _text_length(obj) -> int:
    if obj is None:
        return 0
    if isinstance(obj, (dict, list)):
        try:
            obj = json.dumps(obj)
        except Exception:
            obj = str(obj)
    return len(str(obj))


def _count_program_lines(program: str) -> int:
    """统计程序中有效代码行数（非空、非纯注释）"""
    if not program:
        return 0
    lines = program.strip().split('\n')
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            count += 1
    return count


def _count_table_cells(context: str) -> int:
    """统计 markdown 表格的单元格数量（行数×列数）"""
    if not context or '|' not in context:
        return 0
    lines = context.split('\n')
    table_rows = []
    for line in lines:
        if '|' in line:
            # 简单的 markdown 表格识别
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells and not all(c.replace('-', '').replace(':', '').strip() == '' for c in cells):
                # 排除分隔行（如 | --- | --- |）
                table_rows.append(len(cells))
    
    if not table_rows:
        return 0
    # 返回行数 × 平均列数
    avg_cols = sum(table_rows) / len(table_rows)
    return int(len(table_rows) * avg_cols)


def _find_answer_position(answer: str, context: str) -> float:
    """
    计算答案在 context 中的相对位置（0.0-1.0）
    返回 0.0 表示在最前面，1.0 表示在最后面
    找不到返回 0.5（中性值）
    """
    if not answer or not context:
        return 0.5
    
    answer_str = str(answer).strip().lower()
    context_str = str(context).lower()
    
    # 尝试找到答案在 context 中的位置
    pos = context_str.find(answer_str)
    if pos == -1:
        # 如果精确匹配失败，尝试分词匹配（处理多词答案）
        words = answer_str.split()
        if words:
            first_word_pos = context_str.find(words[0])
            if first_word_pos != -1:
                pos = first_word_pos
    
    if pos == -1:
        return 0.5  # 找不到，返回中性值
    
    # 计算相对位置
    context_len = len(context_str)
    if context_len == 0:
        return 0.5
    
    return pos / context_len


def _is_answer_in_table(answer: str, context: str) -> bool:
    """
    判断答案是否在 markdown 表格中
    """
    if not answer or not context or '|' not in context:
        return False
    
    answer_str = str(answer).strip().lower()
    lines = context.split('\n')
    
    for line in lines:
        if '|' in line:
            # 检查这一行是否包含答案
            if answer_str in line.lower():
                return True
    
    return False


def _count_table_rows(context: str) -> int:
    """
    统计 context 中 markdown 表格的行数（不含分隔行）
    """
    if not context or '|' not in str(context):
        return 0
    lines = str(context).split('\n')
    count = 0
    for line in lines:
        if '|' in line:
            # 排除分隔行（如 | --- | --- |）
            if not line.strip().startswith('|---') and '---' not in line.replace(' ', ''):
                count += 1
    return count


def _count_program_variables(program: str) -> int:
    """
    统计程序中的变量数量（赋值语句数量）
    """
    if not program:
        return 0
    # 统计 '=' 但排除 '=='
    return program.count('=') - program.count('==')


def _count_question_complex_words(question: str) -> int:
    """
    统计问题中的复杂关键词数量
    这些词通常让问题更明确，反而降低难度
    """
    if not question:
        return 0
    q_lower = str(question).lower()
    complex_words = ['ratio', 'percentage', 'compare', 'difference', 'average', 'total', 'calculate', 'sum']
    return sum(1 for word in complex_words if word in q_lower)


def score_finance_reasoning(sample: dict) -> float:
    """
    Finance Reasoning 打分：多维度综合评估
    基于实际错误率分析的四个维度：
    - 文本长度（0.3）：基础维度，文本越长越难
    - 表格复杂度（0.3）：强正相关，表格行数越多越难
    - 变量数惩罚（0.2）：U型曲线，2-3个变量最简单，6+个最难
    - 问题明确度（0.2）：反向，包含明确指令词（ratio/average等）反而简单
    """
    question = sample.get("question", "")
    context = sample.get("context", "") or ""
    program = sample.get("program", "") or ""
    
    # 1. 文本长度（基础维度，权重 0.3）
    text_len = _text_length(question) + _text_length(context)
    text_score = math.log1p(text_len)
    
    # 2. 表格复杂度（强正相关，权重 0.3）
    table_rows = _count_table_rows(context)
    table_score = table_rows  # 行数越多越难
    
    # 3. 变量数惩罚（U型曲线，权重 0.2）
    var_count = _count_program_variables(program)
    if 2 <= var_count <= 3:
        var_penalty = -2  # 简单区间，降低分数
    elif var_count >= 6:
        var_penalty = 3   # 困难区间，提高分数
    else:
        var_penalty = 0
    
    # 4. 问题复杂度词（反向，权重 0.2）
    # 包含明确指令词的问题反而简单（任务明确）
    complex_count = _count_question_complex_words(question)
    clarity_penalty = -complex_count
    
    return 0.3 * text_score + 0.3 * table_score + 0.2 * var_penalty + 0.2 * clarity_penalty


def score_span_extraction(sample: dict) -> float:
    """
    Span extraction 打分：加权文本长度方案
    score = log1p(0.3 * question长度 + 0.7 * context长度)
    context 权重更高，因为答案定位主要在 context 中进行
    """
    question = sample.get("question", "")
    context = sample.get("context", "") or ""
    
    question_len = _text_length(question)
    context_len = _text_length(context)
    
    weighted_len = 0.3 * question_len + 0.7 * context_len
    
    return math.log1p(weighted_len)


def score_knowledge_understand(sample: dict, task_name: str) -> float:
    question = sample.get("question", "")
    context = sample.get("context", "") or ""
    text_len = _text_length(question) + _text_length(context)
    option_count = len(sample.get("options", []) or [])
    label_count = 0
    for key in ("labels", "label", "sentiments"):
        if key in sample:
            val = sample[key]
            if isinstance(val, list):
                label_count = max(label_count, len(val))
            else:
                try:
                    label_count = max(label_count, len(json.loads(str(val))))
                except Exception:
                    label_count = max(label_count, 1)
    prog_complex = _safe_ast_complexity(sample.get("program", "")) if task_name == "FormulaEval" else 0
    secondary = option_count or label_count or prog_complex
    # 去掉数字权重，强调文本长度 + 次级复杂度
    return 0.6 * math.log1p(text_len) + 0.4 * secondary


def compute_scores(task_name: str, samples: List[dict]) -> Tuple[List[float], str]:
    domain = TASK_TO_DOMAIN.get(task_name)
    if not domain:
        raise ValueError(f"未找到 task '{task_name}' 的 domain 映射，DOMAIN_MAP 中未定义。")
    scores: List[float] = []
    for sample in samples:
        if domain == "Finance Reasoning":
            score = score_finance_reasoning(sample)
        elif domain == "Span extraction":
            score = score_span_extraction(sample)
        else:
            score = score_knowledge_understand(sample, task_name)
        scores.append(score)
    return scores, domain


def assign_difficulty(scores: List[float], domain: str = "") -> Tuple[List[str], Dict[str, float]]:
    """
    根据分数分配难度等级
    所有域统一使用 40% / 40% / 20% 分桶策略
    """
    if not scores:
        return [], {}
    sorted_scores = sorted(scores)
    def _quantile(p: float) -> float:
        if len(sorted_scores) == 1:
            return sorted_scores[0]
        idx = p * (len(sorted_scores) - 1)
        low = int(math.floor(idx))
        high = int(math.ceil(idx))
        if low == high:
            return sorted_scores[low]
        return sorted_scores[low] + (sorted_scores[high] - sorted_scores[low]) * (idx - low)

    # 统一使用 40% / 40% / 20% 分桶
    q1 = _quantile(0.4)
    q2 = _quantile(0.8)
    
    thresholds = {"q1": q1, "q2": q2}

    buckets: List[str] = []
    for s in scores:
        if s <= q1:
            buckets.append("easy")
        elif s <= q2:
            buckets.append("middle")
        else:
            buckets.append("hard")
    return buckets, thresholds


def _load_test_results(test_results_path: str) -> Tuple[float, int, List[dict]]:
    with open(test_results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    errors: List[Dict[str, Any]] = []
    total = 0
    accuracy = 0.0

    # Some runners (e.g., certain GEPA offline) store final results here
    if "final_test_results" in payload and isinstance(payload["final_test_results"], dict):
        ftr = payload["final_test_results"]
        total = ftr.get("total", total) or total
        accuracy = ftr.get("accuracy", accuracy) or accuracy
        if isinstance(ftr.get("errors"), list):
            errors = ftr.get("errors", []) or []

    # GEPA: window_results 内部有 errors，index 为窗口内索引，需要加 start
    if "test_results" in payload and isinstance(payload["test_results"], dict):
        tr = payload["test_results"]
        total = tr.get("total", total)
        accuracy = tr.get("accuracy", accuracy) or accuracy
        if "window_results" in tr and isinstance(tr["window_results"], list):
            win_errs: List[Dict[str, Any]] = []
            for w in tr["window_results"]:
                start = w.get("start", 0)
                for err in w.get("errors", []) or []:
                    new_err = dict(err)
                    if "index" in new_err and "global_index" not in new_err:
                        new_err["global_index"] = start + int(new_err["index"])
                    win_errs.append(new_err)
            if win_errs:
                errors = win_errs

        # 如果 window_results 没有 errors，再回退到 tr.errors
        if not errors and "errors" in tr:
            errors = tr.get("errors", []) or []

    # Common patterns（ACE/cot/dc/self_refine/reflexion）
    if not errors and "test_error_log" in payload and isinstance(payload["test_error_log"], dict):
        errors = payload["test_error_log"].get("errors", []) or []
        accuracy = payload["test_error_log"].get("accuracy", accuracy)
    if not errors and "error_log" in payload and isinstance(payload["error_log"], dict):
        errors = payload["error_log"].get("errors", []) or []
        accuracy = payload["error_log"].get("accuracy", accuracy)
    if not errors and "errors" in payload and isinstance(payload["errors"], list):
        errors = payload["errors"]

    if "test_accuracy" in payload:
        accuracy = payload.get("test_accuracy", accuracy)
    if "total" in payload and (not total or total == 0):
        total = payload.get("total", total) or total

    total = int(total) if total else 0
    return accuracy or 0.0, total, errors


def _build_error_index_set(errors: List[dict]) -> Tuple[set, Dict[int, dict]]:
    idx_set = set()
    idx_to_error: Dict[int, dict] = {}
    for err in errors:
        key = None
        if "global_index" in err:
            key = int(err["global_index"])
        elif "index" in err:
            key = int(err["index"])
        if key is not None:
            idx_set.add(key)
            idx_to_error[key] = err
    return idx_set, idx_to_error


def aggregate_by_difficulty(
    difficulties: List[str],
    errors: List[dict],
    total_samples: int,
) -> Dict[str, dict]:
    idx_set, idx_to_error = _build_error_index_set(errors)
    buckets = {
        "easy": {"total": 0, "errors": []},
        "middle": {"total": 0, "errors": []},
        "hard": {"total": 0, "errors": []},
    }

    for idx in range(total_samples):
        bucket = difficulties[idx] if idx < len(difficulties) else "middle"
        buckets.setdefault(bucket, {"total": 0, "errors": []})
        buckets[bucket]["total"] += 1
        if idx in idx_set:
            buckets[bucket]["errors"].append(idx_to_error[idx])

    for bucket in buckets.values():
        total = bucket["total"]
        err_cnt = len(bucket["errors"])
        correct = max(total - err_cnt, 0)
        bucket["correct"] = correct
        bucket["accuracy"] = correct / total if total else 0.0
    return buckets


def _domain_slug(domain: str) -> str:
    return domain.replace(" ", "_")


def _latest_run_dir(root: str, task_name: str, agent_method: str, mode: str) -> str:
    candidate_root = os.path.join(root, task_name, agent_method, mode)
    if not os.path.exists(candidate_root):
        return ""
    subdirs = [
        os.path.join(candidate_root, d)
        for d in os.listdir(candidate_root)
        if os.path.isdir(os.path.join(candidate_root, d))
    ]
    if not subdirs:
        return ""
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


def _load_task_samples(task_name: str, config_path: str) -> List[dict]:
    """
    加载原始样本（不经过 DataProcessor）用于计算难度分数
    因为需要 program、answer 等原始字段
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if task_name not in cfg or "test_data" not in cfg[task_name]:
        raise ValueError(f"{task_name} 在配置文件中缺少 test_data")
    data_path = cfg[task_name]["test_data"]
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    # 直接返回原始样本，不经过 DataProcessor
    # 因为计算难度需要 program、answer 等字段
    return samples


def convert_domain_results(
    domain: str,
    agent_method: str,
    mode: str,
    config_path: str,
    results_root: str,
) -> Optional[str]:
    # 规范化 domain 名称
    domain_name = domain.replace("_", " ")
    if domain_name not in DOMAIN_MAP:
        raise ValueError(f"未知 domain: {domain}. 可选: {list(DOMAIN_MAP.keys())}")

    tasks = DOMAIN_MAP[domain_name]
    all_samples: List[dict] = []
    all_scores: List[float] = []
    all_difficulties: List[str] = []
    all_errors: List[dict] = []
    task_spans = []
    total_offset = 0
    total_samples = 0

    # 收集每个任务最新的运行目录
    run_dirs = {}
    for task in tasks:
        rd = _latest_run_dir(results_root, task, agent_method, mode)
        if rd:
            run_dirs[task] = rd

    missing = [t for t in tasks if t not in run_dirs]
    if missing:
        raise ValueError(f"[EasyHard] domain={domain} 缺少任务结果: {missing}，请先完成这些任务的运行后再转换。根目录={results_root}")

    # 对每个任务加载样本和结果
    for task in tasks:
        samples = _load_task_samples(task, config_path)
        scores, _ = compute_scores(task, samples)
        all_samples.extend(samples)
        all_scores.extend(scores)

        acc, total_from_file, errors = _load_test_results(os.path.join(run_dirs[task], "test_results.json"))
        sample_count = len(samples)
        total = sample_count  # 以样本条数为准，避免结果文件 total 异常
        # 偏移错误索引
        for e in errors:
            if "global_index" in e:
                e["global_index"] = total_offset + int(e["global_index"])
            elif "index" in e:
                e["index"] = total_offset + int(e["index"])
        all_errors.extend(errors)
        task_spans.append({"task": task, "start": total_offset, "count": sample_count, "accuracy": acc, "run_dir": run_dirs[task]})
        total_offset += sample_count
        total_samples += total

    if not all_samples:
        raise ValueError(f"[EasyHard] domain={domain} 无可用样本。")

    difficulties, thresholds = assign_difficulty(all_scores, domain_name)
    bucket_stats = aggregate_by_difficulty(difficulties, all_errors, total_samples)
    correct_overall = total_samples - sum(len(b["errors"]) for b in bucket_stats.values())
    overall_accuracy = correct_overall / total_samples if total_samples else 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    domain_dir = os.path.join(results_root, "easyhard_mode", _domain_slug(domain_name))

    for diff, stats in bucket_stats.items():
        diff_dir = os.path.join(domain_dir, diff, agent_method, mode, timestamp)
        os.makedirs(diff_dir, exist_ok=True)
        payload = {
            "domain": domain_name,
            "difficulty": diff,
            "overall_accuracy": overall_accuracy,  # 域内总体
            "total_samples": total_samples,
            "bucket_accuracy": stats.get("accuracy", 0.0),
            "bucket_total": stats.get("total", 0),
            "bucket_correct": stats.get("correct", 0),
            "thresholds": thresholds,
            "bucket": stats,
            "tasks": task_spans,
            "source_runs": run_dirs,
        }
        with open(os.path.join(diff_dir, "easyhard_results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(os.path.join(diff_dir, "difficulty_assignments.jsonl"), "w", encoding="utf-8") as f:
            for idx, (score, diff_name) in enumerate(zip(all_scores, difficulties)):
                if diff_name != diff:
                    continue
                f.write(json.dumps({"index": idx, "difficulty": diff_name, "score": score}, ensure_ascii=False) + "\n")

    print(f"[EasyHard] domain={domain_name} 聚合完成，输出根目录: {domain_dir}")
    return domain_dir

