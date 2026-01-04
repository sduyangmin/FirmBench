import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


FILENAME_PATTERN = re.compile(r"(?P<entity>[A-Z0-9]+-E)_(?P<year>\d{4})_suppliers\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert factset *_suppliers.csv + *_target_info.json to BizBench JSONL with 3/3/4 splits."
    )
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=default_root / "data" / "factset",
        help="Directory containing *_suppliers.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_root / "data",
        help="Directory to write factset_{train,val,test,all}.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--train_count", type=int, default=3, help="How many files go to train split first.")
    parser.add_argument("--val_count", type=int, default=3, help="How many files go to val split after train.")
    parser.add_argument("--test_count", type=int, default=4, help="How many files go to test split after val.")
    return parser.parse_args()


def clean(value) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    return text if text else "N/A"


def parse_target_info(path: Path) -> Tuple[str, str]:
    match = FILENAME_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")
    return match.group("entity"), match.group("year")


def build_target_context(target_info: Dict) -> str:
    top_segments = target_info.get("top_segments") or target_info.get("self_profile", {}).get("top_segments", [])
    top_segments_str = ", ".join(top_segments) if isinstance(top_segments, list) else clean(top_segments)
    parts = [
        "[目标公司画像]",
        f"目标ID: {clean(target_info.get('target_entity_id') or target_info.get('factset_entity_id'))}",
        f"公司名: {clean(target_info.get('entity_proper_name'))}",
        f"国家: {clean(target_info.get('iso_country'))}",
        f"SIC: {clean(target_info.get('primary_sic_name'))}",
        f"行业: {clean(target_info.get('industry'))}",
        f"业务段: {top_segments_str}",
        f"决策年: {clean(target_info.get('decision_year') or target_info.get('cutoff_date'))}",
    ]
    return "\n".join(parts)


def build_supplier_context(row: Dict[str, str]) -> str:
    parts = [
        "[候选供应商画像]",
        f"供应商ID: {clean(row.get('factset_entity_id'))}",
        f"名称: {clean(row.get('entity_proper_name'))}",
        f"国家: {clean(row.get('iso_country'))}",
        f"SIC: {clean(row.get('primary_sic_name'))}",
        f"行业: {clean(row.get('industry'))}",
        f"业务段: {clean(row.get('top_segments'))}",
        f"单一来源: {clean(row.get('is_single_source'))}",
        (
            "财务指标: "
            f"FF_SALES={clean(row.get('FF_SALES'))}, "
            f"FF_NET_MGN={clean(row.get('FF_NET_MGN'))}, "
            f"FF_DEBT_EQ={clean(row.get('FF_DEBT_EQ'))}, "
            f"FF_ROIC={clean(row.get('FF_ROIC'))}, "
            f"FF_CURR_RATIO={clean(row.get('FF_CURR_RATIO'))}, "
            f"货币={clean(row.get('CURRENCY'))}, "
            f"财报日期={clean(row.get('financial_date'))}"
        ),
        (
            "关系信息: "
            f"revenue_percent={clean(row.get('revenue_percent'))}, "
            f"relationship_start={clean(row.get('relationship_start'))}, "
            f"source={clean(row.get('relationship_source'))}, "
            f"source_label={clean(row.get('source_label'))}"
        ),
    ]
    return "\n".join(parts)


def read_samples(path: Path, target_entity_id: str, year: str, target_info: Dict) -> List[Dict]:
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("ground_truth_label") or "").strip()
            if label == "":
                label = "0"

            supplier_id = (row.get("factset_entity_id") or "").strip()

            context = "\n\n".join(
                [
                    build_target_context(target_info),
                    build_supplier_context(row),
                ]
            )

            sample = {
                "context": context,
                "question": (
                    "基于上述目标公司画像和供应商画像，判断在决策年是否应选择该供应商。"
                    "请只回答 [[1]] 表示选择 或 [[0]] 表示不选择。"
                ),
                "answer": label,
                "target_entity_id": target_entity_id,
                "supplier_entity_id": supplier_id,
                "year": year,
                "target_info": target_info,
                "supplier_raw": row,
                "source_label": row.get("source_label") or "",
                "relationship_source": row.get("relationship_source") or "",
                "revenue_percent": row.get("revenue_percent") or "",
                "is_single_source": row.get("is_single_source") or "",
                "industry": row.get("industry") or "",
                "top_segments": row.get("top_segments") or "",
            }
            samples.append(sample)
    return samples


def assign_splits(
    files: List[Path], train_count: int, val_count: int, test_count: int, seed: int
) -> Dict[str, List[Path]]:
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    for idx, fp in enumerate(shuffled):
        if idx < train_count:
            splits["train"].append(fp)
        elif idx < train_count + val_count:
            splits["val"].append(fp)
        elif idx < train_count + val_count + test_count:
            splits["test"].append(fp)
        else:
            splits["train"].append(fp)
    return splits


def write_jsonl(path: Path, records: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    csv_files = sorted(input_dir.glob("*_suppliers.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_suppliers.csv found under {input_dir}")

    print(f"发现 {len(csv_files)} 个 company-year 文件，开始按 3/3/4 规则切分（seed={args.seed}）")

    splits = assign_splits(csv_files, args.train_count, args.val_count, args.test_count, args.seed)
    for split, files in splits.items():
        print(f"{split}: {len(files)} 个文件")

    split_records: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    all_records: List[Dict] = []

    for split, files in splits.items():
        for fp in files:
            target_entity_id, year = parse_target_info(fp)
            target_info_path = fp.with_name(f"{target_entity_id}_{year}_target_info.json")
            if not target_info_path.exists():
                raise FileNotFoundError(f"Missing target info file: {target_info_path}")
            with target_info_path.open("r", encoding="utf-8") as tf:
                target_info = json.load(tf)

            samples = read_samples(fp, target_entity_id, year, target_info)
            split_records[split].extend(samples)
            all_records.extend(samples)
            print(f"[{split}] {fp.name}: {len(samples)} 条样本")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "factset_train.jsonl", split_records["train"])
    write_jsonl(output_dir / "factset_val.jsonl", split_records["val"])
    write_jsonl(output_dir / "factset_test.jsonl", split_records["test"])
    write_jsonl(output_dir / "factset_all.jsonl", all_records)

    print(
        "写入完成："
        f" train={len(split_records['train'])},"
        f" val={len(split_records['val'])},"
        f" test={len(split_records['test'])},"
        f" all={len(all_records)}"
    )


if __name__ == "__main__":
    main()

