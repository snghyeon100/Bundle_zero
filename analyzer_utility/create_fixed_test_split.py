import argparse
from pathlib import Path

import numpy as np
import yaml


def read_bundle_file(path):
    rows = {}
    order = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) < 2:
                continue
            bundle_id = vals[0]
            if bundle_id not in rows:
                order.append(bundle_id)
            rows[bundle_id] = vals[1:]
    return rows, order


def unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def choose_gt_item(bundle_id, gt_items, selection, seed):
    if selection == "first":
        return gt_items[0]
    if selection != "random":
        raise ValueError(f"Unsupported fixed_test_gt_selection={selection!r}. Use 'random' or 'first'.")
    rng = np.random.default_rng(int(seed) + int(bundle_id))
    return int(rng.choice(gt_items))


def write_bundle_file(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for bundle_id, items in rows:
            f.write(", ".join(str(v) for v in [bundle_id] + items) + "\n")


def create_fixed_test_split(dataset_dir, selection="random", seed=45,
                            input_name="bi_test_input.txt", gt_name="bi_test_gt.txt",
                            output_input_name="bi_fix_test_input.txt",
                            output_gt_name="bi_fix_test_gt.txt"):
    dataset_dir = Path(dataset_dir)
    input_rows, input_order = read_bundle_file(dataset_dir / input_name)
    gt_rows, gt_order = read_bundle_file(dataset_dir / gt_name)

    fixed_input_rows = []
    fixed_gt_rows = []
    skipped = []
    for bundle_id in gt_order:
        gt_items = unique_preserve_order(gt_rows.get(bundle_id, []))
        if not gt_items:
            skipped.append(bundle_id)
            continue
        input_items = unique_preserve_order(input_rows.get(bundle_id, []))
        heldout = choose_gt_item(bundle_id, gt_items, selection, seed)
        full_items = unique_preserve_order(input_items + gt_items)
        fixed_input = [item for item in full_items if item != heldout]
        if not fixed_input:
            skipped.append(bundle_id)
            continue
        fixed_input_rows.append((bundle_id, fixed_input))
        fixed_gt_rows.append((bundle_id, [heldout]))

    write_bundle_file(dataset_dir / output_input_name, fixed_input_rows)
    write_bundle_file(dataset_dir / output_gt_name, fixed_gt_rows)
    return {
        "dataset_dir": str(dataset_dir),
        "bundles": len(fixed_gt_rows),
        "skipped": len(skipped),
        "input_pairs": sum(len(items) for _, items in fixed_input_rows),
        "gt_pairs": sum(len(items) for _, items in fixed_gt_rows),
        "output_input": str(dataset_dir / output_input_name),
        "output_gt": str(dataset_dir / output_gt_name),
    }


def main():
    parser = argparse.ArgumentParser(description="Create bundle-level fixed test split files with one GT item per bundle.")
    parser.add_argument("--config", default="config.yaml", help="Config file to read dataset/data_path/options from.")
    parser.add_argument("--dataset", default="", help="Dataset name override. Defaults to config dataset.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    dataset = args.dataset or conf["dataset"]
    dataset_dir = Path(conf.get("data_path", "./datasets")) / dataset
    stats = create_fixed_test_split(
        dataset_dir,
        selection=conf.get("fixed_test_gt_selection", "random"),
        seed=conf.get("fixed_test_gt_seed", conf.get("seed", 45)),
        output_input_name=conf.get("test_input_file", "bi_fix_test_input.txt"),
        output_gt_name=conf.get("test_gt_file", "bi_fix_test_gt.txt"),
    )

    print("=" * 72)
    print(f"Dataset: {dataset}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=" * 72)


if __name__ == "__main__":
    main()
