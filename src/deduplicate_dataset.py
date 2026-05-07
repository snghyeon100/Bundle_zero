import argparse
import json
import os
import shutil
from pathlib import Path


INTERACTION_FILES = [
    "bi_full.txt",
    "bi_train.txt",
    "bi_valid_input.txt",
    "bi_valid_gt.txt",
    "bi_test_input.txt",
    "bi_test_gt.txt",
    "ui_full.txt",
]


COPY_FILES = [
    "item_info.json",
]


def read_count(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dedup_line(line):
    values = [int(v) for v in line.strip().split(", ") if v]
    if not values:
        return None, 0, 0

    head = values[0]
    items = values[1:]
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    raw_count = len(items)
    dedup_count = len(deduped)
    out_line = ", ".join(str(v) for v in [head] + deduped)
    return out_line, raw_count, dedup_count


def dedup_interaction_file(src_path, dst_path):
    rows = 0
    raw_pairs = 0
    dedup_pairs = 0
    rows_with_duplicates = 0

    with open(src_path, "r", encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            out_line, raw_count, dedup_count = dedup_line(line)
            if out_line is None:
                continue
            rows += 1
            raw_pairs += raw_count
            dedup_pairs += dedup_count
            if raw_count != dedup_count:
                rows_with_duplicates += 1
            dst.write(out_line + "\n")

    return {
        "rows": rows,
        "raw_pairs": raw_pairs,
        "dedup_pairs": dedup_pairs,
        "removed_pairs": raw_pairs - dedup_pairs,
        "rows_with_duplicates": rows_with_duplicates,
    }


def write_count(src_count, stats, dst_path):
    new_count = dict(src_count)

    if "bi_full.txt" in stats:
        new_count["#B-I"] = stats["bi_full.txt"]["dedup_pairs"]
        if new_count.get("#B", 0):
            new_count["#Avg. I/B"] = new_count["#B-I"] / new_count["#B"]

    if "ui_full.txt" in stats:
        new_count["#U-I"] = stats["ui_full.txt"]["dedup_pairs"]
        if new_count.get("#U", 0):
            new_count["#Avg. I/U"] = new_count["#U-I"] / new_count["#U"]

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(new_count, f, ensure_ascii=False)

    return new_count


def maybe_create_image_link(src_dir, dst_dir):
    src_images = src_dir / "images"
    dst_images = dst_dir / "images"
    if not src_images.exists() or dst_images.exists():
        return "skipped"

    try:
        os.symlink(src_images, dst_images, target_is_directory=True)
        return "symlink"
    except OSError:
        return "skipped"


def main():
    parser = argparse.ArgumentParser(description="Create a de-duplicated copy of a bundle dataset.")
    parser.add_argument("--source", default="datasets/pog", help="Source dataset directory.")
    parser.add_argument("--target", default="datasets/pog_dedup", help="Target dataset directory.")
    parser.add_argument("--link-images", action="store_true", help="Try to symlink images directory instead of copying it.")
    args = parser.parse_args()

    src_dir = Path(args.source)
    dst_dir = Path(args.target)
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for name in INTERACTION_FILES:
        src_path = src_dir / name
        if not src_path.exists():
            print(f"[skip] Missing interaction file: {src_path}")
            continue
        dst_path = dst_dir / name
        stats[name] = dedup_interaction_file(src_path, dst_path)

    for name in COPY_FILES:
        src_path = src_dir / name
        if src_path.exists():
            shutil.copy2(src_path, dst_dir / name)

    src_count_path = src_dir / "count.json"
    if src_count_path.exists():
        src_count = read_count(src_count_path)
        new_count = write_count(src_count, stats, dst_dir / "count.json")
    else:
        new_count = {}

    image_status = maybe_create_image_link(src_dir, dst_dir) if args.link_images else "not_requested"

    print("=" * 72)
    print(f"Source: {src_dir}")
    print(f"Target: {dst_dir}")
    print("=" * 72)
    for name, info in stats.items():
        print(
            f"{name}: rows={info['rows']}, raw_pairs={info['raw_pairs']}, "
            f"dedup_pairs={info['dedup_pairs']}, removed={info['removed_pairs']}, "
            f"rows_with_duplicates={info['rows_with_duplicates']}"
        )
    if new_count:
        print("-" * 72)
        print(f"count.json: #B-I={new_count.get('#B-I')}, #U-I={new_count.get('#U-I')}, "
              f"#Avg. I/B={new_count.get('#Avg. I/B')}, #Avg. I/U={new_count.get('#Avg. I/U')}")
    print(f"images link: {image_status}")
    print("=" * 72)


if __name__ == "__main__":
    main()
