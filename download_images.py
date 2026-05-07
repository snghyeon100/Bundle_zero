import argparse
import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yaml

from src.dataset import BundleZeroShotDataset


VALID_EXTS = ("jpg", "jpeg", "png", "gif", "webp")


def infer_dataset_from_csv(csv_path):
    parts = os.path.normpath(csv_path).split(os.sep)
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    base = os.path.basename(csv_path).lower()
    for dataset in ("spotify_sparse", "spotify", "pog_dense", "pog_dedup", "pog"):
        if dataset in base:
            return dataset
    return None


def parse_list(value):
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(str(value))
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []


def item_image_exists(item_id, save_dir):
    item_id = str(item_id)
    if not os.path.exists(save_dir):
        return False

    for ext in VALID_EXTS:
        if os.path.exists(os.path.join(save_dir, f"{item_id}.{ext}")):
            return True

    return any(name.startswith(f"{item_id}.") for name in os.listdir(save_dir))


def collect_item_ids_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    unique_item_ids = set()
    for _, row in df.iterrows():
        unique_item_ids.update(parse_list(row.get("input_indices", [])))
        unique_item_ids.update(parse_list(row.get("candidate_indices", [])))
    return {str(int(item_id)) for item_id in unique_item_ids if str(item_id) != "nan"}, len(df)


def collect_item_ids_from_config(conf):
    dataset_name = conf["dataset"]
    dataset = BundleZeroShotDataset(conf)

    hard_negative_path = os.path.join(
        conf.get("data_path", "./datasets"),
        dataset_name,
        f"hard_negative_samples_{dataset_name}.json",
    )
    if conf.get("use_hard_negative", False) and os.path.exists(hard_negative_path):
        print(f">>> Loading samples from {hard_negative_path}")
        with open(hard_negative_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        print(">>> Generating evaluation samples from config...")
        samples = dataset.get_eval_samples()

    unique_item_ids = set()
    for sample in samples:
        unique_item_ids.update(sample.get("input_indices", []))
        unique_item_ids.update(sample.get("candidate_indices", []))
    return {str(int(item_id)) for item_id in unique_item_ids}, len(samples)


def normalize_url(url):
    if not url:
        return url
    if url.startswith("//"):
        return "https:" + url
    return url


def extension_from_url(url):
    ext = url.split(".")[-1].split("?")[0].split("#")[0].lower()
    return ext if ext in VALID_EXTS else "jpg"


def download_image(item_id, url, save_dir, overwrite=False):
    if not url:
        return item_id, False, "No URL"

    if item_image_exists(item_id, save_dir) and not overwrite:
        return item_id, True, "Already exists"

    url = normalize_url(url)
    ext = extension_from_url(url)
    save_path = os.path.join(save_dir, f"{item_id}.{ext}")

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Referer": "https://www.taobao.com/",
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type.lower():
            return item_id, False, f"Non-image response: {content_type}"

        with open(save_path, "wb") as f:
            f.write(response.content)
        return item_id, True, "Downloaded"
    except Exception as exc:
        return item_id, False, str(exc)


def build_download_tasks(unique_item_ids, item_info, save_dir, overwrite=False):
    tasks = []
    existing = []
    no_url = []

    for item_id in sorted(unique_item_ids, key=lambda x: int(x)):
        if item_image_exists(item_id, save_dir) and not overwrite:
            existing.append(item_id)
            continue

        info = item_info.get(str(item_id), {})
        pic_url = info.get("pic") or info.get("pic_url")
        if pic_url:
            tasks.append((str(item_id), pic_url))
        else:
            no_url.append(str(item_id))

    return tasks, existing, no_url


def save_reports(dataset_name, failed_rows, no_url_items, missing_after):
    report_dir = os.path.join("results", dataset_name)
    os.makedirs(report_dir, exist_ok=True)

    if failed_rows:
        failed_path = os.path.join(report_dir, "image_download_failed.csv")
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False, encoding="utf-8-sig")
        print(f">>> Saved failed download report: {failed_path}")

    if no_url_items:
        no_url_path = os.path.join(report_dir, "image_download_no_url.csv")
        pd.DataFrame({"item_id": no_url_items}).to_csv(no_url_path, index=False, encoding="utf-8-sig")
        print(f">>> Saved no-url report: {no_url_path}")

    missing_path = os.path.join(report_dir, "image_required_missing_after_download.csv")
    pd.DataFrame({"item_id": missing_after}).to_csv(missing_path, index=False, encoding="utf-8-sig")
    print(f">>> Saved missing-after report: {missing_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download item images for LLM-ZeroShot experiments.")
    parser.add_argument(
        "--csv",
        default="",
        help=(
            "Optional result CSV. When provided, item ids are read directly from "
            "input_indices and candidate_indices, avoiding config/seed mismatch."
        ),
    )
    parser.add_argument("--dataset", default="", help="Dataset name. Inferred from --csv or config if omitted.")
    parser.add_argument("--data_path", default="", help="Dataset root. Defaults to config.yaml data_path.")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent download workers.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload even if item_id.* already exists.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(">>> Loading configuration...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    data_path = args.data_path or conf.get("data_path", "./datasets")
    dataset_name = args.dataset or (infer_dataset_from_csv(args.csv) if args.csv else None) or conf["dataset"]

    if args.csv:
        print(f">>> Loading required item ids directly from CSV: {args.csv}")
        unique_item_ids, sample_count = collect_item_ids_from_csv(args.csv)
        print(f"Total evaluated rows in CSV: {sample_count}")
    else:
        conf["dataset"] = dataset_name
        conf["data_path"] = data_path
        unique_item_ids, sample_count = collect_item_ids_from_config(conf)
        print(f"Total evaluated samples from config: {sample_count}")

    print(f"Dataset: {dataset_name}")
    print(f"Total unique items required: {len(unique_item_ids)}")

    info_path = os.path.join(data_path, dataset_name, "item_info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        item_info = json.load(f)

    save_dir = os.path.join(data_path, dataset_name, "images")
    os.makedirs(save_dir, exist_ok=True)
    print(f">>> Image save directory: {save_dir}")

    tasks, existing, no_url_items = build_download_tasks(
        unique_item_ids, item_info, save_dir, overwrite=args.overwrite
    )
    print(f"Already existing images: {len(existing)}")
    print(f"Missing URL/item_info entries: {len(no_url_items)}")
    print(f"Found valid URLs to download: {len(tasks)}")

    success_count = 0
    exist_count = 0
    fail_count = 0
    failed_rows = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_image, item_id, url, save_dir, args.overwrite)
            for item_id, url in tasks
        ]

        for i, future in enumerate(as_completed(futures)):
            item_id, success, msg = future.result()
            if success:
                if msg == "Already exists":
                    exist_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
                failed_rows.append({"item_id": item_id, "reason": msg})

            if (i + 1) % 200 == 0 or (i + 1) == len(futures):
                print(
                    f"Progress: [{i + 1}/{len(futures)}] | "
                    f"Downloaded: {success_count} | "
                    f"Skipped(Exists): {exist_count} | Failed: {fail_count}"
                )

    missing_after = [
        item_id
        for item_id in sorted(unique_item_ids, key=lambda x: int(x))
        if not item_image_exists(item_id, save_dir)
    ]
    print(f">>> Required images still missing: {len(missing_after)}")
    save_reports(dataset_name, failed_rows, no_url_items, missing_after)
    print(">>> All download tasks completed!")


if __name__ == "__main__":
    main()
