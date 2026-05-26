import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai


DEFAULT_DATASETS = ["pog", "pog_dense"]
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DTYPE = "float16"
DEFAULT_CATEGORY_DTYPE = "float32"
CATEGORY_FIELD_CANDIDATES = ["cate_id", "cate", "category"]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean_text(text):
    return " ".join(str(text).split())


def item_title(item_info, item_id):
    item = item_info.get(str(int(item_id)), {})
    return clean_text(item.get("title", f"Item {item_id}"))


def detect_category_field(item_info, requested="auto"):
    if requested != "auto":
        return requested
    counts = {field: 0 for field in CATEGORY_FIELD_CANDIDATES}
    for item in item_info.values():
        for field in CATEGORY_FIELD_CANDIDATES:
            value = item.get(field)
            if value is not None and str(value).strip():
                counts[field] += 1
    resolved = max(counts, key=counts.get)
    if counts[resolved] == 0:
        raise ValueError("No usable category field found in item_info.json.")
    return resolved


def load_train_items(dataset_dir):
    train_path = dataset_dir / "bi_train.txt"
    train_items = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [part.strip() for part in line.strip().split(",") if part.strip()]
            for item_id in parts[1:]:
                train_items.add(int(item_id))
    return train_items


def build_category_items(item_info, category_field, allowed_items=None):
    category_to_items = defaultdict(list)
    for item_id_str, item in item_info.items():
        item_id = int(item_id_str)
        if allowed_items is not None and item_id not in allowed_items:
            continue
        value = item.get(category_field)
        if value is None or not str(value).strip():
            continue
        category_to_items[str(value).strip()].append(item_id)
    for category in category_to_items:
        category_to_items[category].sort()
    return category_to_items


def load_item_embeddings(repo_root, args, dataset):
    cache_dir = (
        repo_root
        / args.item_embedding_cache_root
        / args.embedding_model
        / "all_items"
        / dataset
    )
    cache_path = cache_dir / f"embeddings_{args.embedding_model}_{args.embedding_dtype}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        ids = data["ids"].astype(np.int64)
        embeddings = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    normed = np.zeros_like(embeddings, dtype=np.float32)
    mask = norms > 0
    normed[mask] = embeddings[mask] / norms[mask, None]
    return {int(item_id): idx for idx, item_id in enumerate(ids.tolist())}, normed


def load_category_embeddings(repo_root, args, dataset):
    cache_dir = (
        repo_root
        / args.category_embedding_cache_root
        / args.embedding_model
        / "all_items"
        / dataset
    )
    cache_path = (
        cache_dir
        / f"category_embeddings_{args.embedding_model}_{args.category_embedding_dtype}.npz"
    )
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        category_ids = data["category_ids"].astype(str).tolist()
        embeddings = data["embeddings_normed"].astype(np.float32)
        counts = data["counts"].astype(np.int64)
    return {
        str(category): {
            "row": idx,
            "embedding": embeddings[idx],
            "count": int(counts[idx]),
        }
        for idx, category in enumerate(category_ids)
    }


def select_representative_items(
    category_to_train_items,
    category_embeddings,
    item_to_row,
    item_embeddings,
    items_per_category,
):
    result = {}
    for category in sorted(category_to_train_items):
        items = [
            int(item_id)
            for item_id in category_to_train_items[category]
            if int(item_id) in item_to_row
        ]
        if not items:
            result[category] = []
            continue
        category_entry = category_embeddings.get(category)
        if category_entry is None:
            result[category] = items[:items_per_category]
            continue
        rows = [item_to_row[item_id] for item_id in items]
        centroid = category_entry["embedding"]
        scores = item_embeddings[rows] @ centroid
        order = np.argsort(-scores, kind="mergesort")
        result[category] = [items[int(idx)] for idx in order[:items_per_category].tolist()]
    return result


def make_prompt(batch, dataset):
    payload = []
    for entry in batch:
        payload.append(
            {
                "category_id": entry["category_id"],
                "representative_item_titles": entry["representative_item_titles"],
            }
        )

    return (
        "You are naming hidden fashion e-commerce categories from Chinese item titles.\n"
        "Each category_id is an opaque hash. For each category, infer the shared product type.\n"
        "Use the item titles only; do not invent a brand-specific category unless the category is clearly brand-only.\n"
        "Return concise, reusable category names suitable for prompting another LLM.\n\n"
        f"Dataset: {dataset}\n"
        "Output must be a JSON array with exactly one object per input category, in the same order.\n"
        "Each object must contain:\n"
        "- category_id: same string as input\n"
        "- category_name_en: concise English noun phrase, 2-6 words\n"
        "- category_name_ko: concise Korean noun phrase\n"
        "- short_description_en: one short English sentence\n"
        "- keywords_en: array of 3-6 short English keywords\n"
        "- confidence: number from 0 to 1\n\n"
        "Input categories:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_json_response(text):
    clean = text.strip()
    if "```json" in clean:
        clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in clean:
        clean = clean.split("```", 1)[1].split("```", 1)[0].strip()
    start = clean.find("[")
    end = clean.rfind("]")
    if start != -1 and end != -1 and end > start:
        clean = clean[start : end + 1]
    return json.loads(clean)


def normalize_name_record(record, fallback_category_id):
    category_id = str(record.get("category_id", fallback_category_id))
    keywords = record.get("keywords_en", [])
    if isinstance(keywords, str):
        keywords = [part.strip() for part in re.split(r"[,;/]", keywords) if part.strip()]
    try:
        confidence = float(record.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "category_id": category_id,
        "category_name_en": clean_text(record.get("category_name_en", "")),
        "category_name_ko": clean_text(record.get("category_name_ko", "")),
        "short_description_en": clean_text(record.get("short_description_en", "")),
        "keywords_en": [clean_text(keyword) for keyword in keywords],
        "confidence": max(0.0, min(1.0, confidence)),
    }


def call_gemini_batch(client, model, batch, dataset, max_retries, sleep_seconds):
    prompt = make_prompt(batch, dataset)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            )
            parsed = parse_json_response(response.text)
            if not isinstance(parsed, list):
                raise ValueError("Gemini response is not a JSON array.")
            if len(parsed) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} records, got {len(parsed)}."
                )
            return [
                normalize_name_record(record, batch[idx]["category_id"])
                for idx, record in enumerate(parsed)
            ]
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Gemini batch failed after {max_retries} attempts: {last_error}")


def load_existing_records(path):
    if not path.exists():
        return {}
    data = read_json(path)
    if isinstance(data, dict) and "categories" in data:
        records = data["categories"]
    else:
        records = data
    return {str(record["category_id"]): record for record in records}


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "category_id",
        "category_name_en",
        "category_name_ko",
        "short_description_en",
        "keywords_en",
        "confidence",
        "item_count_all",
        "item_count_train",
        "representative_item_ids",
        "representative_item_titles",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["keywords_en"] = "; ".join(record.get("keywords_en", []))
            row["representative_item_ids"] = json.dumps(
                record.get("representative_item_ids", []), ensure_ascii=False
            )
            row["representative_item_titles"] = json.dumps(
                record.get("representative_item_titles", []), ensure_ascii=False
            )
            writer.writerow(row)


def write_input_csv(path, entries):
    fields = [
        "dataset",
        "category_id",
        "item_count_all",
        "item_count_train",
        "representative_item_ids",
        "representative_item_titles",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            row["representative_item_ids"] = json.dumps(
                row["representative_item_ids"], ensure_ascii=False
            )
            row["representative_item_titles"] = json.dumps(
                row["representative_item_titles"], ensure_ascii=False
            )
            writer.writerow(row)


def save_outputs(out_dir, dataset, records, entries, args):
    ordered = sorted(records.values(), key=lambda r: r["category_id"])
    payload = {
        "dataset": dataset,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "api_key_file": str(args.api_key_file) if args.api_key_file else "",
        "items_per_category": args.items_per_category,
        "batch_size": args.batch_size,
        "category_count": len(ordered),
        "created_or_updated_at_unix": int(time.time()),
        "categories": ordered,
    }
    json_path = out_dir / "category_names.json"
    csv_path = out_dir / "category_names.csv"
    input_path = out_dir / "category_naming_inputs.csv"
    write_json(json_path, payload)
    write_csv(csv_path, ordered)
    write_input_csv(input_path, entries)
    return json_path, csv_path, input_path


def run_dataset(repo_root, dataset, client, args):
    dataset_dir = repo_root / args.data_path / dataset
    item_info = read_json(dataset_dir / "item_info.json")
    category_field = detect_category_field(item_info, args.category_field)
    train_items = load_train_items(dataset_dir)

    category_to_all_items = build_category_items(item_info, category_field)
    category_to_train_items = build_category_items(item_info, category_field, train_items)
    item_to_row, item_embeddings = load_item_embeddings(repo_root, args, dataset)
    category_embeddings = load_category_embeddings(repo_root, args, dataset)
    rep_items_by_category = select_representative_items(
        category_to_train_items=category_to_train_items,
        category_embeddings=category_embeddings,
        item_to_row=item_to_row,
        item_embeddings=item_embeddings,
        items_per_category=args.items_per_category,
    )

    categories = sorted(category_to_all_items)
    entries = []
    for category in categories:
        rep_item_ids = rep_items_by_category.get(category, [])
        entries.append(
            {
                "dataset": dataset,
                "category_id": category,
                "item_count_all": len(category_to_all_items.get(category, [])),
                "item_count_train": len(category_to_train_items.get(category, [])),
                "representative_item_ids": rep_item_ids,
                "representative_item_titles": [
                    item_title(item_info, item_id) for item_id in rep_item_ids
                ],
            }
        )

    out_dir = repo_root / args.output_root / args.provider / dataset
    existing = {} if args.overwrite else load_existing_records(out_dir / "category_names.json")

    remaining = [entry for entry in entries if entry["category_id"] not in existing]
    print(
        f"[{dataset}] categories={len(entries)} existing={len(existing)} "
        f"remaining={len(remaining)} field={category_field}"
    )

    save_outputs(out_dir, dataset, existing, entries, args)
    if args.dry_run or not remaining:
        return out_dir

    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start : start + args.batch_size]
        batch_idx = start // args.batch_size + 1
        total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
        print(
            f"[{dataset}] batch {batch_idx}/{total_batches}: "
            + ", ".join(entry["category_id"] for entry in batch)
        )
        named = call_gemini_batch(
            client=client,
            model=args.model,
            batch=batch,
            dataset=dataset,
            max_retries=args.max_retries,
            sleep_seconds=args.retry_sleep_seconds,
        )
        by_entry = {entry["category_id"]: entry for entry in batch}
        for idx, record in enumerate(named):
            category_id = record["category_id"]
            entry = by_entry.get(category_id)
            if entry is None:
                entry = batch[idx]
                record["category_id"] = entry["category_id"]
            record.update(
                {
                    "dataset": dataset,
                    "item_count_all": entry["item_count_all"],
                    "item_count_train": entry["item_count_train"],
                    "representative_item_ids": entry["representative_item_ids"],
                    "representative_item_titles": entry["representative_item_titles"],
                }
            )
            existing[record["category_id"]] = record
        save_outputs(out_dir, dataset, existing, entries, args)
        time.sleep(args.call_sleep_seconds)

    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate human-readable category names with Gemini from train representative item titles."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--data-path", type=Path, default=Path("datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("analysis/category_names"))
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Environment variable to read Gemini API key from. Falls back to GEMINI_API_KEY, then GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Optional text file containing a Gemini API key. Used before environment fallback.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional Gemini API key literal. Prefer --api-key-env or --api-key-file for safer shell history.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--items-per-category", type=int, default=10)
    parser.add_argument("--category-field", default="auto")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dtype", default=DEFAULT_EMBEDDING_DTYPE)
    parser.add_argument("--category-embedding-dtype", default=DEFAULT_CATEGORY_DTYPE)
    parser.add_argument(
        "--item-embedding-cache-root",
        type=Path,
        default=Path("analysis/openai_embedding_cache"),
    )
    parser.add_argument(
        "--category-embedding-cache-root",
        type=Path,
        default=Path("analysis/category_embedding_cache"),
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--call-sleep-seconds", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")
    api_key = ""
    if args.api_key:
        api_key = args.api_key.strip()
    elif args.api_key_file:
        api_key_path = args.api_key_file
        if not api_key_path.is_absolute():
            api_key_path = repo_root / api_key_path
        api_key = api_key_path.read_text(encoding="utf-8").strip()
    elif args.api_key_env:
        api_key = os.getenv(args.api_key_env, "").strip()
    else:
        api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        raise RuntimeError(
            "Set a Gemini API key with --api-key-env, --api-key-file, --api-key, "
            "GEMINI_API_KEY, or GOOGLE_API_KEY."
        )
    client = None if args.dry_run else genai.Client(api_key=api_key)

    for dataset in args.datasets:
        out_dir = run_dataset(repo_root, dataset, client, args)
        print(f"[{dataset}] wrote {out_dir}")


if __name__ == "__main__":
    main()
