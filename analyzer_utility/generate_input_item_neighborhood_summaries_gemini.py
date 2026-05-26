import argparse
import ast
import csv
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from google import genai


DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_CACHE_ROOT = Path("analysis") / "input_item_neighborhood_summaries" / "gemini"


def parse_id_list(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(x) for x in ast.literal_eval(text)]


def latest_result_csv(repo_root, dataset):
    pattern = repo_root / "results" / dataset / "*.csv"
    files = glob.glob(str(pattern))
    if not files:
        raise FileNotFoundError(f"No result CSV found under {pattern}")
    return Path(max(files, key=os.path.getmtime))


def collect_input_item_ids(result_csv):
    item_ids = set()
    with open(result_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "input_indices" not in (reader.fieldnames or []):
            raise ValueError(f"{result_csv} is missing input_indices")
        for row in reader:
            item_ids.update(parse_id_list(row.get("input_indices")))
    return sorted(item_ids)


def load_item_info(repo_root, data_path, dataset):
    path = repo_root / data_path / dataset / "item_info.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def item_text(dataset, item_info, item_id):
    item = item_info.get(str(int(item_id)), {})
    if "spotify" in dataset:
        parts = [
            str(item.get("track_name", "")).strip(),
            str(item.get("artist_name", "")).strip(),
            str(item.get("album_name", "")).strip(),
        ]
        text = " - ".join([p for p in parts if p])
        return text or f"Track {item_id}"
    return str(item.get("title", "")).strip() or f"Item {item_id}"


def clean_inline_text(text):
    return " ".join(str(text).split())


def read_env_file(path):
    env = {}
    if not path.exists():
        return env
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def get_api_key(repo_root, env_name):
    if os.environ.get(env_name):
        return os.environ[env_name]
    env = read_env_file(repo_root / ".env")
    return env.get(env_name)


def console_safe_text(text):
    encoding = sys.stdout.encoding or "utf-8"
    return str(text).encode(encoding, errors="backslashreplace").decode(encoding)


def is_quota_error(error):
    text = str(error).lower()
    return any(marker in text for marker in ("429", "quota", "resource_exhausted", "rate limit"))


def is_high_demand_error(error):
    text = str(error).lower()
    return any(marker in text for marker in ("503", "overloaded", "high demand", "unavailable"))


def load_train_bundle_items(dataset_dir):
    bundle_items = {}
    item_to_bundles = defaultdict(set)
    with open(dataset_dir / "bi_train.txt", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) < 2:
                continue
            bundle_id = vals[0]
            items = list(dict.fromkeys(vals[1:]))
            bundle_items[bundle_id] = items
            for item_id in items:
                item_to_bundles[item_id].add(bundle_id)
    return bundle_items, item_to_bundles


def soft_mapping_path(dataset_dir, source):
    file_by_source = {
        "item_smoothing_text": "item_smoothing_i2bprime_text_top1.json",
        "item_smoothing_bi_lgcn": "item_smoothing_i2bprime_bi_lgcn_top1.json",
        "bundle_smoothing_text": "bundle_smoothing_i2bprime_text_top1.json",
        "bundle_smoothing_bi_lgcn": "bundle_smoothing_i2bprime_bi_lgcn_top1.json",
    }
    if source not in file_by_source:
        raise ValueError(f"Unknown soft source {source}. Allowed: {sorted(file_by_source)}")
    return dataset_dir / file_by_source[source]


def load_soft_item_to_bundles(dataset_dir, source):
    path = soft_mapping_path(dataset_dir, source)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        int(item_id): {int(bundle_id) for bundle_id in bundle_ids}
        for item_id, bundle_ids in raw.items()
    }, path


def top_co_items(item_id, bundle_ids, bundle_items, item_info, dataset, top_k, exclude_items=None):
    exclude = {int(item_id)}
    if exclude_items:
        exclude.update(int(x) for x in exclude_items)
    counts = Counter()
    for bundle_id in bundle_ids:
        for co_item_id in bundle_items.get(int(bundle_id), []):
            co_item_id = int(co_item_id)
            if co_item_id not in exclude:
                counts[co_item_id] += 1
    top = counts.most_common(int(top_k))
    return [
        {
            "item_id": int(co_item_id),
            "count": int(count),
            "text": clean_inline_text(item_text(dataset, item_info, co_item_id)),
        }
        for co_item_id, count in top
    ]


def format_evidence_items(items):
    if not items:
        return "- None"
    return "\n".join(
        f"- {row['text']} (count={row['count']})"
        for row in items
    )


def summary_prompt(dataset, item_id, text, exact_items, soft_items, soft_source):
    collection = "playlist" if "spotify" in dataset else "fashion outfit"
    item_name = "song" if "spotify" in dataset else "fashion item"
    return (
        "You are a careful evidence summarizer for a zero-shot multiple-choice "
        f"{collection} completion task.\n"
        f"Summarize the historical neighborhood of one input {item_name}. "
        "Use the item's own text plus train-set co-affiliated items.\n"
        "Exact evidence comes from IB x BI: train bundles containing the same item. "
        f"Soft evidence comes from I-I'-B using {soft_source}: train bundles of a title-text-similar item.\n"
        "Do not choose an answer, do not mention candidates, and do not invent unsupported facts. "
        "Treat soft evidence as approximate. Return one concise English sentence under 35 words.\n\n"
        f"Input item id: {item_id}\n"
        f"Input item text: {text}\n\n"
        "Exact co-affiliated items:\n"
        f"{format_evidence_items(exact_items)}\n\n"
        "Soft co-affiliated items:\n"
        f"{format_evidence_items(soft_items)}\n\n"
        "Summary:"
    )


def clean_summary(text):
    text = " ".join(str(text).split())
    for prefix in ("Summary:", "-", "Item summary:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text.strip().strip('"')


def load_cache(path):
    if not path.exists():
        return {"metadata": {}, "items": {}}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "items" not in data:
        data = {"metadata": {}, "items": data}
    return data


def save_cache(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    last_error = None
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except OSError as e:
            last_error = e
            time.sleep(0.5 * (attempt + 1))
    raise last_error


def generate_with_key_fallback(clients, client_idx, prompt, args, item_pos, total):
    last_error = None
    for _ in range(len(clients)):
        env_name, client = clients[client_idx]
        for retry_idx in range(args.high_demand_max_retries + 1):
            try:
                res = client.models.generate_content(
                    model=args.model,
                    contents=prompt,
                    config={"temperature": 0.0, "max_output_tokens": args.max_output_tokens},
                )
                return res, client_idx
            except Exception as e:
                last_error = e
                if is_high_demand_error(e) and retry_idx < args.high_demand_max_retries:
                    wait_s = args.high_demand_base_delay * (retry_idx + 1)
                    print(
                        console_safe_text(
                            f"[{item_pos}/{total}] {env_name} high demand; retrying same key in {wait_s:.1f}s..."
                        )
                    )
                    time.sleep(wait_s)
                    continue
                if len(clients) > 1 and is_quota_error(e):
                    print(
                        console_safe_text(
                            f"[{item_pos}/{total}] {env_name} quota/rate error; switching key..."
                        )
                    )
                    client_idx = (client_idx + 1) % len(clients)
                    time.sleep(max(args.sleep, 1.0))
                    break
                raise
        if len(clients) > 1 and is_quota_error(last_error):
            continue
        raise last_error
    raise last_error


def main():
    parser = argparse.ArgumentParser(
        description="Generate one cached input-item neighborhood summary from exact IBxBI and soft I-I'-B evidence."
    )
    parser.add_argument("--dataset", default="pog")
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--result-csv", default="")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--soft-source", default="item_smoothing_text")
    parser.add_argument("--exact-top-k", type=int, default=5)
    parser.add_argument("--soft-top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY_2")
    parser.add_argument("--api-key-envs", default="")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--high-demand-max-retries", type=int, default=5)
    parser.add_argument("--high-demand-base-delay", type=float, default=10.0)
    parser.add_argument("--max-output-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")
    api_key_envs = [
        name.strip()
        for name in (args.api_key_envs or args.api_key_env).split(",")
        if name.strip()
    ]
    api_keys = []
    for env_name in api_key_envs:
        api_key = get_api_key(repo_root, env_name)
        if api_key:
            api_keys.append((env_name, api_key))
    if not api_keys:
        raise RuntimeError(f"Missing Gemini API key in any of: {api_key_envs}")

    result_csv = Path(args.result_csv) if args.result_csv else latest_result_csv(repo_root, args.dataset)
    if not result_csv.is_absolute():
        result_csv = repo_root / result_csv
    input_ids = collect_input_item_ids(result_csv)
    if args.limit > 0:
        input_ids = input_ids[: args.limit]

    dataset_dir = repo_root / args.data_path / args.dataset
    item_info = load_item_info(repo_root, args.data_path, args.dataset)
    train_bundle_items, item_to_bundles = load_train_bundle_items(dataset_dir)
    soft_item_to_bundles, soft_path = load_soft_item_to_bundles(dataset_dir, args.soft_source)

    cache_path = repo_root / args.cache_root / args.dataset / "input_item_descriptions.json"
    cache = load_cache(cache_path)
    cache["metadata"] = {
        **cache.get("metadata", {}),
        "dataset": args.dataset,
        "result_csv": str(result_csv),
        "model": args.model,
        "api_key_envs": api_key_envs,
        "num_target_input_items": len(input_ids),
        "soft_source": args.soft_source,
        "soft_mapping_path": str(soft_path),
        "exact_top_k": int(args.exact_top_k),
        "soft_top_k": int(args.soft_top_k),
        "prompt_version": "input_item_neighborhood_summary_v1",
        "method": "Input item + exact IBxBI co-affiliated items + title-text embedding I-I'-B soft co-affiliated items -> summary agent.",
    }

    clients = [(env_name, genai.Client(api_key=api_key)) for env_name, api_key in api_keys]
    client_idx = 0
    done = 0
    skipped = 0
    for idx, item_id in enumerate(input_ids, start=1):
        key = str(item_id)
        if key in cache["items"] and cache["items"][key].get("summary") and not args.overwrite:
            skipped += 1
            continue

        text = clean_inline_text(item_text(args.dataset, item_info, item_id))
        exact_bundle_ids = item_to_bundles.get(int(item_id), set())
        soft_bundle_ids = soft_item_to_bundles.get(int(item_id), set())
        exact_items = top_co_items(
            item_id,
            exact_bundle_ids,
            train_bundle_items,
            item_info,
            args.dataset,
            args.exact_top_k,
        )
        soft_items = top_co_items(
            item_id,
            soft_bundle_ids,
            train_bundle_items,
            item_info,
            args.dataset,
            args.soft_top_k,
        )
        prompt = summary_prompt(args.dataset, item_id, text, exact_items, soft_items, args.soft_source)
        res, client_idx = generate_with_key_fallback(clients, client_idx, prompt, args, idx, len(input_ids))
        raw = res.text or ""
        summary = clean_summary(raw)
        cache["items"][key] = {
            "item_id": int(item_id),
            "text": text,
            "summary": summary,
            "description": summary,
            "raw_response": raw,
            "prompt": prompt,
            "model": args.model,
            "exact_bundle_count": int(len(exact_bundle_ids)),
            "soft_bundle_count": int(len(soft_bundle_ids)),
            "exact_coaffiliated_items": exact_items,
            "soft_coaffiliated_items": soft_items,
        }
        save_cache(cache_path, cache)
        done += 1
        print(console_safe_text(f"[{idx}/{len(input_ids)}] item={item_id} summary={summary[:120]}"))
        if args.sleep > 0 and idx < len(input_ids):
            time.sleep(args.sleep)

    print(console_safe_text(f"Wrote {cache_path}"))
    print(f"Generated={done}, skipped_cached={skipped}, total_target={len(input_ids)}")


if __name__ == "__main__":
    main()
