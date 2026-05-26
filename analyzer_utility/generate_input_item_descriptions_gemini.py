import argparse
import ast
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_CACHE_ROOT = Path("analysis") / "input_item_descriptions" / "gemini"


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


def item_text(dataset, item):
    if "spotify" in dataset:
        parts = [
            str(item.get("track_name", "")).strip(),
            str(item.get("artist_name", "")).strip(),
            str(item.get("album_name", "")).strip(),
        ]
        return " - ".join([p for p in parts if p])
    return str(item.get("title", "")).strip()


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


def clean_description(text):
    text = " ".join(str(text).split())
    for prefix in ("Description:", "Item description:", "-"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text.strip().strip('"')


def description_prompt(dataset, text):
    if "spotify" in dataset:
        return (
            "Write one concise English description of this song for a playlist continuation task. "
            "Mention genre, mood, era, or artist context only if it is explicit in the text. "
            "Do not predict playlist fit. Return only the description, under 25 words.\n\n"
            f"Song text: {text}"
        )
    return (
        "Write one concise English product description of this fashion item for an outfit completion task. "
        "Mention item category, style, season, gender, color, material, or occasion only when inferable from the title. "
        "Do not predict the missing outfit item. Return only the description, under 25 words.\n\n"
        f"Item title: {text}"
    )


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


def main():
    parser = argparse.ArgumentParser(description="Generate one cached LLM description per input item.")
    parser.add_argument("--dataset", default="pog")
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--result-csv", default="")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY_2")
    parser.add_argument(
        "--api-key-envs",
        default="",
        help="Comma-separated Gemini API key env names. Used in order; quota/rate errors rotate to the next key.",
    )
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--high-demand-max-retries", type=int, default=5)
    parser.add_argument("--high-demand-base-delay", type=float, default=10.0)
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

    item_info = load_item_info(repo_root, args.data_path, args.dataset)
    cache_path = repo_root / args.cache_root / args.dataset / "input_item_descriptions.json"
    cache = load_cache(cache_path)
    cache["metadata"] = {
        **cache.get("metadata", {}),
        "dataset": args.dataset,
        "result_csv": str(result_csv),
        "model": args.model,
        "api_key_envs": api_key_envs,
        "num_target_input_items": len(input_ids),
        "prompt_version": "input_item_description_v1",
    }

    clients = [(env_name, genai.Client(api_key=api_key)) for env_name, api_key in api_keys]
    client_idx = 0
    done = 0
    skipped = 0
    for idx, item_id in enumerate(input_ids, start=1):
        key = str(item_id)
        if key in cache["items"] and cache["items"][key].get("description") and not args.overwrite:
            skipped += 1
            continue
        item = item_info.get(key, {})
        text = item_text(args.dataset, item) or f"Item {item_id}"
        prompt = description_prompt(args.dataset, text)
        last_error = None
        for _ in range(len(clients)):
            env_name, client = clients[client_idx]
            for retry_idx in range(args.high_demand_max_retries + 1):
                try:
                    res = client.models.generate_content(
                        model=args.model,
                        contents=prompt,
                        config={"temperature": 0.0, "max_output_tokens": 60},
                    )
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if is_high_demand_error(e) and retry_idx < args.high_demand_max_retries:
                        wait_s = args.high_demand_base_delay * (retry_idx + 1)
                        print(
                            console_safe_text(
                                f"[{idx}/{len(input_ids)}] {env_name} high demand; retrying same key in {wait_s:.1f}s..."
                            )
                        )
                        time.sleep(wait_s)
                        continue
                    if len(clients) > 1 and is_quota_error(e):
                        print(
                            console_safe_text(
                                f"[{idx}/{len(input_ids)}] {env_name} quota/rate error; switching key..."
                            )
                        )
                        client_idx = (client_idx + 1) % len(clients)
                        time.sleep(max(args.sleep, 1.0))
                        break
                    raise
            if last_error is None:
                break
            if len(clients) > 1 and is_quota_error(last_error):
                continue
            if len(clients) > 1 and is_high_demand_error(last_error):
                raise last_error
            if len(clients) > 1:
                raise last_error
            if last_error is not None:
                raise last_error
        if last_error is not None:
            raise last_error
        raw = res.text or ""
        description = clean_description(raw)
        cache["items"][key] = {
            "item_id": int(item_id),
            "text": text,
            "description": description,
            "raw_response": raw,
            "model": args.model,
        }
        save_cache(cache_path, cache)
        done += 1
        print(console_safe_text(f"[{idx}/{len(input_ids)}] item={item_id} description={description[:100]}"))
        if args.sleep > 0 and idx < len(input_ids):
            time.sleep(args.sleep)

    print(console_safe_text(f"Wrote {cache_path}"))
    print(f"Generated={done}, skipped_cached={skipped}, total_target={len(input_ids)}")


if __name__ == "__main__":
    main()
