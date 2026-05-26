import argparse
import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_BASE_CACHE_ROOT = Path("analysis") / "openai_embedding_cache" / DEFAULT_MODEL / "all_items"
DEFAULT_DESCRIPTION_ROOT = Path("analysis") / "input_item_descriptions" / "gemini"
DEFAULT_OUTPUT_ROOT = Path("analysis") / "input_item_description_embedding_cache"


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


def get_api_key(repo_root):
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env = read_env_file(repo_root / ".env")
    return env.get("OPENAI_API_KEY")


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


def load_descriptions(repo_root, root, dataset, field):
    path = repo_root / root / dataset / "input_item_descriptions.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    records = raw.get("items", raw)
    descriptions = {}
    for item_id, record in records.items():
        text = record.get(field, "") if isinstance(record, dict) else str(record)
        text = " ".join(str(text).split())
        if text:
            descriptions[int(item_id)] = text
    return descriptions, path


def load_base_cache(repo_root, cache_root, dataset, model, dtype):
    path = repo_root / cache_root / dataset / f"embeddings_{model}_{dtype}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"].astype(np.int64)
        embeddings = data["embeddings"]
        hashes = data["text_sha256"] if "text_sha256" in data else np.asarray([""] * len(ids), dtype="U64")
    return ids, embeddings, hashes, path


def embedding_request(api_key, model, texts, dimensions=None, timeout=120):
    payload = {"model": model, "input": texts}
    if dimensions is not None:
        payload["dimensions"] = dimensions
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_embeddings_with_retry(api_key, model, texts, dimensions, max_retries):
    for attempt in range(max_retries + 1):
        try:
            return embedding_request(api_key, model, texts, dimensions=dimensions)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            retryable = e.code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt >= max_retries:
                raise RuntimeError(f"OpenAI embeddings request failed: HTTP {e.code}: {detail}") from e
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            ConnectionResetError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            socket.timeout,
        ) as e:
            if attempt >= max_retries:
                raise RuntimeError(f"OpenAI embeddings request failed after retries: {e}") from e
        sleep_s = min(60, 2 ** attempt)
        print(f"  retrying after {sleep_s}s...")
        time.sleep(sleep_s)
    raise RuntimeError("Unexpected retry loop exit")


def main():
    parser = argparse.ArgumentParser(description="Build an all-items embedding cache with generated input descriptions.")
    parser.add_argument("--dataset", default="pog")
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--base-cache-root", default=str(DEFAULT_BASE_CACHE_ROOT))
    parser.add_argument("--description-root", default=str(DEFAULT_DESCRIPTION_ROOT))
    parser.add_argument("--description-field", default="description")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dimensions", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    api_key = get_api_key(repo_root)
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")

    ids, base_embeddings, base_hashes, base_path = load_base_cache(
        repo_root,
        Path(args.base_cache_root),
        args.dataset,
        args.model,
        args.dtype,
    )
    descriptions, descriptions_path = load_descriptions(
        repo_root,
        Path(args.description_root),
        args.dataset,
        args.description_field,
    )
    item_info = load_item_info(repo_root, args.data_path, args.dataset)

    id_to_row = {int(item_id): row for row, item_id in enumerate(ids.tolist())}
    target_ids = [item_id for item_id in sorted(descriptions) if item_id in id_to_row]
    output_dtype = np.float16 if args.dtype == "float16" else np.float32
    output_embeddings = base_embeddings.astype(output_dtype, copy=True)
    text_hashes = base_hashes.astype("U64", copy=True)

    augmented_texts = []
    for item_id in target_ids:
        title = item_text(args.dataset, item_info.get(str(item_id), {})) or f"Item {item_id}"
        augmented_texts.append(f"{title} [Generated item description: {descriptions[item_id]}]")

    total_prompt_tokens = 0
    total_tokens = 0
    for start in range(0, len(target_ids), args.batch_size):
        end = min(start + args.batch_size, len(target_ids))
        batch_ids = target_ids[start:end]
        batch_texts = augmented_texts[start:end]
        response = call_embeddings_with_retry(
            api_key,
            args.model,
            batch_texts,
            args.dimensions if args.dimensions > 0 else None,
            args.max_retries,
        )
        vectors = [row["embedding"] for row in response["data"]]
        for item_id, vector in zip(batch_ids, vectors):
            output_embeddings[id_to_row[item_id]] = np.asarray(vector, dtype=output_dtype)
        usage = response.get("usage", {})
        total_prompt_tokens += int(usage.get("prompt_tokens", 0))
        total_tokens += int(usage.get("total_tokens", 0))
        print(f"[{end}/{len(target_ids)}] embedded description-augmented items")

    output_dir = repo_root / args.output_root / args.model / "all_items" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"embeddings_{args.model}_{args.dtype}.npz"
    np.savez_compressed(output_path, ids=ids, text_sha256=text_hashes, embeddings=output_embeddings)

    meta = {
        "dataset": args.dataset,
        "model": args.model,
        "dtype": args.dtype,
        "base_cache": str(base_path),
        "description_cache": str(descriptions_path),
        "output_cache": str(output_path),
        "num_items": int(len(ids)),
        "num_description_augmented_items": int(len(target_ids)),
        "prompt_tokens": int(total_prompt_tokens),
        "total_tokens": int(total_tokens),
        "method": "Base all-item text embeddings, replacing cached input items with title plus generated item description embeddings.",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Wrote {output_path}")
    print(f"Description-augmented items: {len(target_ids)} / {len(ids)}")


if __name__ == "__main__":
    main()
