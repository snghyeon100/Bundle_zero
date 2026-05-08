import argparse
import ast
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULT_FILES = {
    "pog": r"results\pog\results_pog_20260416_142034.csv",
    "pog_dense": r"results\pog_dense\results_pog_dense_HN_C10_T5_20260430_172343.csv",
    "spotify": r"results\spotify\results_spotify_20260411_191411.csv",
    "spotify_sparse": r"results\spotify_sparse\results_spotify_sparse_20260411_195706.csv",
}


DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_BATCH_SIZE = 128
DEFAULT_OUTPUT_ROOT = r"analysis\openai_embedding_cache"


def read_env_file(path):
    env = {}
    if not path.exists():
        return env
    with open(path, encoding="utf-8") as f:
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


def parse_id_list(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    return [int(x) for x in ast.literal_eval(str(value))]


def item_text(dataset, item_info):
    if "spotify" in dataset:
        parts = [
            str(item_info.get("track_name", "")).strip(),
            str(item_info.get("artist_name", "")).strip(),
            str(item_info.get("album_name", "")).strip(),
        ]
        return " - ".join([p for p in parts if p])
    return str(item_info.get("title", "")).strip()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_item_ids(result_csv):
    df = pd.read_csv(result_csv)
    required = {"input_indices", "candidate_indices", "true_indice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{result_csv} is missing required columns: {sorted(missing)}")

    ids = set()
    for _, row in df.iterrows():
        ids.update(parse_id_list(row["input_indices"]))
        ids.update(parse_id_list(row["candidate_indices"]))
        ids.add(int(row["true_indice"]))
    return sorted(ids), len(df)


def build_target_items(repo_root, dataset, result_csv, data_path):
    item_ids, num_rows = collect_item_ids(result_csv)
    item_info_path = repo_root / data_path / dataset / "item_info.json"
    with open(item_info_path, encoding="utf-8") as f:
        info = json.load(f)

    targets = []
    for item_id in item_ids:
        key = str(item_id)
        if key not in info:
            raise KeyError(f"{dataset}: item_id {item_id} not found in {item_info_path}")
        text = item_text(dataset, info[key])
        if not text:
            text = f"Item {item_id}"
        targets.append(
            {
                "dataset": dataset,
                "item_id": item_id,
                "text": text,
                "text_sha256": text_hash(text),
            }
        )
    return targets, num_rows


def build_all_items(repo_root, dataset, data_path):
    item_info_path = repo_root / data_path / dataset / "item_info.json"
    with open(item_info_path, encoding="utf-8") as f:
        info = json.load(f)

    item_ids = sorted(int(item_id) for item_id in info.keys())
    targets = []
    for item_id in item_ids:
        item = info[str(item_id)]
        text = item_text(dataset, item)
        if not text:
            text = f"Item {item_id}"
        targets.append(
            {
                "dataset": dataset,
                "item_id": item_id,
                "text": text,
                "text_sha256": text_hash(text),
            }
        )
    return targets


def write_target_csv(path, targets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "item_id", "text_sha256", "text"])
        writer.writeheader()
        writer.writerows(targets)


def load_done_ids(chunks_dir):
    done = set()
    if not chunks_dir.exists():
        return done
    for path in chunks_dir.glob("chunk_*.npz"):
        with np.load(path, allow_pickle=False) as data:
            done.update(int(x) for x in data["ids"].tolist())
    return done


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
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt >= max_retries:
                raise RuntimeError(f"OpenAI embeddings request failed after retries: {e}") from e

        sleep_s = min(60, 2 ** attempt)
        print(f"  retrying after {sleep_s}s...")
        time.sleep(sleep_s)

    raise RuntimeError("Unexpected retry loop exit")


def next_chunk_index(chunks_dir):
    existing = []
    for path in chunks_dir.glob("chunk_*.npz"):
        try:
            existing.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(existing, default=-1) + 1


def save_chunk(path, ids, text_hashes, embeddings, model, usage):
    emb = np.asarray(embeddings, dtype=np.float16)
    ids_arr = np.asarray(ids, dtype=np.int64)
    hashes_arr = np.asarray(text_hashes, dtype="U64")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=ids_arr,
        text_sha256=hashes_arr,
        embeddings=emb,
        model=np.asarray([model]),
        prompt_tokens=np.asarray([int(usage.get("prompt_tokens", 0))], dtype=np.int64),
        total_tokens=np.asarray([int(usage.get("total_tokens", 0))], dtype=np.int64),
    )


def assemble_cache(dataset_dir, model, source_csv, targets, dtype):
    chunks_dir = dataset_dir / "chunks"
    chunks = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_dir}")

    by_id = {}
    total_prompt_tokens = 0
    total_tokens = 0
    dim = None
    for path in chunks:
        with np.load(path, allow_pickle=False) as data:
            ids = data["ids"]
            embeddings = data["embeddings"]
            hashes = data["text_sha256"]
            total_prompt_tokens += int(data["prompt_tokens"][0]) if "prompt_tokens" in data else 0
            total_tokens += int(data["total_tokens"][0]) if "total_tokens" in data else 0
            dim = embeddings.shape[1]
            for item_id, emb, h in zip(ids.tolist(), embeddings, hashes.tolist()):
                by_id[int(item_id)] = (emb, h)

    expected_ids = [t["item_id"] for t in targets]
    missing = sorted(set(expected_ids) - set(by_id))
    if missing:
        raise ValueError(f"{dataset_dir.name}: {len(missing)} target ids are still missing. First ids: {missing[:20]}")

    sorted_ids = np.asarray(sorted(expected_ids), dtype=np.int64)
    out_dtype = np.float16 if dtype == "float16" else np.float32
    matrix = np.vstack([by_id[int(item_id)][0].astype(out_dtype, copy=False) for item_id in sorted_ids])
    hashes = np.asarray([by_id[int(item_id)][1] for item_id in sorted_ids], dtype="U64")

    output_path = dataset_dir / f"embeddings_{model}_{dtype}.npz"
    np.savez_compressed(output_path, ids=sorted_ids, text_sha256=hashes, embeddings=matrix)

    metadata = {
        "dataset": dataset_dir.name,
        "model": model,
        "source_csv": str(source_csv) if source_csv is not None else None,
        "num_items": int(len(sorted_ids)),
        "embedding_dim": int(dim),
        "dtype": dtype,
        "prompt_tokens": int(total_prompt_tokens),
        "total_tokens": int(total_tokens),
        "output_path": str(output_path),
        "created_at_unix": int(time.time()),
    }
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return output_path, metadata_path, metadata


def chunks(iterable, size):
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def estimate_storage_bytes(num_items, dim, dtype):
    bytes_per_value = 2 if dtype == "float16" else 4
    return num_items * dim * bytes_per_value


def run_dataset(repo_root, dataset, result_csv, args, api_key):
    if args.mode == "all-items":
        result_csv = None
        targets = build_all_items(repo_root, dataset, args.data_path)
        num_rows = None
        dataset_dir = repo_root / args.output_root / args.model / "all_items" / dataset
    else:
        result_csv = Path(result_csv)
        if not result_csv.is_absolute():
            result_csv = repo_root / result_csv
        if not result_csv.exists():
            raise FileNotFoundError(result_csv)
        targets, num_rows = build_target_items(repo_root, dataset, result_csv, args.data_path)
        dataset_dir = repo_root / args.output_root / args.model / dataset
    target_csv = dataset_dir / "target_items.csv"
    write_target_csv(target_csv, targets)

    chunks_dir = dataset_dir / "chunks"
    done_ids = load_done_ids(chunks_dir)
    remaining = [t for t in targets if t["item_id"] not in done_ids]

    print(f"\n=== {dataset} ===")
    print(f"mode            : {args.mode}")
    print(f"source_csv      : {result_csv if result_csv is not None else '(all item_info.json items)'}")
    print(f"result_rows     : {num_rows if num_rows is not None else '(n/a)'}")
    print(f"target_items    : {len(targets)}")
    print(f"already_cached  : {len(done_ids & set(t['item_id'] for t in targets))}")
    print(f"remaining       : {len(remaining)}")
    print(f"target_csv      : {target_csv}")
    est_bytes = estimate_storage_bytes(len(targets), args.dimensions or 3072, args.dtype)
    print(f"est_matrix_size : {est_bytes / 1024 / 1024 / 1024:.2f} GiB ({args.dtype}, before npz compression)")

    if args.dry_run:
        return

    if remaining and not api_key:
        raise RuntimeError("OPENAI_API_KEY is required. Set it in the environment or add OPENAI_API_KEY=... to .env")

    chunk_idx = next_chunk_index(chunks_dir)
    for batch in chunks(remaining, args.batch_size):
        ids = [t["item_id"] for t in batch]
        texts = [t["text"] for t in batch]
        hashes = [t["text_sha256"] for t in batch]
        print(f"  embedding chunk {chunk_idx:05d}: {len(batch)} items, ids {ids[0]}..{ids[-1]}")
        response = call_embeddings_with_retry(
            api_key=api_key,
            model=args.model,
            texts=texts,
            dimensions=args.dimensions,
            max_retries=args.max_retries,
        )
        data = sorted(response["data"], key=lambda x: x["index"])
        embeddings = [entry["embedding"] for entry in data]
        save_chunk(chunks_dir / f"chunk_{chunk_idx:05d}.npz", ids, hashes, embeddings, args.model, response.get("usage", {}))
        chunk_idx += 1

    output_path, metadata_path, metadata = assemble_cache(dataset_dir, args.model, result_csv, targets, args.dtype)
    print(f"final_cache     : {output_path}")
    print(f"metadata        : {metadata_path}")
    print(f"tokens          : {metadata['total_tokens']}")


def parse_result_override(values):
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid --result value: {value}. Use dataset=path")
        dataset, path = value.split("=", 1)
        overrides[dataset.strip()] = path.strip()
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Build OpenAI text embedding caches for result CSV item ids.")
    parser.add_argument("--mode", choices=["result-items", "all-items"], default="result-items")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_RESULT_FILES), choices=list(DEFAULT_RESULT_FILES))
    parser.add_argument("--result", action="append", default=[], help="Override result CSV as dataset=path.")
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dimensions", type=int, default=None, help="Optional embedding dimensions. Omit for model default.")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="Only collect target item ids/texts and write target_items.csv.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    result_files = dict(DEFAULT_RESULT_FILES)
    result_files.update(parse_result_override(args.result))
    api_key = None if args.dry_run else get_api_key(repo_root)

    for dataset in args.datasets:
        run_dataset(repo_root, dataset, result_files[dataset], args, api_key)


if __name__ == "__main__":
    main()
