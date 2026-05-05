import argparse
import json
import os
import re
import time

import pandas as pd


TAG_DEFINITIONS = {
    "easy_outlier_elimination": "Most wrong candidates are obviously incompatible.",
    "explicit_anchor_matching": "The answer is supported by explicit text anchors such as brand, artist, album, season, gender, color, or style keywords.",
    "category_completion": "The task is mainly about finding the missing functional category, such as top/bottom/shoes/accessory.",
    "broad_style_or_genre_match": "The model can solve it by matching broad style, aesthetic, genre, era, or occasion.",
    "fine_grained_similarity_confusion": "Multiple candidates share the right broad category/style, requiring subtle distinctions.",
    "same_category_hard_negative": "The model must choose among candidates from the same item/song category.",
    "counterintuitive_ground_truth": "The ground truth is less intuitive than at least one distractor.",
    "multiple_plausible_candidates": "Several candidates could reasonably fit the input bundle.",
    "collaborative_preference_needed": "Text/common-sense is insufficient; user/bundle co-occurrence or historical preference is needed.",
    "visual_signal_needed": "Visual details such as color, pattern, silhouette, texture, or material are needed.",
    "metadata_shortcut": "The prediction can be explained by shortcut metadata such as popularity, artist/album/brand, or exact keyword overlap.",
    "prompt_or_response_error": "The model failed due to invalid output, formatting, or instruction-following issues.",
    "ambiguous_or_unknown": "The cause is unclear from the provided text.",
}


def load_config(config_path):
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
    except ImportError:
        print(">>> Warning: pyyaml is not installed; continuing without config.yaml.")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_dataset_from_csv(csv_path):
    parts = os.path.normpath(csv_path).split(os.sep)
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    base = os.path.basename(csv_path).lower()
    for dataset in ("spotify_sparse", "spotify", "pog_dense", "pog"):
        if dataset in base:
            return dataset
    return None


def find_latest_result_csv(output_dir, dataset):
    dataset_dir = os.path.join(output_dir, dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset output directory not found: {dataset_dir}")

    candidates = []
    for name in os.listdir(dataset_dir):
        if not name.endswith(".csv"):
            continue
        lowered = name.lower()
        if "meta" in lowered or "semantic" in lowered or "difficulty" in lowered:
            continue
        candidates.append(os.path.join(dataset_dir, name))

    if not candidates:
        raise FileNotFoundError(f"No result CSV found in {dataset_dir}")
    return max(candidates, key=os.path.getctime)


def clean_json_text(text):
    text = (text or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        text = text[start : end + 1]
    return text


def parse_json_response(text):
    clean = clean_json_text(text)
    return json.loads(clean)


def short_text(value, limit=1600):
    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def build_prompt(rows, dataset_name):
    if "spotify" in dataset_name:
        task_name = "playlist continuation"
        bundle_name = "playlist"
        item_name = "song"
    else:
        task_name = "fashion bundle construction"
        bundle_name = "outfit bundle"
        item_name = "fashion item"

    tag_lines = "\n".join([f"- {tag}: {desc}" for tag, desc in TAG_DEFINITIONS.items()])
    case_blocks = []

    for idx, row in rows:
        hit = row.get("hit", "")
        prediction = row.get("prediction", "")
        raw_response = short_text(row.get("raw_response", ""), 180)

        block = (
            f"Case Index: {idx}\n"
            f"Input {bundle_name}: {short_text(row.get('input_str', ''))}\n"
            f"Candidate {item_name}s: {short_text(row.get('target_str', ''))}\n"
            f"Ground Truth Option: {row.get('true_option_char', '')}\n"
            f"Model Prediction: {prediction}\n"
            f"Hit: {hit}\n"
            f"Raw Response: {raw_response}\n"
        )
        case_blocks.append(block)

    return f"""
You are an expert researcher analyzing zero-shot LLM behavior for {task_name}.
Your job is to assign semantic behavior tags to each multiple-choice recommendation case.

Important:
- Do not judge whether the dataset ground truth is objectively the best recommendation. Analyze what kind of signal or failure mode this case represents.
- Use the provided model prediction and hit value.
- Prefer concrete tags from the taxonomy. Use ambiguous_or_unknown only when the text is insufficient.
- primary_tag must be exactly one tag from the taxonomy.
- secondary_tags may contain 0 to 3 tags from the taxonomy.
- evidence must be short and grounded in the provided input/candidates/prediction.

Taxonomy:
{tag_lines}

Return ONLY a JSON list with exactly {len(rows)} objects. Use this schema:
[
  {{
    "index": 0,
    "primary_tag": "fine_grained_similarity_confusion",
    "secondary_tags": ["multiple_plausible_candidates"],
    "llm_strength": "broad semantic filtering",
    "failure_mode": "missed subtle compatibility cue",
    "requires_collaborative_signal": true,
    "ground_truth_plausibility": 3,
    "distractor_hardness": 4,
    "confidence": 0.75,
    "evidence": "The model chose a broadly plausible candidate, but several candidates share the same category/style."
  }}
]

Field rules:
- ground_truth_plausibility: integer 1-5, where 1 means counterintuitive and 5 means very obvious.
- distractor_hardness: integer 1-5, where 1 means easy distractors and 5 means very hard distractors.
- confidence: number from 0.0 to 1.0.
- requires_collaborative_signal: true when text/common-sense alone seems insufficient and historical co-occurrence/preference would help.

Cases:
{chr(10).join(case_blocks)}
""".strip()


def tag_batch(client, model, rows, dataset_name, max_retries=5):
    prompt = build_prompt(rows, dataset_name)
    last_error = None

    for attempt in range(max_retries):
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
                raise ValueError("Model response is not a JSON list.")
            return parsed
        except Exception as exc:
            last_error = exc
            wait = 10 * (attempt + 1)
            print(f">>> Batch failed: {exc} | retrying in {wait}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"Failed to tag batch after {max_retries} attempts: {last_error}")


def validate_tag_result(result):
    primary = result.get("primary_tag", "ambiguous_or_unknown")
    if primary not in TAG_DEFINITIONS:
        result["primary_tag"] = "ambiguous_or_unknown"

    secondary = result.get("secondary_tags", [])
    if not isinstance(secondary, list):
        secondary = []
    result["secondary_tags"] = [tag for tag in secondary if tag in TAG_DEFINITIONS][:3]

    for key in ("ground_truth_plausibility", "distractor_hardness"):
        try:
            result[key] = max(1, min(5, int(result.get(key, 3))))
        except (TypeError, ValueError):
            result[key] = 3

    try:
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    except (TypeError, ValueError):
        result["confidence"] = 0.5

    result["requires_collaborative_signal"] = bool(result.get("requires_collaborative_signal", False))
    result["llm_strength"] = str(result.get("llm_strength", "")).strip()
    result["failure_mode"] = str(result.get("failure_mode", "")).strip()
    result["evidence"] = str(result.get("evidence", "")).strip()
    return result


def summarize(tags_df):
    completed = tags_df[tags_df["primary_tag"].notna()]
    if completed.empty:
        return

    print("\n" + "=" * 56)
    print("Semantic Tag Summary")
    print("=" * 56)
    counts = completed["primary_tag"].value_counts()
    for tag, count in counts.items():
        print(f"- {tag}: {count} ({count / len(completed) * 100:.1f}%)")

    if "hit" in completed.columns:
        print("\nAccuracy by Primary Tag")
        for tag, group in completed.groupby("primary_tag"):
            if group["hit"].notna().any():
                hit_rate = group["hit"].astype(float).mean()
                print(f"- {tag}: {hit_rate * 100:.1f}% ({len(group)} cases)")
    print("=" * 56)


def main():
    parser = argparse.ArgumentParser(description="Assign semantic behavior tags to LLM bundle completion cases.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path.")
    parser.add_argument("--csv", default=None, help="Result CSV path. If omitted, latest result CSV for config dataset is used.")
    parser.add_argument("--dataset", default=None, help="Dataset name. If omitted, inferred from CSV or config.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of cases per API call.")
    parser.add_argument("--limit", type=int, default=-1, help="Only process first N untagged rows.")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model for semantic tagging.")
    parser.add_argument("--output", default=None, help="Output semantic tag CSV path.")
    parser.add_argument("--merge_output", action="store_true", help="Also save a result+semantic merged CSV.")
    args = parser.parse_args()

    conf = load_config(args.config)
    output_dir = conf.get("output_dir", "./results")
    dataset = args.dataset or (infer_dataset_from_csv(args.csv) if args.csv else None) or conf.get("dataset")
    if not dataset:
        raise ValueError("Dataset not provided and could not be inferred.")

    csv_path = args.csv or find_latest_result_csv(output_dir, dataset)
    dataset = args.dataset or infer_dataset_from_csv(csv_path) or dataset
    actual_output_dir = os.path.join(output_dir, dataset)
    os.makedirs(actual_output_dir, exist_ok=True)

    output_path = args.output or os.path.join(actual_output_dir, "problem_semantic_tags.csv")

    df = pd.read_csv(csv_path)
    df["index"] = df.index

    if os.path.exists(output_path):
        tags_df = pd.read_csv(output_path)
        print(f">>> Resuming existing semantic tags: {output_path}")
    else:
        tags_df = pd.DataFrame(
            {
                "index": df["index"],
                "primary_tag": pd.NA,
                "secondary_tags": pd.NA,
                "llm_strength": pd.NA,
                "failure_mode": pd.NA,
                "requires_collaborative_signal": pd.NA,
                "ground_truth_plausibility": pd.NA,
                "distractor_hardness": pd.NA,
                "confidence": pd.NA,
                "evidence": pd.NA,
            }
        )

    tagged = set(tags_df[tags_df["primary_tag"].notna()]["index"].astype(int).tolist())
    unprocessed = [idx for idx in df["index"].tolist() if int(idx) not in tagged]
    if args.limit > 0:
        unprocessed = unprocessed[: args.limit]

    if not unprocessed:
        print(">>> All rows already have semantic tags.")
        summarize(tags_df.merge(df[["index", "hit"]], on="index", how="left"))
        return

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    from dotenv import load_dotenv
    from google import genai

    load_dotenv(dotenv_path=env_path, encoding="utf-8-sig")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY not found in .env.")

    client = genai.Client(api_key=api_key)

    print(f">>> CSV      : {csv_path}")
    print(f">>> Dataset  : {dataset}")
    print(f">>> Output   : {output_path}")
    print(f">>> Untagged : {len(unprocessed)}")

    for start in range(0, len(unprocessed), args.batch_size):
        batch_indices = unprocessed[start : start + args.batch_size]
        batch_rows = [(idx, df.loc[df["index"] == idx].iloc[0]) for idx in batch_indices]
        print(f">>> Tagging batch {start // args.batch_size + 1} ({len(batch_rows)} cases)")

        batch_results = tag_batch(client, args.model, batch_rows, dataset)
        by_index = {}
        for result in batch_results:
            if "index" not in result:
                continue
            result = validate_tag_result(result)
            by_index[int(result["index"])] = result

        for idx in batch_indices:
            result = by_index.get(int(idx))
            if not result:
                print(f">>> Warning: missing result for index {idx}")
                continue

            mask = tags_df["index"].astype(int) == int(idx)
            tags_df.loc[mask, "primary_tag"] = result["primary_tag"]
            tags_df.loc[mask, "secondary_tags"] = json.dumps(result["secondary_tags"], ensure_ascii=False)
            tags_df.loc[mask, "llm_strength"] = result["llm_strength"]
            tags_df.loc[mask, "failure_mode"] = result["failure_mode"]
            tags_df.loc[mask, "requires_collaborative_signal"] = result["requires_collaborative_signal"]
            tags_df.loc[mask, "ground_truth_plausibility"] = result["ground_truth_plausibility"]
            tags_df.loc[mask, "distractor_hardness"] = result["distractor_hardness"]
            tags_df.loc[mask, "confidence"] = result["confidence"]
            tags_df.loc[mask, "evidence"] = result["evidence"]

        tags_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"    Saved progress: {output_path}")

        if start + args.batch_size < len(unprocessed):
            time.sleep(5)

    merged_for_summary = tags_df.merge(df[["index", "hit"]], on="index", how="left")
    summarize(merged_for_summary)

    if args.merge_output:
        merged_path = os.path.join(actual_output_dir, "problem_semantic_tags_merged.csv")
        merged = df.merge(tags_df, on="index", how="left", suffixes=("", "_semantic"))
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print(f">>> Saved merged file: {merged_path}")


if __name__ == "__main__":
    main()
