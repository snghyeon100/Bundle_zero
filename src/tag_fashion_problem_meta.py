import argparse
import json
import os
import re
import time

import pandas as pd


TAGS = {
    "category_completion": "The ground-truth item completes an obvious missing functional category, and this category gap is the strongest reason it fits better than the distractors.",
    "season_match": "Seasonal coherence is the main signal, such as summer items matching summer outfits or winter items matching winter outfits.",
    "brand_or_collection_match": "The ground truth is supported by the same brand, shop, collection, year, product line, or named series.",
    "style_theme_match": "The key signal is a shared fashion style, mood, occasion, or aesthetic, such as casual, vintage, street, office, cute, elegant, French, Korean, or resort.",
    "color_material_pattern_match": "The key signal is shared or complementary color, material, texture, or pattern, such as red, denim, wool, lace, leather, stripe, floral, or leopard.",
    "gender_or_age_filtering": "The problem is mainly solved by filtering by gender, age group, or target wearer, such as men, women, girls, students, or kids.",
    "fine_grained_hard_choice": "Multiple candidates are broadly plausible and the ground truth requires subtle fashion compatibility judgment.",
    "ambiguous_or_counterintuitive_gt": "The ground truth is ambiguous, weakly supported, or less intuitive than at least one distractor.",
}


BASE_COLUMNS = [
    "index",
    "bundle_id",
    "true_indice",
    "true_option_idx",
    "true_option_char",
    "input_indices",
    "candidate_indices",
    "input_str",
    "target_str",
]


def load_config(config_path):
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_dataset_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    base = os.path.basename(path).lower()
    for dataset in ("pog_dense", "pog"):
        if dataset in base:
            return dataset
    return None


def default_input_path(output_dir, dataset):
    return os.path.join(output_dir, dataset, "problem_meta_clean.csv")


def short_text(value, limit=1800):
    value = re.sub(r"\s+", " ", str(value)).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def extract_gt_item(target_str, true_option_char):
    char = re.escape(str(true_option_char).strip().upper())
    text = str(target_str)
    pattern = rf"(?:^|;\s*){char}\.\s*(.*?)(?=;\s*[A-Z]\.|$)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def clean_json_text(text):
    text = (text or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        text = text[start : end + 1]
    return text


def build_prompt(row):
    tag_lines = "\n".join([f"- {tag}: {desc}" for tag, desc in TAGS.items()])
    gt_item = row.get("true_item_str", "")
    if pd.isna(gt_item) or not str(gt_item).strip():
        gt_item = extract_gt_item(row["target_str"], row["true_option_char"])

    return f"""
You are an expert fashion recommendation researcher.
Assign reusable problem-level tags for a zero-shot fashion bundle construction case.

Important:
- Tag the problem itself, not any model prediction.
- You will see only the input outfit items, candidate items, and ground-truth item.
- Choose exactly one primary_tag from the taxonomy.
- secondary_tags may contain 0 to 2 tags from the taxonomy.
- Do not invent tags outside the taxonomy.
- Keep evidence short and grounded in the item text.

Primary tag selection rules:
- Choose primary_tag based on the strongest reason why the ground-truth item fits the input better than the distractors.
- Use secondary_tags only for additional signals that are clearly present.
- Do not add secondary tags for weak or speculative signals.
- Do not choose category_completion just because the ground-truth item adds another clothing/accessory type. In fashion bundles, that is often true by default.
- Choose category_completion as primary only when the input clearly creates a functional gap and the ground truth is selected mainly because it fills that gap, while distractors mostly duplicate existing categories or are wrong categories.
- If a clear explicit signal dominates, prefer the corresponding explicit tag over broad style_theme_match.
- If season, brand/collection, gender/age, color/material/pattern, or style/theme is more decisive than the item category, choose that tag as primary and put category_completion only as a secondary tag if needed.
- If multiple candidates are broadly plausible and the ground truth requires subtle distinction, prefer fine_grained_hard_choice.
- If the ground truth itself appears weak, surprising, or less natural than at least one distractor, prefer ambiguous_or_counterintuitive_gt.
- Use style_theme_match only when the main signal is an overall fashion mood/style rather than category, season, brand, gender, color, material, or pattern.

Taxonomy:
{tag_lines}

Case:
Index: {row["index"]}
Bundle ID: {row["bundle_id"]}
Input outfit items:
{short_text(row["input_str"])}

Candidate items:
{short_text(row["target_str"])}

Ground-truth option: {row["true_option_char"]}
Ground-truth item: {short_text(gt_item) if gt_item else "(see candidate list)"}

Return ONLY one JSON object with this schema:
{{
  "index": {int(row["index"])},
  "primary_tag": "season_match",
  "secondary_tags": ["category_completion"],
  "gt_plausibility": 4,
  "distractor_hardness": 2,
  "confidence": 0.75,
  "evidence": "The input establishes a winter outfit, and the ground-truth boots match the season and complete the outfit."
}}

Field rules:
- gt_plausibility: integer 1-5, where 1 means the ground truth is counterintuitive and 5 means very obvious.
- distractor_hardness: integer 1-5, where 1 means distractors are easy/out-of-context and 5 means distractors are very similar/plausible.
- confidence: number from 0.0 to 1.0.
""".strip()


def validate_result(result, expected_index):
    if int(result.get("index", expected_index)) != int(expected_index):
        result["index"] = int(expected_index)

    primary = result.get("primary_tag", "ambiguous_or_counterintuitive_gt")
    if primary not in TAGS:
        primary = "ambiguous_or_counterintuitive_gt"
    result["primary_tag"] = primary

    secondary = result.get("secondary_tags", [])
    if not isinstance(secondary, list):
        secondary = []
    result["secondary_tags"] = [tag for tag in secondary if tag in TAGS and tag != primary][:2]

    for key in ("gt_plausibility", "distractor_hardness"):
        try:
            result[key] = max(1, min(5, int(result.get(key, 3))))
        except (TypeError, ValueError):
            result[key] = 3

    try:
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    except (TypeError, ValueError):
        result["confidence"] = 0.5

    result["evidence"] = str(result.get("evidence", "")).strip()
    return result


def call_tag_api(client, model, row, max_retries, base_wait):
    prompt = build_prompt(row)
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
            parsed = json.loads(clean_json_text(response.text))
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object.")
            return validate_result(parsed, row["index"])
        except Exception as exc:
            last_error = exc
            wait = base_wait * (attempt + 1)
            print(f">>> API error at index {row['index']}: {exc} | retrying in {wait}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"Failed index {row['index']} after {max_retries} retries: {last_error}")


def make_output_frame(input_df, existing_path):
    if os.path.exists(existing_path):
        out_df = pd.read_csv(existing_path)
        print(f">>> Resuming existing output: {existing_path}")
    else:
        out_df = input_df[BASE_COLUMNS].copy()
        for col in [
            "primary_tag",
            "secondary_tags",
            "gt_plausibility",
            "distractor_hardness",
            "confidence",
            "evidence",
        ]:
            out_df[col] = pd.NA
    return out_df


def summarize(out_df):
    done = out_df[out_df["primary_tag"].notna()]
    print("\n" + "=" * 60)
    print("Fashion Semantic Tag Summary")
    print("=" * 60)
    print(f"Tagged: {len(done)}/{len(out_df)}")
    if not done.empty:
        for tag, count in done["primary_tag"].value_counts().items():
            print(f"- {tag}: {count} ({count / len(done) * 100:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Tag POG/Pog-dense problem_meta_clean.csv with one fashion semantic API call per problem.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path.")
    parser.add_argument("--input", default=None, help="Path to problem_meta_clean.csv.")
    parser.add_argument("--dataset", default=None, help="pog or pog_dense. Inferred from input path if omitted.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Gemini model for tagging.")
    parser.add_argument("--limit", type=int, default=-1, help="Only tag first N untagged rows.")
    parser.add_argument("--start_idx", type=int, default=None, help="Only tag rows with index >= this value.")
    parser.add_argument("--sleep", type=float, default=5.0, help="Sleep seconds after each successful API call.")
    parser.add_argument("--retry_wait", type=float, default=20.0, help="Base wait seconds after an API error.")
    parser.add_argument("--max_retries", type=int, default=10, help="Max retries per problem.")
    args = parser.parse_args()

    conf = load_config(args.config)
    output_dir = conf.get("output_dir", "./results")

    dataset = args.dataset or (infer_dataset_from_path(args.input) if args.input else None) or conf.get("dataset")
    if dataset not in {"pog", "pog_dense"}:
        raise ValueError("This script is intended for pog or pog_dense only. Pass --dataset pog or --dataset pog_dense.")

    input_path = args.input or default_input_path(output_dir, dataset)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input problem meta not found: {input_path}")

    output_path = args.output or os.path.join(output_dir, dataset, "problem_fashion_semantic_tags.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    input_df = pd.read_csv(input_path)
    missing = [col for col in BASE_COLUMNS if col not in input_df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    out_df = make_output_frame(input_df, output_path)
    tagged_indices = set(out_df[out_df["primary_tag"].notna()]["index"].astype(int).tolist())
    pending = [int(idx) for idx in input_df["index"].tolist() if int(idx) not in tagged_indices]

    if args.start_idx is not None:
        pending = [idx for idx in pending if idx >= args.start_idx]
    if args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        print(">>> All requested rows are already tagged.")
        summarize(out_df)
        return

    from dotenv import load_dotenv
    from google import genai

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path, encoding="utf-8-sig")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY not found in .env.")

    client = genai.Client(api_key=api_key)

    print(f">>> Dataset : {dataset}")
    print(f">>> Input   : {input_path}")
    print(f">>> Output  : {output_path}")
    print(f">>> Pending : {len(pending)}")

    for n, idx in enumerate(pending, start=1):
        row = input_df.loc[input_df["index"].astype(int) == idx].iloc[0]
        print(f">>> Tagging {n}/{len(pending)} | index={idx} | bundle_id={row['bundle_id']}")

        result = call_tag_api(client, args.model, row, args.max_retries, args.retry_wait)
        mask = out_df["index"].astype(int) == idx
        out_df.loc[mask, "primary_tag"] = result["primary_tag"]
        out_df.loc[mask, "secondary_tags"] = json.dumps(result["secondary_tags"], ensure_ascii=False)
        out_df.loc[mask, "gt_plausibility"] = result["gt_plausibility"]
        out_df.loc[mask, "distractor_hardness"] = result["distractor_hardness"]
        out_df.loc[mask, "confidence"] = result["confidence"]
        out_df.loc[mask, "evidence"] = result["evidence"]
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"    Saved: {result['primary_tag']}")

        if n < len(pending):
            time.sleep(args.sleep)

    summarize(out_df)


if __name__ == "__main__":
    main()
