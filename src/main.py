import os
import re
import sys
import yaml
import json
import math
import time
import asyncio
import glob
import pandas as pd
from dotenv import load_dotenv
from google import genai
from dataset import BundleZeroShotDataset, set_seed

# Load Env 
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path, encoding='utf-8-sig')

def parse_model_response(raw_text):
    if not raw_text:
        return "ERR_EM"
    # Remove common prefixes like 'Choice:', 'Option:', 'Answer:' (case-insensitive)
    clean_text = re.sub(r'(?i)\b(choice|option|answer)\b[\s]*[:=]*[\s]*', '', raw_text.strip())
    # Extract the first uppercase letter (A-Z)
    match = re.search(r'([A-Z])', clean_text.upper())
    return match.group(1) if match else raw_text.strip()[0].upper()

def option_letters(num_cans):
    return [chr(ord('A') + i) for i in range(int(num_cans))]

def is_ranking_mode(conf):
    return str(conf.get("prediction_mode", "choice")).strip().lower() in {"ranking", "rank"}

def generation_max_output_tokens(conf):
    if is_ranking_mode(conf):
        return int(conf.get("ranking_max_output_tokens", 160))
    return int(conf.get("max_output_tokens", 10))

def _strip_json_fence(raw_text):
    text = str(raw_text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def parse_model_ranking(raw_text, num_cans, fill_missing=True):
    letters = option_letters(num_cans)
    allowed = set(letters)
    text = _strip_json_fence(raw_text)
    parsed = []

    json_candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        json_candidates.append(match.group(0))
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        json_candidates.append(match.group(0))

    for candidate in json_candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            obj = obj.get("ranking", obj.get("rank", obj.get("choices", obj.get("order", []))))
        if isinstance(obj, list):
            for value in obj:
                value_text = str(value).upper()
                letter_match = re.search(rf"(?<![A-Z])([{letters[0]}-{letters[-1]}])(?![A-Z])", value_text)
                if letter_match:
                    parsed.append(letter_match.group(1))
            break

    if not parsed:
        upper_text = text.upper()
        parsed = re.findall(rf"(?<![A-Z])([{letters[0]}-{letters[-1]}])(?![A-Z])", upper_text)

    deduped = []
    seen = set()
    duplicate_found = False
    invalid_found = False
    for letter in parsed:
        if letter not in allowed:
            invalid_found = True
            continue
        if letter in seen:
            duplicate_found = True
            continue
        deduped.append(letter)
        seen.add(letter)

    ranking_valid = (
        len(deduped) == len(letters)
        and set(deduped) == allowed
        and not duplicate_found
        and not invalid_found
    )

    if deduped and fill_missing:
        deduped.extend([letter for letter in letters if letter not in seen])

    return deduped, ranking_valid

def evaluate_model_output(raw_text, true_option_char, conf):
    if is_ranking_mode(conf):
        ranking, ranking_valid = parse_model_ranking(
            raw_text,
            conf.get("num_cans", 10),
            fill_missing=conf.get("ranking_fill_missing", True),
        )
        prediction = ranking[0] if ranking else "ERR_RANK"
        true_rank = ranking.index(true_option_char) + 1 if true_option_char in ranking else None
        mrr = 1.0 / true_rank if true_rank else 0.0

        def hit_at(k):
            return int(true_rank is not None and true_rank <= k)

        def ndcg_at(k):
            if true_rank is None or true_rank > k:
                return 0.0
            return 1.0 / math.log2(true_rank + 1)

        return {
            "prediction": prediction,
            "ranking": json.dumps(ranking, ensure_ascii=False),
            "ranking_valid": int(ranking_valid),
            "true_rank": true_rank if true_rank is not None else "",
            "hit": hit_at(1),
            "hit_at_1": hit_at(1),
            "hit_at_3": hit_at(3),
            "hit_at_5": hit_at(5),
            "mrr": mrr,
            "ndcg_at_3": ndcg_at(3),
            "ndcg_at_5": ndcg_at(5),
            "ndcg_at_10": ndcg_at(min(10, int(conf.get("num_cans", 10)))),
        }

    pred_text = parse_model_response(raw_text)
    return {
        "prediction": pred_text,
        "hit": int(pred_text == true_option_char),
    }

def console_safe_text(text):
    encoding = sys.stdout.encoding or "utf-8"
    return str(text).encode(encoding, errors="backslashreplace").decode(encoding)

def pluralize(count, singular, plural=None):
    return singular if int(count) == 1 else (plural or f"{singular}s")

def print_first_qa_debug(sample, conf, text_prompt=None):
    print("\n[DEBUG] First QA Preview:")
    print(f"  [Bundle ID] {sample.get('bundle_id')}")
    print(f"  [True Option] {sample.get('true_option_char')} | True Item ID: {sample.get('true_indice')}")
    print(f"  [Input Item IDs] {sample.get('input_indices')}")
    print(f"  [Candidate Item IDs] {sample.get('candidate_indices')}")
    print(f"  [Prediction Mode] {conf.get('prediction_mode', 'choice')}")
    if is_ranking_mode(conf):
        print(f"  [Ranking max output tokens] {conf.get('ranking_max_output_tokens', '')}")
        print(f"  [Ranking fill missing] {conf.get('ranking_fill_missing', True)}")
    print(f"  [Input Item Description Aug Enabled] {conf.get('use_input_item_description_aug', False)}")
    if conf.get("use_input_item_description_aug", False):
        print(f"  [Input Item Description root] {conf.get('input_item_description_cache_root', '')}")
        print(f"  [Input Item Description field] {conf.get('input_item_description_field', '')}")
    print(f"  [UI Category Purchase Prior Enabled] {conf.get('use_ui_category_purchase_prior', False)}")
    if conf.get("use_ui_category_purchase_prior", False):
        print(f"  [UI Category Purchase Prior top-k] {conf.get('ui_category_purchase_prior_top_k', '')}")
        print(f"  [UI Category Purchase Prior min support] {conf.get('ui_category_purchase_prior_min_support', '')}")
    print(f"  [Co-occurrence Enabled] {conf.get('use_cooccurrence', False)}")
    print(f"  [Soft Co-occurrence Enabled] {conf.get('use_soft_cooccurrence', False)}")
    if conf.get("use_soft_cooccurrence", False):
        print(f"  [Soft Co-occurrence Source] {conf.get('soft_cooccurrence_source', '')}")
    print(f"  [Item Affiliation Enabled] {conf.get('use_item_bundle_affiliation_desc', False)}")
    if conf.get("use_item_bundle_affiliation_desc", False):
        print(f"  [Item Affiliation k] {conf.get('item_bundle_affiliation_k', '')}")
        print(f"  [Item Affiliation alpha] {conf.get('item_bundle_affiliation_alpha', '')}")
        print(f"  [Exclude Query Items] {conf.get('item_bundle_affiliation_exclude_query_items', False)}")
        print(f"  [Item Affiliation Use Soft] {conf.get('item_bundle_affiliation_use_soft', False)}")
        if conf.get("item_bundle_affiliation_use_soft", False):
            print(f"  [Item Affiliation Soft Source] {conf.get('item_bundle_affiliation_soft_source', '')}")
            print(f"  [Item Affiliation Soft Alpha] {conf.get('item_bundle_affiliation_soft_alpha', '')}")
    print(f"  [User Co-purchase Enabled] {conf.get('use_item_user_copurchase_desc', False)}")
    if conf.get("use_item_user_copurchase_desc", False):
        print(f"  [User Co-purchase k] {conf.get('item_user_copurchase_k', '')}")
        print(f"  [User Co-purchase alpha] {conf.get('item_user_copurchase_alpha', '')}")
        print(f"  [Exclude Query Items] {conf.get('item_user_copurchase_exclude_query_items', False)}")
    print(f"  [Bundle Graph Context Enabled] {conf.get('use_bundle_graph_context', False)}")
    if conf.get("use_bundle_graph_context", False):
        print(f"  [Bundle Graph Context k] {conf.get('bundle_graph_context_k', '')}")
        print(f"  [Bundle Graph Context max items] {conf.get('bundle_graph_context_max_items', '')}")
        print(f"  [Bundle Graph Context Use Soft] {conf.get('bundle_graph_context_use_soft', False)}")
        if conf.get("bundle_graph_context_use_soft", False):
            print(f"  [Bundle Graph Context Soft Source] {conf.get('bundle_graph_context_soft_source', '')}")
            print(f"  [Bundle Graph Context Soft Alpha] {conf.get('bundle_graph_context_soft_alpha', '')}")
    print(f"  [Category Evidence Summary Enabled] {conf.get('use_category_evidence_summary', False)}")
    if conf.get("use_category_evidence_summary", False):
        print(f"  [Category Evidence Summary k] {conf.get('category_evidence_summary_k', '')}")
        print(f"  [Category Evidence Summary include evidence] {conf.get('category_evidence_summary_include_evidence', False)}")
        print(f"  [Category Evidence Summary model] {conf.get('category_evidence_summary_model', '') or conf.get('model', '')}")
        print(f"  [Category Evidence Summary API key env] {conf.get('category_evidence_summary_api_key_env', '') or 'main client'}")
    print(f"  [C-C Retrieval Context Enabled] {conf.get('use_cc_retrieval_context', False)}")
    if conf.get("use_cc_retrieval_context", False):
        print(f"  [C-C Retrieval Context k] {conf.get('cc_retrieval_context_k', '')}")
        print(f"  [C-C Retrieval Context Seed] {conf.get('cc_retrieval_context_seed', '')}")
        print(f"  [C-C Retrieval Overlap Weight] {conf.get('cc_retrieval_overlap_weight', '')}")
        print(f"  [C-C Retrieval Extra Weight] {conf.get('cc_retrieval_extra_weight', '')}")
    print(f"  [Category Completion Prior Enabled] {conf.get('use_category_completion_prior_desc', False)}")
    if conf.get("use_category_completion_prior_desc", False):
        print(f"  [Category Prior top-k] {conf.get('category_prior_top_k', '')}")
        print(f"  [Category Prior verbalization] {conf.get('category_prior_verbalization', '')}")
        print(f"  [Representative items/category] {conf.get('category_prior_rep_items_per_category', '')}")
        print(f"  [Category Prior min support] {conf.get('category_prior_min_support', '')}")
    print(f"  [Category Item Text Aug Enabled] {conf.get('use_category_item_text_aug', False)}")
    if conf.get("use_category_item_text_aug", False):
        print(f"  [Category Item Aug apply-to] {conf.get('category_item_aug_apply_to', '')}")
        print(f"  [Category Item Aug reps/category] {conf.get('category_item_aug_rep_items_per_category', '')}")
    print(f"  [Category Name Aug Enabled] {conf.get('use_category_name_aug', False)}")
    if conf.get("use_category_name_aug", False):
        print(f"  [Category Name Aug apply-to] {conf.get('category_name_aug_apply_to', '')}")
        print(f"  [Category Name field] {conf.get('category_name_field', '')}")
        print(f"  [Category Name root] {conf.get('category_name_root', '')}")
    print(f"  [Input Category Co-occur Enabled] {conf.get('input_category_co_occur', False)}")
    if conf.get("input_category_co_occur", False):
        print(f"  [Input Category Co-occur apply-to] {conf.get('input_category_co_occur_apply_to', '')}")
        print(f"  [Input Category Co-occur verbalization] {conf.get('input_category_co_occur_verbalization', '')}")
        print(f"  [Input Category Co-occur top-k] {conf.get('input_category_co_occur_top_k', '')}")
        print(f"  [Input Category Co-occur reps/category] {conf.get('input_category_co_occur_rep_items_per_category', '')}")
    print("\n[DEBUG] First Question:")
    print(console_safe_text(sample.get("input_str", "")))
    print("\n[DEBUG] First Options:")
    print(console_safe_text(sample.get("target_str", "")))
    if text_prompt is not None:
        print("\n[DEBUG] First Prompt Sent To Model:")
        print(console_safe_text(text_prompt))
    print("-" * 50 + "\n")

def generate_prompt(dataset_name, input_str, target_str, use_multimodal=False,
                    use_cooccurrence=False, use_soft_cooccurrence=False, soft_cooccurrence_source="",
                    icl_example=None, user_context_block="", bundle_graph_context_block="",
                    category_prior_context_block="", ui_category_purchase_prior_block="",
                    cc_retrieval_context_block="",
                    category_evidence_summary_block="",
                    use_image_category_completion_prompt=False,
                    prediction_mode="choice",
                    num_cans=10):
    if "spotify" in dataset_name:
        t_name = "playlist continuation"
        b_name = "music playlist"
        i_name = "song"
    else:
        t_name = "bundle construction"
        b_name = "fashion outfit"
        i_name = "fashion item"

    if use_multimodal:
        extra_instruction = (
            f"First infer the intent of the given {b_name}. Then, use the process of elimination: "
            f"evaluate each option, identify why the incorrect options do not fit the intent, "
            f"and eliminate them one by one until you find the best candidate {i_name}.\n"
        )
    else:
        extra_instruction = (
            f"First infer the intent of the given {b_name}. Then, use the process of elimination: "
            f"evaluate each option, identify why the incorrect options do not fit the intent, "
            f"and eliminate them one by one until you find the best candidate {i_name}.\n"
        )
    #extra_instruction = f"First infer the intent of the given {b_name}, and then choose the candidate {i_name} that fits that intent.\n"

    image_instruction = ""
    if use_multimodal:
        image_instruction = (
            "Use the images as visual references for the corresponding input items and candidate options, "
            "while also considering the item titles.\n"
        )
        if use_image_category_completion_prompt:
            image_instruction += (
                "Using both item titles and images, infer the coarse category of each input item and candidate option "
                "(e.g., top, bottom, dress, outerwear, shoes, bag, hat, jewelry, accessory, other). "
                "Prefer a candidate whose category naturally completes the partial outfit, while still prioritizing "
                "overall compatibility in style, season, gender, color, and occasion.\n"
            )

    if cc_retrieval_context_block:
        question_block = (
            f"Question: Given the partial {b_name}: {input_str}."
            f"{cc_retrieval_context_block}"
            f"Which candidate {i_name} should be included into this {b_name}?\n"
        )
    else:
        question_block = (
            f"Question: Given the partial {b_name}: {input_str}, "
            f"which candidate {i_name} should be included into this {b_name}?\n"
        )


    cf_legend = ""
    if use_cooccurrence or use_soft_cooccurrence:
        history_name = "past playlists" if "spotify" in dataset_name else "past outfits"
        cf_legend = f"Some options include a short history note from {history_name}.\n"

    icl_block = ""
    if icl_example:
        icl_block = (
            "Here is one solved example from historical training bundles:\n"
            f"Example question: Given the partial {b_name}: {icl_example['input_str']}, "
            f"which candidate {i_name} should be included into this {b_name}?\n"
            f"Example options: {icl_example['target_str']}\n"
            f"Correct answer: {icl_example['true_option_char']}. {icl_example['true_item_text']}\n\n"
            "Now solve the target question.\n"
        )

    ranking_mode = str(prediction_mode).strip().lower() in {"ranking", "rank"}
    if ranking_mode:
        letters = option_letters(num_cans)
        json_example = json.dumps({"ranking": letters}, separators=(",", ":"))
        task_instruction = (
            f"You should rank all candidate options from most likely to least likely to complete the {b_name}. "
            f"The first option in the ranking should be the single best completion.\n"
        )
        answer_instruction = (
            f"Rank all candidate options from most likely to least likely. "
            f"Return only valid JSON in this exact format: {json_example}\n"
            f"Rules: include each option letter exactly once, use only option letters {letters[0]}-{letters[-1]}, "
            f"do not include explanations, item names, or markdown.\nRanking: "
        )
    else:
        task_instruction = (
            f"You should directly answer the question by choosing the letter of the correct option. "
            f"Only provide the letter of your answer, without any explanation or mentioning the option content.\n"
        )
        answer_instruction = (
            f"Your answer should indicate your choice with a single letter (e.g., \"A,\" \"B,\" \"C,\" etc.).\nChoice: "
        )

    prompt = (
        f"You are a helpful and honest assistant. The following are multiple choice questions about {t_name}. "
        f"{task_instruction}"
        f"{image_instruction}"
        f"{cf_legend}"
        f"{icl_block}"
        f"{question_block}"
        f"{user_context_block}"
        f"{bundle_graph_context_block}"
        f"{ui_category_purchase_prior_block}"
        f"{category_prior_context_block}"
        f"{category_evidence_summary_block}"
        f"Options: {target_str}\n"
        #f"First, analyze the overall combination and coherence of the items in the {b_name}. Then, choose the candidate {i_name} that best completes the set."
        #f"{extra_instruction}"
        f"{answer_instruction}"
    )
    return prompt

def generate_category_evidence_summary_prompt(dataset_name, evidence_block):
    b_name = "music playlist" if "spotify" in dataset_name else "fashion outfit"
    return (
        "You are a careful assistant summarizing train-set historical category evidence for a "
        f"{b_name} completion task.\n"
        "You will see only input category names and retrieved historical outfit category evidence. "
        "Do not choose an answer and do not mention candidate options. "
        "Summarize the likely missing item roles/categories as a soft historical prior. "
        "Also mention likely duplicate roles to avoid when clear.\n\n"
        f"{evidence_block}\n\n"
        "Write 2-4 concise sentences. Start directly with the summary."
    )

def format_category_evidence_summary_block(summary_text, evidence_block="", include_evidence=False):
    summary_text = str(summary_text).strip()
    if not summary_text:
        return ""
    lines = [
        "Historical category summary:",
        summary_text,
    ]
    if include_evidence and evidence_block:
        lines.extend([
            "",
            "Retrieved category evidence used for the summary:",
            evidence_block.strip(),
        ])
    lines.append("Use this as a soft historical hint, while still choosing the candidate that best completes the given items.")
    return "\n".join(lines) + "\n\n"

async def generate_category_evidence_summary_async(client, model, conf, evidence_block, sample_idx=0, total=0):
    prompt = generate_category_evidence_summary_prompt(conf["dataset"], evidence_block)
    summary_model = conf.get("category_evidence_summary_model") or model
    max_tokens = int(conf.get("category_evidence_summary_max_output_tokens", 180))
    max_retries = 5
    base_delay = 10
    for attempt in range(max_retries):
        try:
            res = await client.aio.models.generate_content(
                model=summary_model,
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": max_tokens}
            )
            raw_text = res.text if res.text else ""
            return raw_text.strip(), raw_text, prompt
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "503" in err_str or "quota" in err_str or "demand" in err_str or "overloaded" in err_str:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (attempt + 1)
                    short_err = str(e).replace('\n', ' ')[:150]
                    prefix = f"[{sample_idx+1}/{total}] " if total else ""
                    print(f"{prefix}Summary API Error: {short_err}... | Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
            return "", str(e), prompt
    return "", "", prompt

def generate_category_evidence_summary_sync(client, model, conf, evidence_block):
    prompt = generate_category_evidence_summary_prompt(conf["dataset"], evidence_block)
    summary_model = conf.get("category_evidence_summary_model") or model
    max_tokens = int(conf.get("category_evidence_summary_max_output_tokens", 180))
    max_retries = 5
    base_delay = 10
    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model=summary_model,
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": max_tokens}
            )
            raw_text = res.text if res.text else ""
            return raw_text.strip(), raw_text, prompt
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "503" in err_str or "quota" in err_str or "demand" in err_str or "overloaded" in err_str:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (attempt + 1))
                    continue
            return "", str(e), prompt
    return "", "", prompt

def add_cooccurrence_to_options(sample, dataset, conf):
    if (
        not conf.get("use_cooccurrence", False)
        and not conf.get("use_soft_cooccurrence", False)
    ) or dataset is None:
        return sample["target_str"], {}

    cooc_stats = None
    if conf.get("use_cooccurrence", False):
        cooc_stats = dataset.get_cooccurrence_stats(
            sample.get("input_indices", []),
            sample.get("candidate_indices", [])
        )
    soft_cooc_stats = None
    if conf.get("use_soft_cooccurrence", False):
        soft_cooc_stats = dataset.get_soft_cooccurrence_stats(
            sample.get("input_indices", []),
            sample.get("candidate_indices", [])
        )

    options = sample["target_str"].split("; ")
    enriched_options = []
    is_spotify = "spotify" in conf.get("dataset", "")
    item_name = "song" if is_spotify else "item"
    item_plural = "songs" if is_spotify else "items"
    collection_name = "playlist" if is_spotify else "outfit"
    collection_plural = "playlists" if is_spotify else "outfits"
    for idx, option in enumerate(options):
        tags = []
        if cooc_stats is not None and idx < len(cooc_stats):
            stat = cooc_stats[idx]
            denom = stat["candidate_train_bundles"]
            shared = stat["shared_train_bundles"]
            tags.append(
                f"Past {collection_name} matches: this {item_name} appeared in {denom} "
                f"past {pluralize(denom, collection_name, collection_plural)}; "
                f"the given {item_plural} appeared in {shared} of them"
            )
        if soft_cooc_stats is not None and idx < len(soft_cooc_stats):
            stat = soft_cooc_stats[idx]
            denom = stat["candidate_train_bundles"]
            shared = stat["shared_train_bundles"]
            source = stat["source"]
            if source == "item_smoothing_text":
                tags.append(
                    f"Similar-{item_name} matches: this {item_name} appeared in {denom} "
                    f"past {pluralize(denom, collection_name, collection_plural)}; "
                    f"{item_plural} similar to the given {item_plural} appeared in {shared} of them"
                )
            else:
                tags.append(
                    f"Similar-{collection_name} matches: this {item_name} appeared in {denom} "
                    f"past {pluralize(denom, collection_name, collection_plural)}; "
                    f"{shared} of them were similar to {collection_plural} containing the given {item_plural}"
                )
        if tags:
            option = f"{option} [{' | '.join(tags)}]"
        enriched_options.append(option)

    signal_stats = {}
    if cooc_stats is not None:
        signal_stats["cooccurrence_stats"] = cooc_stats
    if soft_cooc_stats is not None:
        signal_stats["soft_cooccurrence_stats"] = soft_cooc_stats
    return "; ".join(enriched_options), signal_stats

def find_item_image(img_dir, item_id):
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        img_path = os.path.join(img_dir, f"{item_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    matches = glob.glob(os.path.join(img_dir, f"{item_id}.*"))
    if matches:
        return matches[0]
    return None

def build_interleaved_multimodal_contents(text_prompt, sample, enriched_target_str, img_dir):
    contents = []
    loaded_image_count = 0
    requested_image_ids = []
    found_image_paths = []
    missing_image_ids = []
    failed_image_paths = []

    def append_items_with_images(item_texts, item_ids):
        nonlocal loaded_image_count
        for idx, item_text in enumerate(item_texts):
            if idx > 0:
                contents.append("; ")
            contents.append(item_text)
            if idx >= len(item_ids):
                continue
            item_id = item_ids[idx]
            requested_image_ids.append(int(item_id))
            img_path = find_item_image(img_dir, item_id)
            if img_path:
                try:
                    contents.append(Image.open(img_path))
                    loaded_image_count += 1
                    found_image_paths.append(img_path)
                except Exception as e:
                    failed_image_paths.append(f"{img_path} ({e})")
            else:
                missing_image_ids.append(int(item_id))

    input_str = sample["input_str"]
    target_str = enriched_target_str
    try:
        before_input, after_input = text_prompt.split(input_str, 1)
        before_options, after_options = after_input.split(target_str, 1)
    except ValueError:
        contents.append(text_prompt)
        return contents, {
            "loaded_image_count": loaded_image_count,
            "requested_image_ids": requested_image_ids,
            "found_image_paths": found_image_paths,
            "missing_image_ids": missing_image_ids,
            "failed_image_paths": failed_image_paths,
        }

    contents.append(before_input)
    append_items_with_images(input_str.split("; "), sample.get("input_indices", []))
    contents.append(before_options)
    append_items_with_images(target_str.split("; "), sample.get("candidate_indices", []))
    contents.append(after_options)

    return contents, {
        "loaded_image_count": loaded_image_count,
        "requested_image_ids": requested_image_ids,
        "found_image_paths": found_image_paths,
        "missing_image_ids": missing_image_ids,
        "failed_image_paths": failed_image_paths,
    }

def save_intermediate_results(results, conf, timestamp, is_final=False):
    df = pd.DataFrame(results)
    hit_rate = df['hit'].mean() if not df.empty else 0.0
    valid_options = option_letters(conf.get("num_cans", 10))
    if is_ranking_mode(conf):
        valid_mask = df['ranking_valid'].astype(bool) if 'ranking_valid' in df.columns else pd.Series(False, index=df.index)
    else:
        valid_mask = df['prediction'].isin(valid_options)
    valid_ratio = valid_mask.mean() if not df.empty else 0.0
    valid_only_hit_rate = df.loc[valid_mask, 'hit'].mean() if valid_mask.sum() > 0 else 0.0
    
    df['overall_hit_rate'] = hit_rate
    df['overall_valid_ratio'] = valid_ratio
    df['valid_only_hit_rate'] = valid_only_hit_rate
    if is_ranking_mode(conf):
        true_rank_numeric = pd.to_numeric(df.get('true_rank', pd.Series(dtype=float)), errors='coerce')
        df['overall_hit_at_3'] = df['hit_at_3'].mean() if 'hit_at_3' in df.columns and not df.empty else 0.0
        df['overall_hit_at_5'] = df['hit_at_5'].mean() if 'hit_at_5' in df.columns and not df.empty else 0.0
        df['overall_mrr'] = df['mrr'].mean() if 'mrr' in df.columns and not df.empty else 0.0
        df['overall_ndcg_at_3'] = df['ndcg_at_3'].mean() if 'ndcg_at_3' in df.columns and not df.empty else 0.0
        df['overall_ndcg_at_5'] = df['ndcg_at_5'].mean() if 'ndcg_at_5' in df.columns and not df.empty else 0.0
        df['overall_ndcg_at_10'] = df['ndcg_at_10'].mean() if 'ndcg_at_10' in df.columns and not df.empty else 0.0
        df['overall_mean_rank'] = true_rank_numeric.mean() if true_rank_numeric.notna().any() else 0.0
        df['overall_median_rank'] = true_rank_numeric.median() if true_rank_numeric.notna().any() else 0.0
        df['valid_only_mrr'] = df.loc[valid_mask, 'mrr'].mean() if valid_mask.sum() > 0 and 'mrr' in df.columns else 0.0
    df['cfg_num_cans'] = conf.get("num_cans", "")
    df['cfg_num_token'] = conf.get("num_token", "")
    df['cfg_toy_eval'] = conf.get("toy_eval", "")
    df['cfg_prediction_mode'] = conf.get("prediction_mode", "choice")
    df['cfg_ranking_max_output_tokens'] = conf.get("ranking_max_output_tokens", "")
    df['cfg_ranking_fill_missing'] = conf.get("ranking_fill_missing", "")
    df['cfg_seed'] = conf.get("seed", "")
    df['cfg_shuffle_seed'] = conf.get("shuffle_seed", "")
    df['cfg_use_fixed_test_split'] = conf.get("use_fixed_test_split", False)
    df['cfg_test_input_file'] = conf.get("test_input_file", "")
    df['cfg_test_gt_file'] = conf.get("test_gt_file", "")
    df['cfg_use_hard_negative'] = conf.get("use_hard_negative", False)
    df['cfg_use_hard_negative_effective'] = conf.get("use_hard_negative", False) and not conf.get("use_fixed_test_split", False)
    df['cfg_use_cooccurrence'] = conf.get("use_cooccurrence", False)
    df['cfg_use_soft_cooccurrence'] = conf.get("use_soft_cooccurrence", False)
    df['cfg_soft_cooccurrence_source'] = conf.get("soft_cooccurrence_source", "")
    df['cfg_use_input_item_description_aug'] = conf.get("use_input_item_description_aug", False)
    df['cfg_input_item_description_cache_root'] = conf.get("input_item_description_cache_root", "")
    df['cfg_input_item_description_field'] = conf.get("input_item_description_field", "")
    df['cfg_use_ui_category_purchase_prior'] = conf.get("use_ui_category_purchase_prior", False)
    df['cfg_ui_category_purchase_prior_top_k'] = conf.get("ui_category_purchase_prior_top_k", "")
    df['cfg_ui_category_purchase_prior_min_support'] = conf.get("ui_category_purchase_prior_min_support", "")
    df['cfg_use_icl_retrieval'] = conf.get("use_icl_retrieval", False)
    df['cfg_icl_retrieval_method'] = conf.get("icl_retrieval_method", "")
    df['cfg_use_user_context'] = conf.get("use_user_context", False)
    df['cfg_user_context_selection'] = conf.get("user_context_selection", "")
    df['cfg_use_item_bundle_affiliation_desc'] = conf.get("use_item_bundle_affiliation_desc", False)
    df['cfg_item_bundle_affiliation_k'] = conf.get("item_bundle_affiliation_k", "")
    df['cfg_item_bundle_affiliation_alpha'] = conf.get("item_bundle_affiliation_alpha", "")
    df['cfg_item_bundle_affiliation_exclude_query_items'] = conf.get("item_bundle_affiliation_exclude_query_items", False)
    df['cfg_item_bundle_affiliation_use_soft'] = conf.get("item_bundle_affiliation_use_soft", False)
    df['cfg_item_bundle_affiliation_soft_source'] = conf.get("item_bundle_affiliation_soft_source", "")
    df['cfg_item_bundle_affiliation_soft_alpha'] = conf.get("item_bundle_affiliation_soft_alpha", "")
    df['cfg_use_item_user_copurchase_desc'] = conf.get("use_item_user_copurchase_desc", False)
    df['cfg_item_user_copurchase_k'] = conf.get("item_user_copurchase_k", "")
    df['cfg_item_user_copurchase_alpha'] = conf.get("item_user_copurchase_alpha", "")
    df['cfg_item_user_copurchase_exclude_query_items'] = conf.get("item_user_copurchase_exclude_query_items", False)
    df['cfg_use_bundle_graph_context'] = conf.get("use_bundle_graph_context", False)
    df['cfg_bundle_graph_context_k'] = conf.get("bundle_graph_context_k", "")
    df['cfg_bundle_graph_context_max_items'] = conf.get("bundle_graph_context_max_items", "")
    df['cfg_bundle_graph_context_use_soft'] = conf.get("bundle_graph_context_use_soft", False)
    df['cfg_bundle_graph_context_soft_source'] = conf.get("bundle_graph_context_soft_source", "")
    df['cfg_bundle_graph_context_soft_alpha'] = conf.get("bundle_graph_context_soft_alpha", "")
    df['cfg_use_category_evidence_summary'] = conf.get("use_category_evidence_summary", False)
    df['cfg_category_evidence_summary_k'] = conf.get("category_evidence_summary_k", "")
    df['cfg_category_evidence_summary_include_evidence'] = conf.get("category_evidence_summary_include_evidence", False)
    df['cfg_category_evidence_summary_model'] = conf.get("category_evidence_summary_model", "")
    df['cfg_category_evidence_summary_api_key_env'] = conf.get("category_evidence_summary_api_key_env", "")
    df['cfg_category_evidence_summary_max_output_tokens'] = conf.get("category_evidence_summary_max_output_tokens", "")
    df['cfg_use_cc_retrieval_context'] = conf.get("use_cc_retrieval_context", False)
    df['cfg_cc_retrieval_context_k'] = conf.get("cc_retrieval_context_k", "")
    df['cfg_cc_retrieval_context_seed'] = conf.get("cc_retrieval_context_seed", "")
    df['cfg_cc_retrieval_overlap_weight'] = conf.get("cc_retrieval_overlap_weight", "")
    df['cfg_cc_retrieval_extra_weight'] = conf.get("cc_retrieval_extra_weight", "")
    df['cfg_use_category_completion_prior_desc'] = conf.get("use_category_completion_prior_desc", False)
    df['cfg_use_category_item_text_aug'] = conf.get("use_category_item_text_aug", False)
    df['cfg_category_item_aug_apply_to'] = conf.get("category_item_aug_apply_to", "")
    df['cfg_category_item_aug_rep_items_per_category'] = conf.get("category_item_aug_rep_items_per_category", "")
    df['cfg_use_category_name_aug'] = conf.get("use_category_name_aug", False)
    df['cfg_category_name_aug_apply_to'] = conf.get("category_name_aug_apply_to", "")
    df['cfg_category_name_field'] = conf.get("category_name_field", "")
    df['cfg_category_name_root'] = conf.get("category_name_root", "")
    df['cfg_input_category_co_occur'] = conf.get("input_category_co_occur", False)
    df['cfg_input_category_co_occur_apply_to'] = conf.get("input_category_co_occur_apply_to", "")
    df['cfg_input_category_co_occur_verbalization'] = conf.get("input_category_co_occur_verbalization", "")
    df['cfg_input_category_co_occur_top_k'] = conf.get("input_category_co_occur_top_k", "")
    df['cfg_input_category_co_occur_rep_items_per_category'] = conf.get("input_category_co_occur_rep_items_per_category", "")
    df['cfg_category_prior_top_k'] = conf.get("category_prior_top_k", "")
    df['cfg_category_prior_verbalization'] = conf.get("category_prior_verbalization", "")
    df['cfg_category_prior_rep_items_per_category'] = conf.get("category_prior_rep_items_per_category", "")
    df['cfg_category_prior_min_support'] = conf.get("category_prior_min_support", "")
    df['cfg_category_prior_embedding_model'] = conf.get("category_prior_embedding_model", "")
    
    actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
    os.makedirs(actual_output_dir, exist_ok=True)
    ranking_str = "RANK_" if is_ranking_mode(conf) else ""
    cooc_str = "COOC_" if conf.get("use_cooccurrence", False) else ""
    soft_source = conf.get("soft_cooccurrence_source", "")
    soft_cooc_str = f"SOFTCOOC_{soft_source}_" if conf.get("use_soft_cooccurrence", False) else ""
    input_desc_str = "INPDESC_" if conf.get("use_input_item_description_aug", False) else ""
    ui_cat_purchase_str = "UICATPUR_" if conf.get("use_ui_category_purchase_prior", False) else ""
    hn_str = "HN_" if conf.get("use_hard_negative", False) else ""
    icl_str = "ICL_" if conf.get("use_icl_retrieval", False) else ""
    user_str = "USER_" if conf.get("use_user_context", False) else ""
    if conf.get("use_item_bundle_affiliation_desc", False) and conf.get("item_bundle_affiliation_use_soft", False):
        item_aff_str = f"ITEMAFF_SOFT_{conf.get('item_bundle_affiliation_soft_source', '')}_"
    else:
        item_aff_str = "ITEMAFF_" if conf.get("use_item_bundle_affiliation_desc", False) else ""
    user_pur_str = "USERPUR_" if conf.get("use_item_user_copurchase_desc", False) else ""
    if conf.get("use_bundle_graph_context", False) and conf.get("bundle_graph_context_use_soft", False):
        bundle_ctx_str = f"BGRAPH_SOFT_{conf.get('bundle_graph_context_soft_source', '')}_"
    else:
        bundle_ctx_str = "BGRAPH_" if conf.get("use_bundle_graph_context", False) else ""
    category_evidence_summary_str = "CATSUM_" if conf.get("use_category_evidence_summary", False) else ""
    cc_retrieval_str = "CCRET_" if conf.get("use_cc_retrieval_context", False) else ""
    category_prior_str = "CATPRIOR_" if conf.get("use_category_completion_prior_desc", False) else ""
    if conf.get("use_category_completion_prior_desc", False) and str(conf.get("category_prior_verbalization", "")).strip().lower() in {"category_names", "category_name", "names", "name"}:
        category_prior_str = "CATPRIORNAME_"
    category_item_aug_str = "CATITEMAUG_" if conf.get("use_category_item_text_aug", False) else ""
    category_name_aug_str = "CATNAMEAUG_" if conf.get("use_category_name_aug", False) else ""
    input_category_co_occur_str = ""
    if conf.get("input_category_co_occur", False):
        if str(conf.get("input_category_co_occur_verbalization", "")).strip().lower() in {"category_names", "category_name", "names", "name"}:
            input_category_co_occur_str = "INPCATNAMECOOC_"
        else:
            input_category_co_occur_str = "INPCATCOOC_"
    partial_str = "" if is_final else "_partial"
    save_path = os.path.join(actual_output_dir, f"results_{conf['dataset']}_{ranking_str}{icl_str}{user_str}{item_aff_str}{user_pur_str}{bundle_ctx_str}{input_desc_str}{ui_cat_purchase_str}{category_evidence_summary_str}{cc_retrieval_str}{category_prior_str}{category_item_aug_str}{category_name_aug_str}{input_category_co_occur_str}{cooc_str}{soft_cooc_str}{hn_str}C{conf.get('num_cans', '')}_T{conf.get('num_token', '')}_{timestamp}{partial_str}.csv")
    tmp_path = f"{save_path}.tmp"
    last_error = None
    for attempt in range(5):
        try:
            df.to_csv(tmp_path, index=False, encoding='utf-8-sig')
            os.replace(tmp_path, save_path)
            last_error = None
            break
        except OSError as e:
            last_error = e
            time.sleep(0.5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return save_path, df, hit_rate, valid_ratio, valid_only_hit_rate, valid_mask.sum()

async def process_sync_samples(client, model, samples, conf, timestamp, initial_results=None, start_idx=0, dataset=None, icl_retriever=None, user_context_retriever=None, summary_client=None):
    results = initial_results if initial_results is not None else []

    print(f">>> Processing {len(samples)} remaining samples sequentially to avoid rate limits...")
    total_samples_len = start_idx + len(samples)

    for idx, sample in enumerate(samples):
        current_idx = start_idx + idx
        enriched_target_str, signal_stats = add_cooccurrence_to_options(sample, dataset, conf)
        if signal_stats:
            sample.update(signal_stats)
        
        icl_example = None
        if icl_retriever is not None:
            icl_example = icl_retriever.retrieve(sample)
            sample.update(icl_retriever.metadata_for_csv(icl_example))

        if idx == 0 and icl_example is not None:
            print("\n[DEBUG] ICL Retrieval Check (First Sample):")
            print(f"  [Enabled] {conf.get('use_icl_retrieval', False)}")
            print(f"  [Method] {conf.get('icl_retrieval_method', '')}")
            print(f"  [Example Bundle ID] {icl_example['bundle_id']}")
            print(f"  [Retrieval Score] {icl_example['retrieval_score']:.6f}")
            print(f"  [Rank After Filter] {icl_example['retrieval_rank_after_filter']}")
            print(f"  [Input Overlap] {icl_example['query_input_overlap_count']}")
            print(f"  [Candidate Overlap] {icl_example['query_candidate_overlap_count']}")
            print(f"  [Example GT In Query Candidates] {icl_example['example_gt_in_query_candidates']}")
            print(f"  [Example Input IDs] {icl_example['input_indices']}")
            print(f"  [Example Candidate IDs] {icl_example['candidate_indices']}")
            print(f"  [Example Correct] {icl_example['true_option_char']}. {icl_example['true_item_text'][:120]}")
            print(f"  [Example Input Text] {icl_example['input_str'][:300]}...")
            print(f"  [Example Options Text] {icl_example['target_str'][:300]}...")
            print("-" * 50 + "\n")

        user_context = None
        user_context_block = ""
        if user_context_retriever is not None:
            user_context = user_context_retriever.retrieve(sample)
            sample.update(user_context_retriever.metadata_for_csv(user_context))
            user_context_block = user_context_retriever.format_context(user_context)

        if idx == 0 and user_context is not None:
            print("\n[DEBUG] User Context Check (First Sample):")
            print(f"  [Enabled] {conf.get('use_user_context', False)}")
            print(f"  [Selection] {conf.get('user_context_selection', '')}")
            print(f"  [User ID] {user_context['user_id']}")
            print(f"  [Input Overlap] {user_context['input_overlap_count']}")
            print(f"  [Tie Pool Size] {user_context['tie_pool_size']}")
            print(f"  [User History Size] {user_context['user_history_size']}")
            print(f"  [Selected Item IDs] {user_context['selected_item_indices']}")
            print(f"  [Selected Scores] {[round(x, 6) for x in user_context['selected_item_scores']]}")
            print(f"  [Context Text] {user_context_block[:500]}...")
            print("-" * 50 + "\n")

        bundle_graph_context = None
        bundle_graph_context_block = ""
        if conf.get("use_bundle_graph_context", False) and dataset is not None:
            bundle_graph_context = dataset.retrieve_bundle_graph_context(sample)
            if bundle_graph_context is not None:
                bundle_graph_context_block = bundle_graph_context["context_block"]
                sample.update(bundle_graph_context["metadata"])

        if idx == 0 and bundle_graph_context is not None:
            print("\n[DEBUG] Bundle Graph Context Check (First Sample):")
            print(f"  [Bundle IDs] {bundle_graph_context['metadata']['bundle_graph_context_bundle_ids']}")
            print(f"  [Overlap Counts] {bundle_graph_context['metadata']['bundle_graph_context_overlap_counts']}")
            print(f"  [Soft Hit Counts] {bundle_graph_context['metadata'].get('bundle_graph_context_soft_hit_counts', [])}")
            print(f"  [Context Scores] {bundle_graph_context['metadata'].get('bundle_graph_context_scores', [])}")
            print(f"  [IDF Scores] {[round(x, 6) for x in bundle_graph_context['metadata']['bundle_graph_context_idf_scores']]}")
            print(f"  [Context Text] {console_safe_text(bundle_graph_context_block[:1000])}...")
            print("-" * 50 + "\n")

        ui_category_purchase_prior = None
        ui_category_purchase_prior_block = ""
        if conf.get("use_ui_category_purchase_prior", False) and dataset is not None:
            ui_category_purchase_prior = dataset.retrieve_ui_category_purchase_prior_context(sample)
            if ui_category_purchase_prior is not None:
                ui_category_purchase_prior_block = ui_category_purchase_prior["context_block"]
                sample.update(ui_category_purchase_prior["metadata"])

        if idx == 0 and ui_category_purchase_prior is not None:
            print("\n[DEBUG] UI Category Purchase Prior Check (First Sample):")
            print(f"  [Input Categories] {ui_category_purchase_prior['metadata'].get('ui_category_purchase_prior_input_category_names', '')}")
            print(f"  [Top Categories] {ui_category_purchase_prior['metadata'].get('ui_category_purchase_prior_top_category_names', '')}")
            print(f"  [Scores] {ui_category_purchase_prior['metadata'].get('ui_category_purchase_prior_scores', '')}")
            print(f"  [Context Text] {console_safe_text(ui_category_purchase_prior_block[:1000])}...")
            print("-" * 50 + "\n")

        category_prior_context = None
        category_prior_context_block = ""
        if conf.get("use_category_completion_prior_desc", False) and dataset is not None:
            category_prior_context = dataset.retrieve_category_completion_prior_context(sample)
            if category_prior_context is not None:
                category_prior_context_block = category_prior_context["context_block"]
                sample.update(category_prior_context["metadata"])

        if idx == 0 and category_prior_context is not None:
            print("\n[DEBUG] Category Completion Prior Check (First Sample):")
            print(f"  [Observed Categories] {category_prior_context['metadata'].get('category_prior_observed_categories', '')}")
            print(f"  [Observed Support] {category_prior_context['metadata'].get('category_prior_observed_support', '')}")
            print(f"  [Verbalization] {category_prior_context['metadata'].get('category_prior_verbalization', '')}")
            print(f"  [Top Categories] {category_prior_context['metadata'].get('category_prior_top_categories', '')}")
            print(f"  [Top Category Names] {category_prior_context['metadata'].get('category_prior_top_category_names', '')}")
            print(f"  [Representative Item IDs] {category_prior_context['metadata'].get('category_prior_rep_item_ids', '')}")
            print(f"  [Context Text] {console_safe_text(category_prior_context_block[:1000])}...")
            print("-" * 50 + "\n")

        cc_retrieval_context = None
        cc_retrieval_context_block = ""
        if conf.get("use_cc_retrieval_context", False) and dataset is not None:
            cc_retrieval_context = dataset.retrieve_cc_retrieval_context(sample)
            if cc_retrieval_context is not None:
                cc_retrieval_context_block = cc_retrieval_context["context_block"]
                sample.update(cc_retrieval_context["metadata"])

        if idx == 0 and cc_retrieval_context is not None:
            print("\n[DEBUG] C-C Retrieval Context Check (First Sample):")
            print(f"  [Bundle IDs] {cc_retrieval_context['metadata'].get('cc_retrieval_context_bundle_ids', '')}")
            print(f"  [Scores] {cc_retrieval_context['metadata'].get('cc_retrieval_context_scores', '')}")
            print(f"  [Overlap Counts] {cc_retrieval_context['metadata'].get('cc_retrieval_context_overlap_counts', '')}")
            print(f"  [Extra Priors] {cc_retrieval_context['metadata'].get('cc_retrieval_context_extra_priors', '')}")
            print(f"  [Jaccards] {cc_retrieval_context['metadata'].get('cc_retrieval_context_jaccards', '')}")
            print(f"  [Context Text] {console_safe_text(cc_retrieval_context_block[:1000])}...")
            print("-" * 50 + "\n")

        category_evidence_summary_context = None
        category_evidence_summary_block = ""
        if conf.get("use_category_evidence_summary", False) and dataset is not None:
            category_evidence_summary_context = dataset.retrieve_category_evidence_summary_context(sample)
            if category_evidence_summary_context is not None:
                evidence_block = category_evidence_summary_context["evidence_block"]
                sample.update(category_evidence_summary_context["metadata"])
                sample["category_evidence_summary_evidence"] = evidence_block
                summary_text, summary_raw, summary_prompt = await generate_category_evidence_summary_async(
                    summary_client or client,
                    model,
                    conf,
                    evidence_block,
                    sample_idx=current_idx,
                    total=total_samples_len,
                )
                sample["category_evidence_summary"] = summary_text
                sample["category_evidence_summary_raw_response"] = summary_raw
                sample["category_evidence_summary_prompt"] = summary_prompt
                category_evidence_summary_block = format_category_evidence_summary_block(
                    summary_text,
                    evidence_block=evidence_block,
                    include_evidence=conf.get("category_evidence_summary_include_evidence", False),
                )

        if idx == 0 and category_evidence_summary_context is not None:
            print("\n[DEBUG] Category Evidence Summary Check (First Sample):")
            print(f"  [Selected Count] {category_evidence_summary_context['metadata'].get('category_evidence_summary_selected_count', '')}")
            print(f"  [Match Levels] {category_evidence_summary_context['metadata'].get('category_evidence_summary_match_levels', '')}")
            print(f"  [Summary] {console_safe_text(sample.get('category_evidence_summary', '')[:1000])}...")
            print(f"  [Evidence Text] {console_safe_text(sample.get('category_evidence_summary_evidence', '')[:1000])}...")
            print("-" * 50 + "\n")
        
        text_prompt = generate_prompt(
            conf["dataset"], sample["input_str"], enriched_target_str,
            use_multimodal=conf.get("use_multimodal", False),
            use_cooccurrence=conf.get("use_cooccurrence", False),
            use_soft_cooccurrence=conf.get("use_soft_cooccurrence", False),
            soft_cooccurrence_source=conf.get("soft_cooccurrence_source", ""),
            icl_example=icl_example,
            user_context_block=user_context_block,
            bundle_graph_context_block=bundle_graph_context_block,
            ui_category_purchase_prior_block=ui_category_purchase_prior_block,
            category_prior_context_block=category_prior_context_block,
            cc_retrieval_context_block=cc_retrieval_context_block,
            category_evidence_summary_block=category_evidence_summary_block,
            use_image_category_completion_prompt=conf.get("use_image_category_completion_prompt", False),
            prediction_mode=conf.get("prediction_mode", "choice"),
            num_cans=conf.get("num_cans", 10)
        )

        if idx == 0:
            debug_sample = dict(sample)
            debug_sample["target_str"] = enriched_target_str
            print_first_qa_debug(debug_sample, conf, text_prompt=text_prompt)
        
        contents = text_prompt
        loaded_image_count = 0
        requested_image_ids = []
        found_image_paths = []
        missing_image_ids = []
        failed_image_paths = []
        if conf.get("use_multimodal", False):
            img_dir = os.path.join(conf.get("data_path", "./datasets"), conf["dataset"], "images")
            contents, image_debug = build_interleaved_multimodal_contents(
                text_prompt,
                sample,
                enriched_target_str,
                img_dir,
            )
            loaded_image_count = image_debug["loaded_image_count"]
            requested_image_ids = image_debug["requested_image_ids"]
            found_image_paths = image_debug["found_image_paths"]
            missing_image_ids = image_debug["missing_image_ids"]
            failed_image_paths = image_debug["failed_image_paths"]

        if idx == 0 and conf.get("use_multimodal", False):
            print("\n[DEBUG] Multimodal Input Check (First Sample):")
            print(f"  [Image Count] {loaded_image_count}")
            print(f"  [Image Dir] {img_dir}")
            print(f"  [Requested Item IDs] {requested_image_ids}")
            if missing_image_ids:
                print(f"  [Missing Image IDs] {missing_image_ids}")
            if failed_image_paths:
                print(f"  [Failed Image Opens] {failed_image_paths[:5]}")
            if found_image_paths:
                preview_paths = [os.path.basename(p) for p in found_image_paths[:10]]
                print(f"  [Found Image Files] {preview_paths}")
            for c in contents:
                if isinstance(c, str):
                    print(f"  [Text] {c[:60]}..." if len(c) > 60 else f"  [Text] {c}")
                else:
                    print(f"  [Image] {getattr(c, 'filename', 'Unknown')} | Size: {c.size}")
            print("-" * 50 + "\n")

        max_retries = 10
        base_delay = 20
        
        for attempt in range(max_retries):
            try:
                res = await client.aio.models.generate_content(
                    model=model, 
                    contents=contents,
                    config={"temperature": conf["temperature"], "max_output_tokens": generation_max_output_tokens(conf)}
                )
                raw_text = res.text if res.text else ""
                pred_info = evaluate_model_output(raw_text, sample['true_option_char'], conf)
                break  # Success! Break out of the retry loop
            except Exception as e:
                err_str = str(e).lower()
                # Check for rate limits (429) or high demand/server errors (503)
                if "429" in err_str or "503" in err_str or "quota" in err_str or "demand" in err_str or "overloaded" in err_str:
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (attempt + 1)
                        # Print the original API error so quota/server issues are visible during retries.
                        short_err = str(e).replace('\n', ' ')[:150] 
                        print(f"[{current_idx+1}/{total_samples_len}] API Error: {short_err}... | Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                raw_text = str(e)
                pred_info = {"prediction": "ERR_EX", "hit": 0}
                break
        
        sample.update(pred_info)
        sample['raw_response'] = raw_text # Save verbatim output or error trace
        results.append(sample)
        
        # Save after each sample so interrupted runs can resume.
        save_intermediate_results(results, conf, timestamp, is_final=False)
        
        if is_ranking_mode(conf):
            print(f"[{current_idx+1}/{total_samples_len}] True: {sample['true_option_char']} | Pred: {sample['prediction']} | Rank: {sample.get('true_rank', '')}")
        else:
            print(f"[{current_idx+1}/{total_samples_len}] True: {sample['true_option_char']} | Pred: {sample['prediction']}")
        
        # Enforce rate limit (Dynamic based on model Free Tier limits)
        # Gemini 2.5 Flash / Pro -> 5 requests / min = 12s interval. (Using 13s)
        # Gemini 1.5 Flash / 3.x Lite -> 15 requests / min = 4s interval. (Using 4.5s)
        sleep_time = 15.0
        if "gemma" in model or "lite" in model.lower():
            sleep_time = 5
            
        if idx < len(samples) - 1:
            await asyncio.sleep(sleep_time)
            
    return results

def process_batch_samples(client, model, samples, conf, dataset=None, summary_client=None):
    print(">>> 1. Creating JSONL for Batch API...")
    jsonl_path = os.path.join(conf["output_dir"], f"batch_requests_{conf['dataset']}.jsonl")
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            enriched_target_str, signal_stats = add_cooccurrence_to_options(sample, dataset, conf)
            if signal_stats:
                sample.update(signal_stats)
            bundle_graph_context = None
            bundle_graph_context_block = ""
            if conf.get("use_bundle_graph_context", False) and dataset is not None:
                bundle_graph_context = dataset.retrieve_bundle_graph_context(sample)
                if bundle_graph_context is not None:
                    bundle_graph_context_block = bundle_graph_context["context_block"]
                    sample.update(bundle_graph_context["metadata"])
            ui_category_purchase_prior = None
            ui_category_purchase_prior_block = ""
            if conf.get("use_ui_category_purchase_prior", False) and dataset is not None:
                ui_category_purchase_prior = dataset.retrieve_ui_category_purchase_prior_context(sample)
                if ui_category_purchase_prior is not None:
                    ui_category_purchase_prior_block = ui_category_purchase_prior["context_block"]
                    sample.update(ui_category_purchase_prior["metadata"])
            category_prior_context = None
            category_prior_context_block = ""
            if conf.get("use_category_completion_prior_desc", False) and dataset is not None:
                category_prior_context = dataset.retrieve_category_completion_prior_context(sample)
                if category_prior_context is not None:
                    category_prior_context_block = category_prior_context["context_block"]
                    sample.update(category_prior_context["metadata"])
            cc_retrieval_context = None
            cc_retrieval_context_block = ""
            if conf.get("use_cc_retrieval_context", False) and dataset is not None:
                cc_retrieval_context = dataset.retrieve_cc_retrieval_context(sample)
                if cc_retrieval_context is not None:
                    cc_retrieval_context_block = cc_retrieval_context["context_block"]
                    sample.update(cc_retrieval_context["metadata"])
            category_evidence_summary_context = None
            category_evidence_summary_block = ""
            if conf.get("use_category_evidence_summary", False) and dataset is not None:
                category_evidence_summary_context = dataset.retrieve_category_evidence_summary_context(sample)
                if category_evidence_summary_context is not None:
                    evidence_block = category_evidence_summary_context["evidence_block"]
                    sample.update(category_evidence_summary_context["metadata"])
                    sample["category_evidence_summary_evidence"] = evidence_block
                    summary_text, summary_raw, summary_prompt = generate_category_evidence_summary_sync(
                        summary_client or client,
                        model,
                        conf,
                        evidence_block,
                    )
                    sample["category_evidence_summary"] = summary_text
                    sample["category_evidence_summary_raw_response"] = summary_raw
                    sample["category_evidence_summary_prompt"] = summary_prompt
                    category_evidence_summary_block = format_category_evidence_summary_block(
                        summary_text,
                        evidence_block=evidence_block,
                        include_evidence=conf.get("category_evidence_summary_include_evidence", False),
                    )
            prompt = generate_prompt(
                conf["dataset"],
                sample["input_str"],
                enriched_target_str,
                conf.get("use_multimodal", False),
                use_cooccurrence=conf.get("use_cooccurrence", False),
                use_soft_cooccurrence=conf.get("use_soft_cooccurrence", False),
                soft_cooccurrence_source=conf.get("soft_cooccurrence_source", ""),
                bundle_graph_context_block=bundle_graph_context_block,
                ui_category_purchase_prior_block=ui_category_purchase_prior_block,
                category_prior_context_block=category_prior_context_block,
                cc_retrieval_context_block=cc_retrieval_context_block,
                category_evidence_summary_block=category_evidence_summary_block,
                use_image_category_completion_prompt=conf.get("use_image_category_completion_prompt", False),
                prediction_mode=conf.get("prediction_mode", "choice"),
                num_cans=conf.get("num_cans", 10)
            )
            if idx == 0 and bundle_graph_context is not None:
                print("\n[DEBUG] Bundle Graph Context Check (First Sample):")
                print(f"  [Bundle IDs] {bundle_graph_context['metadata']['bundle_graph_context_bundle_ids']}")
                print(f"  [Overlap Counts] {bundle_graph_context['metadata']['bundle_graph_context_overlap_counts']}")
                print(f"  [Soft Hit Counts] {bundle_graph_context['metadata'].get('bundle_graph_context_soft_hit_counts', [])}")
                print(f"  [Context Scores] {bundle_graph_context['metadata'].get('bundle_graph_context_scores', [])}")
                print(f"  [IDF Scores] {[round(x, 6) for x in bundle_graph_context['metadata']['bundle_graph_context_idf_scores']]}")
                print(f"  [Context Text] {console_safe_text(bundle_graph_context_block[:1000])}...")
                print("-" * 50 + "\n")

            if idx == 0 and ui_category_purchase_prior is not None:
                print("\n[DEBUG] UI Category Purchase Prior Check (First Sample):")
                print(f"  [Input Categories] {ui_category_purchase_prior['metadata'].get('ui_category_purchase_prior_input_category_names', '')}")
                print(f"  [Top Categories] {ui_category_purchase_prior['metadata'].get('ui_category_purchase_prior_top_category_names', '')}")
                print(f"  [Scores] {ui_category_purchase_prior['metadata'].get('ui_category_purchase_prior_scores', '')}")
                print(f"  [Context Text] {console_safe_text(ui_category_purchase_prior_block[:1000])}...")
                print("-" * 50 + "\n")

            if idx == 0 and cc_retrieval_context is not None:
                print("\n[DEBUG] C-C Retrieval Context Check (First Sample):")
                print(f"  [Bundle IDs] {cc_retrieval_context['metadata'].get('cc_retrieval_context_bundle_ids', '')}")
                print(f"  [Scores] {cc_retrieval_context['metadata'].get('cc_retrieval_context_scores', '')}")
                print(f"  [Overlap Counts] {cc_retrieval_context['metadata'].get('cc_retrieval_context_overlap_counts', '')}")
                print(f"  [Extra Priors] {cc_retrieval_context['metadata'].get('cc_retrieval_context_extra_priors', '')}")
                print(f"  [Jaccards] {cc_retrieval_context['metadata'].get('cc_retrieval_context_jaccards', '')}")
                print(f"  [Context Text] {console_safe_text(cc_retrieval_context_block[:1000])}...")
                print("-" * 50 + "\n")

            if idx == 0 and category_evidence_summary_context is not None:
                print("\n[DEBUG] Category Evidence Summary Check (First Sample):")
                print(f"  [Selected Count] {category_evidence_summary_context['metadata'].get('category_evidence_summary_selected_count', '')}")
                print(f"  [Match Levels] {category_evidence_summary_context['metadata'].get('category_evidence_summary_match_levels', '')}")
                print(f"  [Summary] {console_safe_text(sample.get('category_evidence_summary', '')[:1000])}...")
                print(f"  [Evidence Text] {console_safe_text(sample.get('category_evidence_summary_evidence', '')[:1000])}...")
                print("-" * 50 + "\n")
            if idx == 0:
                debug_sample = dict(sample)
                debug_sample["target_str"] = enriched_target_str
                print_first_qa_debug(debug_sample, conf, text_prompt=prompt)
            req_obj = {
                "id": str(idx),
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": conf["temperature"], "maxOutputTokens": generation_max_output_tokens(conf)}
                }
            }
            f.write(json.dumps(req_obj) + "\n")
            
    print(f">>> 2. Uploading file to Gemini API...")
    uploaded_file = client.files.upload(
        file=jsonl_path,
        config={'mime_type': 'application/jsonl'}
    )
    
    print(f">>> 3. Submitting Batch Job...")
    batch_job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": f"{conf['batch_display_name']}_{conf['dataset']}"}
    )
    
    print(f">>> Batch Job Submitted! ID: {batch_job.name}")
    print(">>> 4. Polling until complete...")
    
    while True:
        job = client.batches.get(name=batch_job.name)
        state_str = str(job.state)
        print(f"[{time.strftime('%X')}] Status: {state_str}")
        if "SUCCEEDED" in state_str or "FAILED" in state_str or "CANCELLED" in state_str:
            break
        time.sleep(conf.get("poll_interval", 60))
        
    if "SUCCEEDED" in str(job.state):
        out_file_name = job.dest.file_name
        print(f">>> 5. Complete! Results file format: {out_file_name}")
        print(">>> 6. Downloading and parsing batch results...")
        
        file_bytes = client.files.download(file=out_file_name)
        out_jsonl_path = os.path.join(conf["output_dir"], f"batch_response_{conf['dataset']}.jsonl")
        with open(out_jsonl_path, "wb") as f:
            f.write(file_bytes)
            
        print(">>> 7. Calculating Hit Rate...")
        # Parse the JSONL results 
        result_map = {}
        for line in file_bytes.decode("utf-8").splitlines():
            if not line.strip(): continue
            try:
                resp_obj = json.loads(line)
                req_id = resp_obj.get("id")
                # Try to extract the generated text
                if "response" in resp_obj:
                    raw_text = resp_obj["response"].get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    pred_info = evaluate_model_output(raw_text, samples[int(req_id)]['true_option_char'], conf)
                elif "error" in resp_obj:
                    raw_text = str(resp_obj["error"])
                    pred_info = {"prediction": "ERR_API", "hit": 0}
                else:
                    raw_text = "UNKNOWN_FORMAT"
                    pred_info = {"prediction": "ERR_API", "hit": 0}
                result_map[int(req_id)] = (pred_info, raw_text)
            except Exception as e:
                continue

        # Merge with samples and evaluate
        results = []
        for idx, sample in enumerate(samples):
            pred_info, raw_response = result_map.get(idx, ({"prediction": "ERR_MISSING", "hit": 0}, "Not found in batch response"))
            sample.update(pred_info)
            sample['raw_response'] = raw_response
            results.append(sample)

        # Calculate metrics and prepare DataFrame
        df = pd.DataFrame(results)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        hit_rate = df['hit'].mean()
        valid_options = option_letters(conf["num_cans"])
        if is_ranking_mode(conf):
            valid_mask = df['ranking_valid'].astype(bool) if 'ranking_valid' in df.columns else pd.Series(False, index=df.index)
        else:
            valid_mask = df['prediction'].isin(valid_options)
        valid_ratio = valid_mask.mean()
        
        valid_only_hit_rate = df.loc[valid_mask, 'hit'].mean() if valid_mask.sum() > 0 else 0.0
        
        df['overall_hit_rate'] = hit_rate
        df['overall_valid_ratio'] = valid_ratio
        df['valid_only_hit_rate'] = valid_only_hit_rate
        if is_ranking_mode(conf):
            true_rank_numeric = pd.to_numeric(df.get('true_rank', pd.Series(dtype=float)), errors='coerce')
            df['overall_hit_at_3'] = df['hit_at_3'].mean() if 'hit_at_3' in df.columns and not df.empty else 0.0
            df['overall_hit_at_5'] = df['hit_at_5'].mean() if 'hit_at_5' in df.columns and not df.empty else 0.0
            df['overall_mrr'] = df['mrr'].mean() if 'mrr' in df.columns and not df.empty else 0.0
            df['overall_ndcg_at_3'] = df['ndcg_at_3'].mean() if 'ndcg_at_3' in df.columns and not df.empty else 0.0
            df['overall_ndcg_at_5'] = df['ndcg_at_5'].mean() if 'ndcg_at_5' in df.columns and not df.empty else 0.0
            df['overall_ndcg_at_10'] = df['ndcg_at_10'].mean() if 'ndcg_at_10' in df.columns and not df.empty else 0.0
            df['overall_mean_rank'] = true_rank_numeric.mean() if true_rank_numeric.notna().any() else 0.0
            df['overall_median_rank'] = true_rank_numeric.median() if true_rank_numeric.notna().any() else 0.0
            df['valid_only_mrr'] = df.loc[valid_mask, 'mrr'].mean() if valid_mask.sum() > 0 and 'mrr' in df.columns else 0.0
        
        # Insert experiment configurations
        df['cfg_num_cans'] = conf.get("num_cans", "")
        df['cfg_num_token'] = conf.get("num_token", "")
        df['cfg_toy_eval'] = conf.get("toy_eval", "")
        df['cfg_prediction_mode'] = conf.get("prediction_mode", "choice")
        df['cfg_ranking_max_output_tokens'] = conf.get("ranking_max_output_tokens", "")
        df['cfg_ranking_fill_missing'] = conf.get("ranking_fill_missing", "")
        df['cfg_seed'] = conf.get("seed", "")
        df['cfg_shuffle_seed'] = conf.get("shuffle_seed", "")
        df['cfg_use_fixed_test_split'] = conf.get("use_fixed_test_split", False)
        df['cfg_test_input_file'] = conf.get("test_input_file", "")
        df['cfg_test_gt_file'] = conf.get("test_gt_file", "")
        df['cfg_use_hard_negative'] = conf.get("use_hard_negative", False)
        df['cfg_use_hard_negative_effective'] = conf.get("use_hard_negative", False) and not conf.get("use_fixed_test_split", False)
        df['cfg_use_cooccurrence'] = conf.get("use_cooccurrence", False)
        df['cfg_use_soft_cooccurrence'] = conf.get("use_soft_cooccurrence", False)
        df['cfg_soft_cooccurrence_source'] = conf.get("soft_cooccurrence_source", "")
        df['cfg_use_ui_category_purchase_prior'] = conf.get("use_ui_category_purchase_prior", False)
        df['cfg_ui_category_purchase_prior_top_k'] = conf.get("ui_category_purchase_prior_top_k", "")
        df['cfg_ui_category_purchase_prior_min_support'] = conf.get("ui_category_purchase_prior_min_support", "")
        df['cfg_use_item_bundle_affiliation_desc'] = conf.get("use_item_bundle_affiliation_desc", False)
        df['cfg_item_bundle_affiliation_k'] = conf.get("item_bundle_affiliation_k", "")
        df['cfg_item_bundle_affiliation_alpha'] = conf.get("item_bundle_affiliation_alpha", "")
        df['cfg_item_bundle_affiliation_exclude_query_items'] = conf.get("item_bundle_affiliation_exclude_query_items", False)
        df['cfg_item_bundle_affiliation_use_soft'] = conf.get("item_bundle_affiliation_use_soft", False)
        df['cfg_item_bundle_affiliation_soft_source'] = conf.get("item_bundle_affiliation_soft_source", "")
        df['cfg_item_bundle_affiliation_soft_alpha'] = conf.get("item_bundle_affiliation_soft_alpha", "")
        df['cfg_use_item_user_copurchase_desc'] = conf.get("use_item_user_copurchase_desc", False)
        df['cfg_item_user_copurchase_k'] = conf.get("item_user_copurchase_k", "")
        df['cfg_item_user_copurchase_alpha'] = conf.get("item_user_copurchase_alpha", "")
        df['cfg_item_user_copurchase_exclude_query_items'] = conf.get("item_user_copurchase_exclude_query_items", False)
        df['cfg_use_bundle_graph_context'] = conf.get("use_bundle_graph_context", False)
        df['cfg_bundle_graph_context_k'] = conf.get("bundle_graph_context_k", "")
        df['cfg_bundle_graph_context_max_items'] = conf.get("bundle_graph_context_max_items", "")
        df['cfg_bundle_graph_context_use_soft'] = conf.get("bundle_graph_context_use_soft", False)
        df['cfg_bundle_graph_context_soft_source'] = conf.get("bundle_graph_context_soft_source", "")
        df['cfg_bundle_graph_context_soft_alpha'] = conf.get("bundle_graph_context_soft_alpha", "")
        df['cfg_use_category_evidence_summary'] = conf.get("use_category_evidence_summary", False)
        df['cfg_category_evidence_summary_k'] = conf.get("category_evidence_summary_k", "")
        df['cfg_category_evidence_summary_include_evidence'] = conf.get("category_evidence_summary_include_evidence", False)
        df['cfg_category_evidence_summary_model'] = conf.get("category_evidence_summary_model", "")
        df['cfg_category_evidence_summary_api_key_env'] = conf.get("category_evidence_summary_api_key_env", "")
        df['cfg_category_evidence_summary_max_output_tokens'] = conf.get("category_evidence_summary_max_output_tokens", "")
        df['cfg_use_cc_retrieval_context'] = conf.get("use_cc_retrieval_context", False)
        df['cfg_cc_retrieval_context_k'] = conf.get("cc_retrieval_context_k", "")
        df['cfg_cc_retrieval_context_seed'] = conf.get("cc_retrieval_context_seed", "")
        df['cfg_cc_retrieval_overlap_weight'] = conf.get("cc_retrieval_overlap_weight", "")
        df['cfg_cc_retrieval_extra_weight'] = conf.get("cc_retrieval_extra_weight", "")
        df['cfg_use_category_completion_prior_desc'] = conf.get("use_category_completion_prior_desc", False)
        df['cfg_use_category_item_text_aug'] = conf.get("use_category_item_text_aug", False)
        df['cfg_category_item_aug_apply_to'] = conf.get("category_item_aug_apply_to", "")
        df['cfg_category_item_aug_rep_items_per_category'] = conf.get("category_item_aug_rep_items_per_category", "")
        df['cfg_use_category_name_aug'] = conf.get("use_category_name_aug", False)
        df['cfg_category_name_aug_apply_to'] = conf.get("category_name_aug_apply_to", "")
        df['cfg_category_name_field'] = conf.get("category_name_field", "")
        df['cfg_category_name_root'] = conf.get("category_name_root", "")
        df['cfg_input_category_co_occur'] = conf.get("input_category_co_occur", False)
        df['cfg_input_category_co_occur_apply_to'] = conf.get("input_category_co_occur_apply_to", "")
        df['cfg_input_category_co_occur_verbalization'] = conf.get("input_category_co_occur_verbalization", "")
        df['cfg_input_category_co_occur_top_k'] = conf.get("input_category_co_occur_top_k", "")
        df['cfg_input_category_co_occur_rep_items_per_category'] = conf.get("input_category_co_occur_rep_items_per_category", "")
        df['cfg_category_prior_top_k'] = conf.get("category_prior_top_k", "")
        df['cfg_category_prior_verbalization'] = conf.get("category_prior_verbalization", "")
        df['cfg_category_prior_rep_items_per_category'] = conf.get("category_prior_rep_items_per_category", "")
        df['cfg_category_prior_min_support'] = conf.get("category_prior_min_support", "")
        df['cfg_category_prior_embedding_model'] = conf.get("category_prior_embedding_model", "")

        # Save results in dataset-specific subfolder
        actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
        os.makedirs(actual_output_dir, exist_ok=True)
        
        ranking_str = "RANK_" if is_ranking_mode(conf) else ""
        cooc_str = "COOC_" if conf.get("use_cooccurrence", False) else ""
        soft_source = conf.get("soft_cooccurrence_source", "")
        soft_cooc_str = f"SOFTCOOC_{soft_source}_" if conf.get("use_soft_cooccurrence", False) else ""
        ui_cat_purchase_str = "UICATPUR_" if conf.get("use_ui_category_purchase_prior", False) else ""
        hn_str = "HN_" if conf.get("use_hard_negative", False) else ""
        if conf.get("use_item_bundle_affiliation_desc", False) and conf.get("item_bundle_affiliation_use_soft", False):
            item_aff_str = f"ITEMAFF_SOFT_{conf.get('item_bundle_affiliation_soft_source', '')}_"
        else:
            item_aff_str = "ITEMAFF_" if conf.get("use_item_bundle_affiliation_desc", False) else ""
        user_pur_str = "USERPUR_" if conf.get("use_item_user_copurchase_desc", False) else ""
        if conf.get("use_bundle_graph_context", False) and conf.get("bundle_graph_context_use_soft", False):
            bundle_ctx_str = f"BGRAPH_SOFT_{conf.get('bundle_graph_context_soft_source', '')}_"
        else:
            bundle_ctx_str = "BGRAPH_" if conf.get("use_bundle_graph_context", False) else ""
        category_evidence_summary_str = "CATSUM_" if conf.get("use_category_evidence_summary", False) else ""
        cc_retrieval_str = "CCRET_" if conf.get("use_cc_retrieval_context", False) else ""
        category_prior_str = "CATPRIOR_" if conf.get("use_category_completion_prior_desc", False) else ""
        if conf.get("use_category_completion_prior_desc", False) and str(conf.get("category_prior_verbalization", "")).strip().lower() in {"category_names", "category_name", "names", "name"}:
            category_prior_str = "CATPRIORNAME_"
        category_item_aug_str = "CATITEMAUG_" if conf.get("use_category_item_text_aug", False) else ""
        category_name_aug_str = "CATNAMEAUG_" if conf.get("use_category_name_aug", False) else ""
        input_category_co_occur_str = ""
        if conf.get("input_category_co_occur", False):
            if str(conf.get("input_category_co_occur_verbalization", "")).strip().lower() in {"category_names", "category_name", "names", "name"}:
                input_category_co_occur_str = "INPCATNAMECOOC_"
            else:
                input_category_co_occur_str = "INPCATCOOC_"
        save_path = os.path.join(actual_output_dir, f"results_{conf['dataset']}_batch_{ranking_str}{item_aff_str}{user_pur_str}{bundle_ctx_str}{ui_cat_purchase_str}{category_evidence_summary_str}{cc_retrieval_str}{category_prior_str}{category_item_aug_str}{category_name_aug_str}{input_category_co_occur_str}{cooc_str}{soft_cooc_str}{hn_str}C{conf.get('num_cans', '')}_T{conf.get('num_token', '')}_{timestamp}.csv")
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("-" * 30)
        print(f"Batch Dataset: {conf['dataset']}")
        print(f"Saved to: {save_path}")
        print(f"Overall Hit Rate: {hit_rate:.4f}")
        if is_ranking_mode(conf):
            print(f"Hit@3: {df['hit_at_3'].mean():.4f} | Hit@5: {df['hit_at_5'].mean():.4f} | MRR: {df['mrr'].mean():.4f} | NDCG@10: {df['ndcg_at_10'].mean():.4f}")
            print(f"Mean Rank: {pd.to_numeric(df['true_rank'], errors='coerce').mean():.4f}")
        print(f"Valid-Only Hit Rate: {valid_only_hit_rate:.4f} (from {valid_mask.sum()} samples without errors)")
        print(f"Valid Ratio: {valid_ratio:.4f}")
        print("-" * 30)
        
        # save_translated_csv(df, conf, timestamp, mode_suffix="_batch", actual_output_dir=actual_output_dir)
    else:
        print(">>> Batch job did not succeed.")

def save_translated_csv(df, conf, base_timestamp, mode_suffix="", actual_output_dir=None):
    if "spotify" in conf["dataset"].lower():
        return
        
    print(">>> Translating input/target columns to Korean (via Google Translate Batch Mode - 10 per batch)...")
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='ko')
        
        df_kor = df.copy()
        
        # Translate safely in small batches.
        def batch_translate(series):
            texts = series.tolist()
            translated = []
            batch_size = 10 
            try:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    try:
                        translated.extend(translator.translate_batch(batch))
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"[Warning] Batch {i//batch_size + 1} translation failed: {e}. Keeping original text.")
                        translated.extend(batch)
            except KeyboardInterrupt:
                print("\n>>> [Stopped] Translation was interrupted by the user.")
                exit(1)
            return translated

        df_kor['input_str'] = batch_translate(df_kor['input_str'])
        df_kor['target_str'] = batch_translate(df_kor['target_str'])
        
        if not actual_output_dir:
            actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
            
        save_path_kor = os.path.join(actual_output_dir, f"results_{conf['dataset']}{mode_suffix}_kor_{base_timestamp}.csv")
        df_kor.to_csv(save_path_kor, index=False)
        print(f">>> Saved translated file: {save_path_kor}")
    except ImportError:
        print("[Warning] deep-translator is required.")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM Zero-Shot Bundle Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to a YAML config file")
    parser.add_argument("--start_idx", type=int, default=0, help="Start from a specific sample index (0-based)")
    parser.add_argument("--resume", type=str, default="", help="Path to a _partial.csv file to resume from")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    print(f">>> Loaded config: {args.config}")

    os.makedirs(conf["output_dir"], exist_ok=True)
    set_seed(conf["seed"])
    
    dataset = BundleZeroShotDataset(conf)
    
    # Offline hard-negative files are tied to the split they were generated from.
    # When using a fixed bundle-level split, regenerate samples from the fixed files.
    hard_negative_path = os.path.join(conf.get("data_path", "./datasets"), conf["dataset"], f"hard_negative_samples_{conf['dataset']}.json")
    use_hard_negative = conf.get("use_hard_negative", False) and not conf.get("use_fixed_test_split", False)
    if conf.get("use_hard_negative", False) and conf.get("use_fixed_test_split", False):
        print(">>> Skipping pre-generated hard negatives because use_fixed_test_split=True")
    if use_hard_negative and os.path.exists(hard_negative_path):
        print(f">>> Loading PRE-GENERATED HARD NEGATIVE samples from {hard_negative_path}")
        with open(hard_negative_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        samples = dataset.get_eval_samples()
        
    initial_results = None
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if args.resume and os.path.exists(args.resume):
        print(f">>> Resuming from {args.resume}")
        try:
            df_prev = pd.read_csv(args.resume, encoding='utf-8-sig')
        except UnicodeDecodeError:
            # Fallback for files saved with system default encoding (e.g. cp949 on Windows)
            df_prev = pd.read_csv(args.resume, encoding='cp949')
        initial_results = df_prev.to_dict('records')
        args.start_idx = len(initial_results)
        
        match = re.search(r'_(\d{8}_\d{6})(_partial)?\.csv$', args.resume)
        if match:
            timestamp = match.group(1)
            print(f">>> Reusing timestamp: {timestamp}")

    if args.start_idx > 0:
        print(f">>> Slicing samples: Starting from index {args.start_idx} (Total before: {len(samples)})")
        samples = samples[args.start_idx:]
        
    print(f"Total test samples prepared: {len(samples)} (Start Idx: {args.start_idx})")

    icl_retriever = None
    if conf.get("use_icl_retrieval", False):
        if conf.get("mode") == "batch":
            raise NotImplementedError("ICL retrieval is currently wired for sync mode only.")
        from retrieve_icl_examples import InputEmbeddingICLRetriever
        print(">>> Building ICL retriever from training bundles...")
        icl_retriever = InputEmbeddingICLRetriever(conf, dataset)

    user_context_retriever = None
    if conf.get("use_user_context", False):
        if conf.get("mode") == "batch":
            raise NotImplementedError("User context is currently wired for sync mode only.")
        from retrieve_user_context import UserContextRetriever
        print(">>> Building user context retriever from user-item interactions...")
        user_context_retriever = UserContextRetriever(conf, dataset)
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[Error] GEMINI_API_KEY or GOOGLE_API_KEY is not set in .env.")
        exit(1)
        
    client = genai.Client(api_key=api_key)
    summary_client = None
    summary_api_key_env = str(conf.get("category_evidence_summary_api_key_env", "")).strip()
    if conf.get("use_category_evidence_summary", False) and summary_api_key_env:
        summary_api_key = os.getenv(summary_api_key_env, "").strip()
        if not summary_api_key:
            print(f"[Error] {summary_api_key_env} is not set for category evidence summary.")
            exit(1)
        summary_client = genai.Client(api_key=summary_api_key)
        print(f">>> Category evidence summary will use API key env: {summary_api_key_env}")
    
    if conf["mode"] == "sync":
        print(">>> Running in Sync mode...")
        import asyncio
        results = asyncio.run(process_sync_samples(client, conf["model"], samples, conf, timestamp, initial_results=initial_results, start_idx=args.start_idx, dataset=dataset, icl_retriever=icl_retriever, user_context_retriever=user_context_retriever, summary_client=summary_client))
        
        # Final save
        save_path, df, hit_rate, valid_ratio, valid_only_hit_rate, valid_sum = save_intermediate_results(results, conf, timestamp, is_final=True)
        
        # Remove partial file
        partial_path = save_path.replace(".csv", "_partial.csv")
        if os.path.exists(partial_path):
            os.remove(partial_path)
            
        print("-" * 30)
        print(f"Dataset: {conf['dataset']}")
        print(f"Saved to: {save_path}")
        print(f"Overall Hit Rate: {hit_rate:.4f}")
        if is_ranking_mode(conf):
            print(f"Hit@3: {df['hit_at_3'].mean():.4f} | Hit@5: {df['hit_at_5'].mean():.4f} | MRR: {df['mrr'].mean():.4f} | NDCG@10: {df['ndcg_at_10'].mean():.4f}")
            print(f"Mean Rank: {pd.to_numeric(df['true_rank'], errors='coerce').mean():.4f}")
        print(f"Valid-Only Hit Rate: {valid_only_hit_rate:.4f} (from {valid_sum} samples without errors)")
        print(f"Valid Ratio: {valid_ratio:.4f}")
        print("-" * 30)
        
        actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
        # save_translated_csv(df, conf, timestamp, mode_suffix="", actual_output_dir=actual_output_dir)

    elif conf["mode"] == "batch":
        print(">>> Running in Batch API mode...")
        process_batch_samples(client, conf["model"], samples, conf, dataset=dataset, summary_client=summary_client)
