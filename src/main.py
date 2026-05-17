import os
import re
import sys
import yaml
import json
import time
import asyncio
import glob
import pandas as pd
from dotenv import load_dotenv
from google import genai
from dataset import BundleZeroShotDataset, set_seed
from PIL import Image

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
                    icl_example=None, user_context_block="", bundle_graph_context_block=""):
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

    prompt = (
        f"You are a helpful and honest assistant. The following are multiple choice questions about {t_name}. "
        f"You should directly answer the question by choosing the letter of the correct option. "
        f"Only provide the letter of your answer, without any explanation or mentioning the option content.\n"
        f"{cf_legend}"
        f"{icl_block}"
        f"{user_context_block}"
        f"{bundle_graph_context_block}"
        f"Question: Given the partial {b_name}: {input_str}, which candidate {i_name} should be included into this {b_name}?\n"
        f"Options: {target_str}\n"
        #f"First, analyze the overall combination and coherence of the items in the {b_name}. Then, choose the candidate {i_name} that best completes the set."
        #f"{extra_instruction}"
        f"Your answer should indicate your choice with a single letter (e.g., \u201cA,\u201d \u201cB,\u201d \u201cC,\u201d etc.).\nChoice: "
    )
    return prompt

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

def save_intermediate_results(results, conf, timestamp, is_final=False):
    df = pd.DataFrame(results)
    hit_rate = df['hit'].mean() if not df.empty else 0.0
    valid_options = [chr(ord('A')+i) for i in range(conf.get("num_cans", 10))]
    valid_mask = df['prediction'].isin(valid_options)
    valid_ratio = valid_mask.mean() if not df.empty else 0.0
    valid_only_hit_rate = df.loc[valid_mask, 'hit'].mean() if valid_mask.sum() > 0 else 0.0
    
    df['overall_hit_rate'] = hit_rate
    df['overall_valid_ratio'] = valid_ratio
    df['valid_only_hit_rate'] = valid_only_hit_rate
    df['cfg_num_cans'] = conf.get("num_cans", "")
    df['cfg_num_token'] = conf.get("num_token", "")
    df['cfg_toy_eval'] = conf.get("toy_eval", "")
    df['cfg_seed'] = conf.get("seed", "")
    df['cfg_shuffle_seed'] = conf.get("shuffle_seed", "")
    df['cfg_use_hard_negative'] = conf.get("use_hard_negative", False)
    df['cfg_use_cooccurrence'] = conf.get("use_cooccurrence", False)
    df['cfg_use_soft_cooccurrence'] = conf.get("use_soft_cooccurrence", False)
    df['cfg_soft_cooccurrence_source'] = conf.get("soft_cooccurrence_source", "")
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
    
    actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
    os.makedirs(actual_output_dir, exist_ok=True)
    cooc_str = "COOC_" if conf.get("use_cooccurrence", False) else ""
    soft_source = conf.get("soft_cooccurrence_source", "")
    soft_cooc_str = f"SOFTCOOC_{soft_source}_" if conf.get("use_soft_cooccurrence", False) else ""
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
    partial_str = "" if is_final else "_partial"
    save_path = os.path.join(actual_output_dir, f"results_{conf['dataset']}_{icl_str}{user_str}{item_aff_str}{user_pur_str}{bundle_ctx_str}{cooc_str}{soft_cooc_str}{hn_str}C{conf.get('num_cans', '')}_T{conf.get('num_token', '')}_{timestamp}{partial_str}.csv")
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

async def process_sync_samples(client, model, samples, conf, timestamp, initial_results=None, start_idx=0, dataset=None, icl_retriever=None, user_context_retriever=None):
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
        
        text_prompt = generate_prompt(
            conf["dataset"], sample["input_str"], enriched_target_str,
            use_multimodal=conf.get("use_multimodal", False),
            use_cooccurrence=conf.get("use_cooccurrence", False),
            use_soft_cooccurrence=conf.get("use_soft_cooccurrence", False),
            soft_cooccurrence_source=conf.get("soft_cooccurrence_source", ""),
            icl_example=icl_example,
            user_context_block=user_context_block,
            bundle_graph_context_block=bundle_graph_context_block
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
            contents = []
            img_dir = os.path.join(conf.get("data_path", "./datasets"), conf["dataset"], "images")
            
            contents.append("Images for the items currently in the bundle:")
            for i, item_id in enumerate(sample.get("input_indices", [])):
                requested_image_ids.append(int(item_id))
                img_path = find_item_image(img_dir, item_id)
                if img_path:
                    try:
                        contents.append(f"[Input Item {i+1}]")
                        contents.append(Image.open(img_path))
                        loaded_image_count += 1
                        found_image_paths.append(img_path)
                    except Exception as e:
                        failed_image_paths.append(f"{img_path} ({e})")
                else:
                    missing_image_ids.append(int(item_id))
                    
            contents.append("Images for the candidate items:")
            for i, item_id in enumerate(sample.get("candidate_indices", [])):
                opt_char = chr(ord('A') + i)
                requested_image_ids.append(int(item_id))
                img_path = find_item_image(img_dir, item_id)
                if img_path:
                    try:
                        contents.append(f"[Candidate {opt_char}]")
                        contents.append(Image.open(img_path))
                        loaded_image_count += 1
                        found_image_paths.append(img_path)
                    except Exception as e:
                        failed_image_paths.append(f"{img_path} ({e})")
                else:
                    missing_image_ids.append(int(item_id))
                    
            contents.append(text_prompt)

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
                    config={"temperature": conf["temperature"], "max_output_tokens": 10}
                )
                raw_text = res.text if res.text else ""
                pred_text = parse_model_response(raw_text)
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
                pred_text = "ERR_EX"
                break
        
        sample['prediction'] = pred_text
        sample['raw_response'] = raw_text # Save verbatim output or error trace
        sample['hit'] = int(pred_text == sample['true_option_char'])
        results.append(sample)
        
        # Save after each sample so interrupted runs can resume.
        save_intermediate_results(results, conf, timestamp, is_final=False)
        
        print(f"[{current_idx+1}/{total_samples_len}] True: {sample['true_option_char']} | Pred: {pred_text}")
        
        # Enforce rate limit (Dynamic based on model Free Tier limits)
        # Gemini 2.5 Flash / Pro -> 5 requests / min = 12s interval. (Using 13s)
        # Gemini 1.5 Flash / 3.x Lite -> 15 requests / min = 4s interval. (Using 4.5s)
        sleep_time = 15.0
        if "gemma" in model or "lite" in model.lower():
            sleep_time = 5
            
        if idx < len(samples) - 1:
            await asyncio.sleep(sleep_time)
            
    return results

def process_batch_samples(client, model, samples, conf, dataset=None):
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
            prompt = generate_prompt(
                conf["dataset"],
                sample["input_str"],
                enriched_target_str,
                conf.get("use_multimodal", False),
                use_cooccurrence=conf.get("use_cooccurrence", False),
                use_soft_cooccurrence=conf.get("use_soft_cooccurrence", False),
                soft_cooccurrence_source=conf.get("soft_cooccurrence_source", ""),
                bundle_graph_context_block=bundle_graph_context_block
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
            if idx == 0:
                debug_sample = dict(sample)
                debug_sample["target_str"] = enriched_target_str
                print_first_qa_debug(debug_sample, conf, text_prompt=prompt)
            req_obj = {
                "id": str(idx),
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": conf["temperature"], "maxOutputTokens": 10}
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
                    pred_text = parse_model_response(raw_text)
                elif "error" in resp_obj:
                    raw_text = str(resp_obj["error"])
                    pred_text = "ERR_API"
                else:
                    raw_text = "UNKNOWN_FORMAT"
                    pred_text = "ERR_API"
                result_map[int(req_id)] = (pred_text, raw_text)
            except Exception as e:
                continue

        # Merge with samples and evaluate
        results = []
        for idx, sample in enumerate(samples):
            pred_info = result_map.get(idx, ("ERR_MISSING", "Not found in batch response"))
            sample['prediction'] = pred_info[0]
            sample['raw_response'] = pred_info[1]
            sample['hit'] = int(pred_info[0] == sample['true_option_char'])
            results.append(sample)

        # Calculate metrics and prepare DataFrame
        df = pd.DataFrame(results)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        hit_rate = df['hit'].mean()
        valid_options = [chr(ord('A')+i) for i in range(conf["num_cans"])]
        valid_mask = df['prediction'].isin(valid_options)
        valid_ratio = valid_mask.mean()
        
        valid_only_hit_rate = df.loc[valid_mask, 'hit'].mean() if valid_mask.sum() > 0 else 0.0
        
        df['overall_hit_rate'] = hit_rate
        df['overall_valid_ratio'] = valid_ratio
        df['valid_only_hit_rate'] = valid_only_hit_rate
        
        # Insert experiment configurations
        df['cfg_num_cans'] = conf.get("num_cans", "")
        df['cfg_num_token'] = conf.get("num_token", "")
        df['cfg_toy_eval'] = conf.get("toy_eval", "")
        df['cfg_seed'] = conf.get("seed", "")
        df['cfg_shuffle_seed'] = conf.get("shuffle_seed", "")
        df['cfg_use_hard_negative'] = conf.get("use_hard_negative", False)
        df['cfg_use_cooccurrence'] = conf.get("use_cooccurrence", False)
        df['cfg_use_soft_cooccurrence'] = conf.get("use_soft_cooccurrence", False)
        df['cfg_soft_cooccurrence_source'] = conf.get("soft_cooccurrence_source", "")
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

        # Save results in dataset-specific subfolder
        actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
        os.makedirs(actual_output_dir, exist_ok=True)
        
        cooc_str = "COOC_" if conf.get("use_cooccurrence", False) else ""
        soft_source = conf.get("soft_cooccurrence_source", "")
        soft_cooc_str = f"SOFTCOOC_{soft_source}_" if conf.get("use_soft_cooccurrence", False) else ""
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
        save_path = os.path.join(actual_output_dir, f"results_{conf['dataset']}_batch_{item_aff_str}{user_pur_str}{bundle_ctx_str}{cooc_str}{soft_cooc_str}{hn_str}C{conf.get('num_cans', '')}_T{conf.get('num_token', '')}_{timestamp}.csv")
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("-" * 30)
        print(f"Batch Dataset: {conf['dataset']}")
        print(f"Saved to: {save_path}")
        print(f"Overall Hit Rate: {hit_rate:.4f}")
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
    
    # NEW LOGIC: optionally load offline-generated hard negatives
    hard_negative_path = os.path.join(conf.get("data_path", "./datasets"), conf["dataset"], f"hard_negative_samples_{conf['dataset']}.json")
    if conf.get("use_hard_negative", False) and os.path.exists(hard_negative_path):
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
    
    if conf["mode"] == "sync":
        print(">>> Running in Sync mode...")
        import asyncio
        results = asyncio.run(process_sync_samples(client, conf["model"], samples, conf, timestamp, initial_results=initial_results, start_idx=args.start_idx, dataset=dataset, icl_retriever=icl_retriever, user_context_retriever=user_context_retriever))
        
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
        print(f"Valid-Only Hit Rate: {valid_only_hit_rate:.4f} (from {valid_sum} samples without errors)")
        print(f"Valid Ratio: {valid_ratio:.4f}")
        print("-" * 30)
        
        actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
        # save_translated_csv(df, conf, timestamp, mode_suffix="", actual_output_dir=actual_output_dir)

    elif conf["mode"] == "batch":
        print(">>> Running in Batch API mode...")
        process_batch_samples(client, conf["model"], samples, conf, dataset=dataset)
