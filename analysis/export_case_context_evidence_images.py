import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DATASET = "pog_dense"
BUNDLE_ID = 11208
CANDIDATE_C_ID = 15382


EN_TITLES = {
    15382: "Candidate C: light denim jacket",
    18531: "Input: floral chiffon dress",
    24847: "Input: long pearl earrings",
    12800: "GT: flat fur slippers",
    29602: "Retrieved: flat pointed shoes",
    14273: "Retrieved: fur flat slippers",
}


def load_train_bundles(path):
    bundles = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split(", ") if x.strip()]
            if parts:
                bundles[parts[0]] = parts[1:]
    return bundles


def sample_from_counter(counter, item_id, k=3, alpha=0.5, seed=45):
    import numpy as np

    if not counter:
        return []
    indices = np.array(sorted(counter), dtype=np.int64)
    counts = np.array([counter[int(item)] for item in indices], dtype=np.float64)
    weights = np.power(counts, float(alpha))
    rng = np.random.default_rng(int(item_id) + int(seed))
    sample_size = min(int(k), len(indices))
    sampled = rng.choice(indices, size=sample_size, replace=False, p=weights / weights.sum())
    return [int(x) for x in sampled.tolist()]


def bundle_neighbors(train_bundles, item_id):
    counter = Counter()
    for items in train_bundles.values():
        item_set = set(int(x) for x in items)
        if int(item_id) not in item_set:
            continue
        for other in item_set:
            if other != int(item_id):
                counter[other] += 1
    return counter


def user_neighbors(ui_path, item_id):
    counter = Counter()
    with open(ui_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split(", ") if x.strip()]
            items = parts[1:]
            item_set = set(items)
            if int(item_id) not in item_set:
                continue
            for other in item_set:
                if other != int(item_id):
                    counter[other] += 1
    return counter


def fit_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines[:3]


def make_sheet(cards, out_path, cols=3, thumb=190, pad=18):
    font = ImageFont.load_default()
    label_h = 62
    rows = (len(cards) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, card in enumerate(cards):
        row = idx // cols
        col = idx % cols
        x = pad + col * (thumb + pad)
        y = pad + row * (thumb + label_h + pad)
        image = Image.open(card["path"]).convert("RGB")
        image.thumbnail((thumb, thumb))
        canvas.paste(image, (x + (thumb - image.width) // 2, y + (thumb - image.height) // 2))
        color = card.get("outline", (210, 210, 210))
        width = card.get("outline_width", 1)
        draw.rectangle((x, y, x + thumb, y + thumb), outline=color, width=width)
        for line_idx, line in enumerate(fit_text(draw, card["title"], font, thumb)):
            draw.text((x, y + thumb + 6 + line_idx * 14), line, fill=(20, 20, 20), font=font)
    canvas.save(out_path)


def copy_cards(repo_root, out_dir, view_name, item_ids, item_info, titles):
    image_dir = repo_root / "datasets" / DATASET / "images"
    view_dir = out_dir / view_name
    view_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    rows = []
    for idx, item_id in enumerate(item_ids, start=1):
        src = image_dir / f"{item_id}.png"
        dst = view_dir / f"{idx}_{item_id}.png"
        title = titles.get(item_id, EN_TITLES.get(item_id, item_info[str(item_id)]["title"]))
        image_available = src.exists()
        if image_available:
            shutil.copy2(src, dst)
        else:
            placeholder = Image.new("RGB", (360, 360), (244, 244, 244))
            draw = ImageDraw.Draw(placeholder)
            font = ImageFont.load_default()
            draw.rectangle((8, 8, 352, 352), outline=(190, 190, 190), width=2)
            draw.text((24, 130), "Image not found", fill=(80, 80, 80), font=font)
            draw.text((24, 155), f"item_id: {item_id}", fill=(80, 80, 80), font=font)
            for line_idx, line in enumerate(fit_text(draw, title, font, 300)):
                draw.text((24, 190 + line_idx * 16), line, fill=(40, 40, 40), font=font)
            placeholder.save(dst)
        cards.append({"title": title, "path": dst})
        rows.append({
            "view": view_name,
            "order": idx,
            "item_id": item_id,
            "english_title": title,
            "original_title": item_info[str(item_id)]["title"],
            "image_available": image_available,
            "image": str(dst),
        })
    make_sheet(cards, view_dir / f"{view_name}_context_sheet.png", cols=min(3, len(cards)))
    return rows


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "analysis" / "case_study_slide_assets" / f"{DATASET}_bundle_{BUNDLE_ID}_context_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    item_info = json.loads((repo_root / "datasets" / DATASET / "item_info.json").read_text(encoding="utf-8"))
    train_bundles = load_train_bundles(repo_root / "datasets" / DATASET / "bi_train.txt")

    ib_context = sample_from_counter(bundle_neighbors(train_bundles, CANDIDATE_C_ID), CANDIDATE_C_ID)
    iu_context = sample_from_counter(user_neighbors(repo_root / "datasets" / DATASET / "ui_full.txt", CANDIDATE_C_ID), CANDIDATE_C_ID)

    # The BIxIB evidence comes from the retrieved related outfits for bundle 11208.
    # These are the non-input shoe/slipper items that reveal the missing slot.
    bixib_context = [29602, 14273]

    titles = {
        CANDIDATE_C_ID: "Wrong top prediction: light denim jacket",
        ib_context[0]: "IBxBI context: white sneakers",
        ib_context[1]: "IBxBI context: long statement earrings",
        iu_context[0]: "IUxUI context: chain crossbody bag",
        iu_context[1]: "IUxUI context: chiffon dress",
        iu_context[2]: "IUxUI context: French vintage dress",
        29602: "BIxIB context: flat pointed shoes",
        14273: "BIxIB context: fur flat slippers",
    }

    manifest_rows = []
    manifest_rows.extend(copy_cards(repo_root, out_dir, "IBxBI_candidate_C_context", [CANDIDATE_C_ID] + ib_context, item_info, titles))
    manifest_rows.extend(copy_cards(repo_root, out_dir, "IUxUI_candidate_C_context", [CANDIDATE_C_ID] + iu_context, item_info, titles))
    manifest_rows.extend(copy_cards(repo_root, out_dir, "BIxIB_retrieved_outfit_context", [12800] + bixib_context, item_info, titles))

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Saved context evidence images to {out_dir}")
    print("IBxBI context item ids:", [CANDIDATE_C_ID] + ib_context)
    print("IUxUI context item ids:", [CANDIDATE_C_ID] + iu_context)
    print("BIxIB context item ids:", [12800] + bixib_context)


if __name__ == "__main__":
    main()
