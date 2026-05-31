import ast
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DATASET = "pog_dense"
BUNDLE_ID = 11208


EN_TITLES = {
    18531: "Floral chiffon dress",
    24847: "Long pearl earrings",
    16366: "Oversized denim jacket",
    879: "Short faux-fur biker jacket",
    15382: "Light denim jacket",
    12800: "GT: flat fur slippers",
    17810: "Fleece slip-on shoes",
    8933: "High-waist knit skirt",
    9552: "Men's white sneakers",
    21513: "Bohemian summer dress",
    5579: "Short pink bodycon dress",
    29465: "Large leather tote bag",
}


def load_case(repo_root):
    path = repo_root / "analysis" / f"{DATASET}_ranking_view_analysis" / "per_sample_oracle.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    row = df[df["bundle_id"] == BUNDLE_ID].iloc[0]
    input_ids = [int(x) for x in ast.literal_eval(str(row["input_indices_BIxIB"]))]
    candidate_ids = [int(x) for x in ast.literal_eval(str(row["candidate_indices"]))]
    return row, input_ids, candidate_ids


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


def make_sheet(cards, out_path, cols=4, thumb=180, pad=18):
    font = ImageFont.load_default()
    label_h = 58
    rows = (len(cards) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, card in enumerate(cards):
        row = idx // cols
        col = idx % cols
        x = pad + col * (thumb + pad)
        y = pad + row * (thumb + label_h + pad)
        img = Image.open(card["path"]).convert("RGB")
        img.thumbnail((thumb, thumb))
        bx = x + (thumb - img.width) // 2
        by = y + (thumb - img.height) // 2
        sheet.paste(img, (bx, by))
        draw.rectangle((x, y, x + thumb, y + thumb), outline=(210, 210, 210), width=1)
        if card.get("is_gt"):
            draw.rectangle((x - 2, y - 2, x + thumb + 2, y + thumb + 2), outline=(214, 55, 55), width=4)
        label = f"{card['label']}. {card['title']}" if card.get("label") else card["title"]
        lines = fit_text(draw, label, font, thumb)
        for line_idx, line in enumerate(lines):
            draw.text((x, y + thumb + 6 + line_idx * 14), line, fill=(20, 20, 20), font=font)

    sheet.save(out_path)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "analysis" / "case_study_slide_assets" / f"{DATASET}_bundle_{BUNDLE_ID}_BIxIB"
    input_dir = out_dir / "input"
    candidate_dir = out_dir / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    candidate_dir.mkdir(exist_ok=True)

    row, input_ids, candidate_ids = load_case(repo_root)
    true_id = int(row["true_indice"])
    true_option = str(row["true_option_char"])
    image_dir = repo_root / "datasets" / DATASET / "images"
    item_info = json.loads((repo_root / "datasets" / DATASET / "item_info.json").read_text(encoding="utf-8"))

    manifest = []
    input_cards = []
    candidate_cards = []

    for idx, item_id in enumerate(input_ids, start=1):
        src = image_dir / f"{item_id}.png"
        dst = input_dir / f"input_{idx}_{item_id}.png"
        shutil.copy2(src, dst)
        title = EN_TITLES.get(item_id, item_info[str(item_id)]["title"])
        manifest.append({"role": "input", "label": str(idx), "item_id": item_id, "english_title": title, "image": str(dst)})
        input_cards.append({"label": str(idx), "title": title, "path": dst})

    for idx, item_id in enumerate(candidate_ids):
        label = chr(ord("A") + idx)
        src = image_dir / f"{item_id}.png"
        prefix = "GT_" if item_id == true_id else ""
        dst = candidate_dir / f"{prefix}{label}_{item_id}.png"
        shutil.copy2(src, dst)
        title = EN_TITLES.get(item_id, item_info[str(item_id)]["title"])
        manifest.append({
            "role": "candidate",
            "label": label,
            "item_id": item_id,
            "english_title": title,
            "is_gt": item_id == true_id,
            "image": str(dst),
        })
        candidate_cards.append({"label": label, "title": title, "path": dst, "is_gt": item_id == true_id})

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    make_sheet(input_cards, out_dir / "input_sheet.png", cols=2)
    make_sheet(candidate_cards, out_dir / "candidate_sheet.png", cols=5)
    make_sheet(input_cards + candidate_cards, out_dir / "case_sheet.png", cols=4)

    print(f"Saved case-study image assets to {out_dir}")
    print(f"Ground truth: option {true_option}, item {true_id}")


if __name__ == "__main__":
    main()
