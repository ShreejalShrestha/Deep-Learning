"""
Diagnose one live prediction against the real training dataset.

    python diagnose_prediction.py debug_captures/capture_20261006_120000_123.jpg

Everything here is read-only: the checkpoint, the notebooks and the dataset are
never written to, and nothing is trained. The model, the architecture and the
preprocessing are imported from main.py rather than re-implemented, so what the
diagnosis sees is exactly what the live API sees.

What it reports
---------------
1. the model's own top-5 for the capture (confirms the webcam result offline);
2. nearest training images in the CNN's feature space (penultimate 256-d layer,
   cosine similarity) and in raw pixel space (normalised correlation);
3. a contact sheet under debug_results/ showing the capture beside its nearest
   training neighbours, with their true classes and scores;
4. simple, measurable differences between the capture and the training images
   of the predicted class: hand size, position, background, brightness,
   contrast, and whether the hand touches the crop edges.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The architecture, the weights, the preprocessing and the label maps all come
# from the live API module, so they cannot drift apart from it.
import main as api  # noqa: E402

PROJECT_ROOT = HERE.parent
RESULTS_DIR = HERE / "debug_results"
SAMPLES_PER_CLASS = 24        # training images per class used for the neighbour search
TOP_K_NEIGHBOURS = 6
RANDOM_SEED = 42

FONT_CANDIDATES = ("C:/Windows/Fonts/Nirmala.ttc", "C:/Windows/Fonts/arial.ttf")


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def find_dataset_root():
    """Locate the same NSL dataset the notebook trained on (archive/.../<0-35>/)."""
    candidates = [
        PROJECT_ROOT / "archive" / "NSL" / "NSL",
        PROJECT_ROOT / "archive" / "NSLv2" / "data",
    ]
    for root in candidates:
        if root.is_dir():
            backgrounds = [d for d in sorted(root.iterdir())
                           if d.is_dir() and (d / "0").is_dir()]
            if backgrounds:
                return root, backgrounds
    raise SystemExit(f"Could not find the NSL dataset under {PROJECT_ROOT / 'archive'}")


def sample_training_paths(root, backgrounds, per_class=SAMPLES_PER_CLASS, classes=None):
    """A reproducible sample of training images, balanced over classes and backgrounds."""
    rng = random.Random(RANDOM_SEED)
    wanted = range(api.N_CLASSES) if classes is None else classes
    items = []
    for cls in wanted:
        paths = []
        for bg in backgrounds:
            paths += sorted((bg / str(cls)).glob("*.jpg"))
        if not paths:
            continue
        take = min(per_class, len(paths))
        items += [(cls, p) for p in rng.sample(paths, take)]
    return items


# ---------------------------------------------------------------------------
# Model: predictions and embeddings
# ---------------------------------------------------------------------------
def embed_and_logits(tensor_batch):
    """Return (features_before_final_layer, logits) for a preprocessed batch.

    api.model.classifier is
        [AdaptiveAvgPool2d, Flatten, Linear(c_in,256), ReLU, Dropout, Linear(256,36)]
    so everything except the last Linear gives the 256-d penultimate feature.
    """
    with torch.no_grad():
        conv = api.model.features(tensor_batch)
        feats = conv
        for layer in list(api.model.classifier)[:-1]:
            feats = layer(feats)
        logits = list(api.model.classifier)[-1](feats)
    return feats, logits


def predict(image):
    """Top-5 (class_id, roman, devanagari, probability) for one PIL image."""
    x = api.preprocess(image)
    feats, logits = embed_and_logits(x)
    probs = torch.softmax(logits, dim=1)[0]
    top = torch.topk(probs, k=5)
    rows = [(int(i), api.ROMAN_MAP[int(i)], api.CHAR_MAP[int(i)], float(p))
            for p, i in zip(top.values, top.indices)]
    return rows, feats[0].cpu().numpy(), probs.cpu().numpy()


def embed_paths(paths, batch=64):
    """Penultimate features for a list of image paths, plus their 128x128 pixels."""
    feats, pixels = [], []
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        tensors, raws = [], []
        for p in chunk:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((api.IMG_SIZE, api.IMG_SIZE), Image.BILINEAR)
                arr = np.array(im, dtype=np.uint8)
            raws.append(arr)
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0))
        batch_t = torch.stack(tensors).to(api.DEVICE)
        f, _ = embed_and_logits(batch_t)
        feats.append(f.cpu().numpy())
        pixels.append(np.stack(raws))
    return np.concatenate(feats), np.concatenate(pixels)


def cosine(a, B):
    a = a / (np.linalg.norm(a) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return B @ a


def pixel_correlation(a_img, B_imgs):
    """Normalised pixel correlation on greyscale 128x128 (an explainable baseline)."""
    a = a_img.mean(axis=2).ravel().astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-9)
    B = B_imgs.mean(axis=3).reshape(len(B_imgs), -1).astype(np.float64)
    B = (B - B.mean(axis=1, keepdims=True)) / (B.std(axis=1, keepdims=True) + 1e-9)
    return (B @ a) / a.size


# ---------------------------------------------------------------------------
# Simple, measurable image statistics
# ---------------------------------------------------------------------------
def skin_mask(arr):
    """Crude RGB skin rule - good enough to measure hand size and position."""
    a = arr.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mx, mn = a.max(2), a.min(2)
    return ((r > 95) & (g > 40) & (b > 20) & ((mx - mn) > 15) &
            (np.abs(r - g) > 15) & (r > g) & (r > b))


def image_stats(arr):
    """Hand coverage, centre, bounding box, edge contact, brightness, contrast."""
    m = skin_mask(arr)
    h, w = m.shape
    stats = {
        "hand_fraction": float(m.mean()),
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
    }
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        stats.update({"centre_x": None, "centre_y": None, "bbox_fraction": None,
                      "touches_edge": None, "edges_touched": []})
        return stats
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    stats["centre_x"] = float(xs.mean() / w)
    stats["centre_y"] = float(ys.mean() / h)
    stats["bbox_fraction"] = float(((x2 - x1 + 1) * (y2 - y1 + 1)) / (w * h))
    k = max(2, h // 64)
    edges = [name for name, hit in (
        ("left", m[:, :k].mean() > 0.02), ("right", m[:, -k:].mean() > 0.02),
        ("top", m[:k, :].mean() > 0.02), ("bottom", m[-k:, :].mean() > 0.02)) if hit]
    stats["edges_touched"] = edges
    stats["touches_edge"] = bool(edges)
    return stats


def summarise(values):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if not len(arr):
        return None
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90))}


def compare_stat(name, live, ref, unit="", scale=1.0):
    if live is None or ref is None:
        return f"  {name:<22} live n/a"
    lo, hi = ref["p10"], ref["p90"]
    verdict = "typical" if lo <= live <= hi else ("LOW" if live < lo else "HIGH")
    return (f"  {name:<22} live {live * scale:7.2f}{unit}   "
            f"training {ref['mean'] * scale:6.2f}{unit} "
            f"(p10-p90 {lo * scale:.2f}-{hi * scale:.2f})   -> {verdict}")


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------
def load_font(size):
    for cand in FONT_CANDIDATES:
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                continue
    return ImageFont.load_default()


def contact_sheet(capture_img, top5, neighbours, out_path):
    """Capture + nearest training images, with true classes and scores."""
    cell, pad, header = 190, 14, 92
    cols = 1 + len(neighbours)
    W = pad + cols * (cell + pad)
    H = header + cell + 74
    sheet = Image.new("RGB", (W, H), (22, 22, 26))
    draw = ImageDraw.Draw(sheet)
    f_title = load_font(19)
    f_small = load_font(14)
    f_tiny = load_font(12)

    cid, roman, dev, prob = top5[0]
    draw.text((pad, 12), "Live ROI diagnosis", font=f_title, fill=(255, 255, 255))
    draw.text((pad, 40),
              f"model says: {roman} ({dev})  id {cid}   confidence {prob * 100:.2f}%",
              font=f_small, fill=(255, 215, 0))
    draw.text((pad, 62),
              "runner-up: " + ",  ".join(f"{r} {p * 100:.1f}%" for _, r, _, p in top5[1:4]),
              font=f_tiny, fill=(190, 190, 190))

    x = pad
    sheet.paste(capture_img.resize((cell, cell), Image.BILINEAR), (x, header))
    draw.rectangle([x, header, x + cell, header + cell], outline=(255, 215, 0), width=3)
    draw.text((x, header + cell + 6), "LIVE webcam ROI", font=f_small, fill=(255, 215, 0))
    draw.text((x, header + cell + 26), "(exactly what the CNN saw)",
              font=f_tiny, fill=(160, 160, 160))

    for n in neighbours:
        x += cell + pad
        with Image.open(n["path"]) as im:
            thumb = im.convert("RGB").resize((cell, cell), Image.BILINEAR)
        sheet.paste(thumb, (x, header))
        same = n["class_id"] == cid
        draw.rectangle([x, header, x + cell, header + cell],
                       outline=(80, 220, 120) if same else (220, 90, 90), width=2)
        draw.text((x, header + cell + 6),
                  f"{n['roman']} ({api.CHAR_MAP[n['class_id']]})  id {n['class_id']}",
                  font=f_small, fill=(80, 220, 120) if same else (220, 90, 90))
        draw.text((x, header + cell + 26),
                  f"cosine {n['cosine']:.3f}   pixel {n['pixel']:+.3f}",
                  font=f_tiny, fill=(190, 190, 190))
        draw.text((x, header + cell + 44), Path(n["path"]).parent.parent.name[:22],
                  font=f_tiny, fill=(140, 140, 140))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="path to a saved ROI image (debug_captures/*.jpg)")
    ap.add_argument("--per-class", type=int, default=SAMPLES_PER_CLASS,
                    help=f"training images sampled per class (default {SAMPLES_PER_CLASS})")
    ap.add_argument("--neighbours", type=int, default=TOP_K_NEIGHBOURS,
                    help=f"nearest neighbours to show (default {TOP_K_NEIGHBOURS})")
    args = ap.parse_args()

    capture_path = Path(args.capture)
    if not capture_path.is_file():
        raise SystemExit(f"No such capture: {capture_path}")

    root, backgrounds = find_dataset_root()
    print("=" * 74)
    print("LIVE PREDICTION DIAGNOSIS")
    print("=" * 74)
    print(f"capture    : {capture_path}")
    print(f"checkpoint : {api.CKPT_PATH}  (device {api.DEVICE})")
    print(f"dataset    : {root}")
    print(f"backgrounds: {', '.join(b.name for b in backgrounds)}")

    meta_path = capture_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        live = meta.get("last_api_reply") or {}
        print(f"\nsaved at   : {meta.get('timestamp')}")
        print(f"ROI        : {meta['roi']['width']}x{meta['roi']['height']} px at "
              f"x{meta['roi']['x1']}-{meta['roi']['x2']}, y{meta['roi']['y1']}-{meta['roi']['y2']}"
              f"  (mirrored: {meta.get('mirrored')})")
        if live:
            print(f"live API   : {live.get('roman')} {live.get('devanagari')} "
                  f"at {live.get('confidence', 0) * 100:.2f}%")

    # ---- 1. the model's own verdict, offline -------------------------------
    with Image.open(capture_path) as im:
        capture_img = im.convert("RGB")
    top5, feat, probs = predict(capture_img)
    print("\n" + "-" * 74)
    print("1. THE SAME CNN, RUN ON THE SAVED CROP (top 5)")
    print("-" * 74)
    for rank, (cid, roman, dev, p) in enumerate(top5, 1):
        bar = "#" * int(p * 40)
        print(f"  {rank}. id {cid:>2}  {roman:<5} {dev}  {p * 100:6.2f}%  {bar}")
    pred_id, pred_roman = top5[0][0], top5[0][1]

    # ---- 2. nearest neighbours among real training images ------------------
    print("\n" + "-" * 74)
    print(f"2. NEAREST TRAINING IMAGES  ({args.per_class}/class over "
          f"{api.N_CLASSES} classes, cosine on the CNN's 256-d penultimate layer)")
    print("-" * 74)
    items = sample_training_paths(root, backgrounds, args.per_class)
    paths = [p for _, p in items]
    labels = np.array([c for c, _ in items])
    print(f"  embedding {len(paths)} training images ...")
    train_feats, train_pixels = embed_paths(paths)

    cos = cosine(feat, train_feats)
    live_small = np.array(capture_img.resize((api.IMG_SIZE, api.IMG_SIZE), Image.BILINEAR))
    pix = pixel_correlation(live_small, train_pixels)

    order = np.argsort(-cos)[:args.neighbours]
    neighbours = [{"path": paths[i], "class_id": int(labels[i]),
                   "roman": api.ROMAN_MAP[int(labels[i])],
                   "cosine": float(cos[i]), "pixel": float(pix[i])} for i in order]
    print(f"\n  live ROI prediction: {pred_roman} - {top5[0][3] * 100:.1f}%\n")
    print("  nearest training neighbours (CNN feature space):")
    for rank, n in enumerate(neighbours, 1):
        flag = "  <- same as prediction" if n["class_id"] == pred_id else ""
        print(f"   {rank}. {n['roman']:<5} (id {n['class_id']:>2})  cosine {n['cosine']:.4f}"
              f"   pixel corr {n['pixel']:+.4f}{flag}")

    agree = sum(n["class_id"] == pred_id for n in neighbours)
    print(f"\n  {agree}/{len(neighbours)} nearest neighbours carry the predicted label "
          f"'{pred_roman}'")

    pix_order = np.argsort(-pix)[:args.neighbours]
    print("\n  nearest training neighbours (raw pixel correlation, for comparison):")
    for rank, i in enumerate(pix_order, 1):
        print(f"   {rank}. {api.ROMAN_MAP[int(labels[i])]:<5} (id {int(labels[i]):>2})"
              f"  pixel corr {pix[i]:+.4f}")

    # per-class mean similarity: is the predicted class really the closest cluster?
    print("\n  mean cosine similarity per class (top 6 clusters):")
    means = [(c, float(cos[labels == c].mean())) for c in np.unique(labels)]
    for c, m in sorted(means, key=lambda t: -t[1])[:6]:
        flag = "  <- predicted" if c == pred_id else ""
        print(f"   {api.ROMAN_MAP[c]:<5} (id {c:>2})  {m:.4f}{flag}")

    # ---- 3. measurable differences vs the predicted class -------------------
    print("\n" + "-" * 74)
    print(f"3. HOW THE CAPTURE DIFFERS FROM REAL '{pred_roman}' TRAINING IMAGES")
    print("-" * 74)
    live_stats = image_stats(live_small)
    ref_items = sample_training_paths(root, backgrounds, 40, classes=[pred_id])
    ref_stats = []
    for _, p in ref_items:
        with Image.open(p) as im:
            ref_stats.append(image_stats(
                np.array(im.convert("RGB").resize((api.IMG_SIZE, api.IMG_SIZE),
                                                  Image.BILINEAR))))
    refs = {k: summarise([s[k] for s in ref_stats])
            for k in ("hand_fraction", "bbox_fraction", "centre_x", "centre_y",
                      "brightness", "contrast")}
    print(compare_stat("hand pixels (%)", live_stats["hand_fraction"],
                       refs["hand_fraction"], "%", 100))
    print(compare_stat("hand bbox area (%)", live_stats["bbox_fraction"],
                       refs["bbox_fraction"], "%", 100))
    print(compare_stat("hand centre x (0-1)", live_stats["centre_x"], refs["centre_x"]))
    print(compare_stat("hand centre y (0-1)", live_stats["centre_y"], refs["centre_y"]))
    print(compare_stat("brightness (0-255)", live_stats["brightness"], refs["brightness"]))
    print(compare_stat("contrast (std)", live_stats["contrast"], refs["contrast"]))
    bg_live = 1 - (live_stats["hand_fraction"] or 0)
    bg_ref = 1 - refs["hand_fraction"]["mean"]
    print(f"  {'background (%)':<22} live {bg_live * 100:7.2f}%   "
          f"training {bg_ref * 100:6.2f}%")
    edge_ref = sum(bool(s["edges_touched"]) for s in ref_stats) / len(ref_stats)
    print(f"  {'hand touches crop edge':<22} live {str(live_stats['edges_touched']):<24}"
          f" training images doing so: {edge_ref * 100:.0f}%")
    if live_stats["edges_touched"]:
        print("     -> fingertips or wrist may be cut off; widen the ROI or move back")
    if live_stats["hand_fraction"] is not None and refs["hand_fraction"]:
        ratio = live_stats["hand_fraction"] / max(refs["hand_fraction"]["mean"], 1e-6)
        print(f"  {'hand size vs training':<22} {ratio:.2f}x "
              f"({'smaller' if ratio < 0.8 else 'larger' if ratio > 1.25 else 'comparable'})")

    # ---- 4. contact sheet ---------------------------------------------------
    out = RESULTS_DIR / f"diagnosis_{capture_path.stem}.png"
    contact_sheet(capture_img, top5, neighbours, out)
    print("\n" + "-" * 74)
    print(f"4. CONTACT SHEET -> {out}")
    print("-" * 74)
    print("  green border = neighbour shares the predicted class; red = a different class")
    print("\nNote: these are the dataset's own labels and the model's own features.")
    print("Nothing here is retrained, relabelled or modified.")


if __name__ == "__main__":
    main()
