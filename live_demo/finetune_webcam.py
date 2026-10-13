"""
Fine-tune a COPY of the Week 4 "Low LR" CNN on webcam images of my own hand.

    python finetune_webcam.py                 # stage 1: classifier head only
    python finetune_webcam.py --stage2        # then also the last conv block
    python finetune_webcam.py --eval-only     # just compare the two checkpoints

What this is and is not
-----------------------
models/low_lr.pt is the academic result and is never written to: this script
reads it, hashes it before and after the run, and refuses to save anywhere that
resolves to it. The fine-tuned weights go to models/live_finetuned.pt, which is
only a deployment adaptation for the webcam demo. main.py is not changed, so
the live API keeps serving the original model until the comparison below
justifies a switch.

Why fine-tuning rather than more webcam preprocessing
-----------------------------------------------------
The diagnostics showed the CNN placing webcam crops confidently inside the
wrong training clusters. Mirroring, ROI geometry, background and brightness
were all matched already, so what is left is a domain shift: new camera, new
signer, new room. The cure is a small amount of labelled data from that domain.

How it is kept conservative
---------------------------
  * start from the trained weights, never from scratch;
  * stage 1 trains the classifier only, with the convolutional feature
    extractor frozen, at lr 1e-4;
  * stage 2 (opt-in) additionally unfreezes the last conv block at lr 1e-5;
  * every BatchNorm layer stays in eval mode throughout, so a few dozen webcam
    images cannot move the running statistics the original training set up;
  * each epoch mixes the webcam images with a balanced sample of the ORIGINAL
    TRAINING split - all 36 classes - against catastrophic forgetting;
  * augmentation is the notebook's own train_augment: small rotation,
    translation, scale and brightness/contrast, and no horizontal flip;
  * early stopping on held-out webcam images, which the optimiser never sees.

The original split is reproduced exactly as Section 5 of the notebook made it
(MD5 de-duplication, then a stratified 70/15/15 with seed 0), so the replay
sample is drawn from train only. The original TEST split is never read.
"""

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2 as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Architecture, preprocessing constants and the 36-class mapping all come from
# the live API module; the dataset layout comes from the collector.
import main as api                      # noqa: E402
import collect_webcam_data as collector  # noqa: E402

PROJECT_ROOT = HERE.parent
ORIGINAL_CKPT = PROJECT_ROOT / "models" / "low_lr.pt"
DEFAULT_OUT = PROJECT_ROOT / "models" / "live_finetuned.pt"
CACHE_DIR = HERE / "cache"
REPORT_DIR = HERE / "finetune_reports"

DEVICE = api.DEVICE
SEED = 0
VAL_FRACTION = 0.20

# What Section 5 of Shreejal.Shrestha.ipynb printed. The reproduction below is
# checked against these numbers, because a replay sample is only safe if the
# split it came from really is the notebook's split.
NOTEBOOK_SPLIT = {"indexed": 54000, "dedup": 53329,
                  "train": 37330, "val": 7999, "test": 8000}

# Identical to the notebook's Section 14.3 augmentation. Realistic for a
# webcam (small shifts, small zoom, small tilt, mild lighting change) and
# deliberately without RandomHorizontalFlip: mirroring a sign can turn it into
# a different character.
train_augment = T.Compose([
    T.RandomAffine(degrees=12, translate=(0.10, 0.10), scale=(0.90, 1.10)),
    T.ColorJitter(brightness=0.25, contrast=0.25),
])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def md5_file(path, buf=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_images(paths):
    """Paths -> uint8 NHWC array, resized exactly as api.preprocess does."""
    if not paths:
        return np.zeros((0, api.IMG_SIZE, api.IMG_SIZE, 3), dtype=np.uint8)
    out = np.empty((len(paths), api.IMG_SIZE, api.IMG_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB").resize((api.IMG_SIZE, api.IMG_SIZE), Image.BILINEAR)
            out[i] = np.asarray(im, dtype=np.uint8)
    return out


def to_device_batch(x_uint8_nhwc):
    """uint8 NHWC on the CPU -> float32 NCHW in [0,1] on the device."""
    xb = torch.from_numpy(x_uint8_nhwc).to(DEVICE)
    return xb.permute(0, 3, 1, 2).float().div_(255.0)


# ---------------------------------------------------------------------------
# 1. The webcam dataset and its deterministic 80/20 split
# ---------------------------------------------------------------------------
def webcam_items(dataset_dir):
    """[(class_id, Path)] for every image under <dataset_dir>/<id>_<roman>/.

    The label is taken from the directory's numeric prefix and the roman
    suffix is verified against the mapping, so a renamed folder is an error
    rather than a silently wrong label.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"No webcam dataset at {dataset_dir}.\n"
                         f"Collect some first:  python collect_webcam_data.py ka")
    items = []
    for sub in sorted(dataset_dir.iterdir()):
        if not sub.is_dir():
            continue
        prefix, _, roman = sub.name.partition("_")
        if not prefix.isdigit() or int(prefix) not in api.ROMAN_MAP:
            print(f"  skipping {sub.name}: not a '<id>_<roman>' class folder")
            continue
        class_id = int(prefix)
        if roman != api.ROMAN_MAP[class_id]:
            raise SystemExit(
                f"{sub.name} disagrees with the class mapping: id {class_id} is "
                f"'{api.ROMAN_MAP[class_id]}', not '{roman}'. The mapping is fixed; "
                f"rename the folder to {collector.class_dir_name(class_id)}.")
        items += [(class_id, p) for p in sorted(sub.glob("*.jpg"))]
    if not items:
        raise SystemExit(f"{dataset_dir} holds no images yet.")
    return items


def split_webcam(items, seed=SEED, val_fraction=VAL_FRACTION):
    """Deterministic per-class train/validation split of the webcam images.

    Each image is assigned by a hash of its own path salted with the seed, so
    the assignment is reproducible and - unlike a shuffle of the whole folder -
    an image does not move between train and validation when more images are
    collected later. Every class with at least two images keeps at least one
    image on each side.
    """
    by_class = {}
    for class_id, path in items:
        by_class.setdefault(class_id, []).append(path)

    train, val = [], []
    for class_id in sorted(by_class):
        paths = sorted(by_class[class_id])
        keyed = sorted(
            (int(hashlib.md5(f"{seed}|{p.parent.name}/{p.name}".encode()).hexdigest()[:8],
                 16) / 0xFFFFFFFF, p)
            for p in paths
        )
        n_val = sum(1 for score, _ in keyed if score < val_fraction)
        if len(keyed) >= 2:                      # never leave a side empty
            n_val = min(max(n_val, 1), len(keyed) - 1)
        val += [(class_id, p) for _, p in keyed[:n_val]]
        train += [(class_id, p) for _, p in keyed[n_val:]]
    return train, val


# ---------------------------------------------------------------------------
# 2. The original split, reproduced exactly as the notebook made it
# ---------------------------------------------------------------------------
def find_original_root():
    """The notebook's find_dataset_root, over this project's archive/."""
    for current, subdirs, _ in os.walk(PROJECT_ROOT / "archive"):
        if {"Plain Background", "Random Background"} & set(subdirs):
            return Path(current)
    raise SystemExit(f"Could not locate the NSL dataset under "
                     f"{PROJECT_ROOT / 'archive'}")


def index_original(root):
    """Every image under root, in the notebook's Section 2 order."""
    records = []
    for background in sorted(os.listdir(root)):
        bg_path = root / background
        if not bg_path.is_dir():
            continue
        for class_folder in sorted(os.listdir(bg_path),
                                   key=lambda x: int(x) if x.isdigit() else x):
            cls_path = bg_path / class_folder
            if not cls_path.is_dir():
                continue
            for fname in os.listdir(cls_path):      # raw order, as the notebook
                records.append((int(class_folder), cls_path / fname))
    return records


def build_original_split(allow_mismatch=False):
    """Reproduce Section 5: MD5 de-dup, then stratified 70/15/15 with seed 0.

    Returns {"train": [(class_id, path)], "val": [...], "sizes": {...}}.
    The test split is computed (its size is checked) but its paths are dropped
    here, so nothing downstream can train on them by accident.
    """
    from sklearn.model_selection import train_test_split  # only needed here

    root = find_original_root()
    records = index_original(root)
    print(f"  dataset root : {root}")
    print(f"  indexed      : {len(records):,} images")

    print("  hashing (MD5 de-duplication, as the notebook did before splitting) ...")
    seen, kept = set(), []
    t0 = time.time()
    for i, (class_id, path) in enumerate(records, 1):
        digest = md5_file(path)
        if digest not in seen:                      # drop_duplicates(keep="first")
            seen.add(digest)
            kept.append((class_id, path))
        if i % 10000 == 0:
            print(f"    {i:,}/{len(records):,}  ({time.time() - t0:.0f}s)")

    labels = [c for c, _ in kept]
    idx = np.arange(len(kept))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=labels,
                                           random_state=SEED)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50,
                                         stratify=temp_labels, random_state=SEED)

    sizes = {"indexed": len(records), "dedup": len(kept), "train": len(train_idx),
             "val": len(val_idx), "test": len(test_idx)}
    print(f"  de-duplicated: {sizes['dedup']:,}")
    print(f"  train/val/test: {sizes['train']:,} / {sizes['val']:,} / {sizes['test']:,}")

    if sizes != NOTEBOOK_SPLIT:
        message = ("The reproduced split does not match the notebook's "
                   f"({sizes} vs {NOTEBOOK_SPLIT}). Replay samples could then "
                   "include images the notebook put in its test split.")
        if not allow_mismatch:
            raise SystemExit("  " + message +
                             "\n  Re-run with --allow-split-mismatch to accept it, "
                             "or with --replay-per-class 0 to skip replay entirely.")
        print("  WARNING: " + message)

    return {"root": str(root), "sizes": sizes,
            "train": [kept[i] for i in train_idx],
            "val": [kept[i] for i in val_idx]}


def original_split(rebuild=False, allow_mismatch=False):
    """The reproduced split, cached so the MD5 pass runs once."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "original_split.json"
    if cache.exists() and not rebuild:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("sizes") == NOTEBOOK_SPLIT or allow_mismatch:
            print(f"  using cached split manifest ({cache.name})")
            return {"root": data["root"], "sizes": data["sizes"],
                    "train": [(c, PROJECT_ROOT / p) for c, p in data["train"]],
                    "val": [(c, PROJECT_ROOT / p) for c, p in data["val"]]}
        print("  cached manifest disagrees with the notebook's sizes; rebuilding")

    split = build_original_split(allow_mismatch)
    cache.write_text(json.dumps({
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
        "root": split["root"], "sizes": split["sizes"],
        "seed": SEED,
        "note": "Section 5 of the notebook, reproduced. Test paths are not stored.",
        "train": [[c, str(p.relative_to(PROJECT_ROOT))] for c, p in split["train"]],
        "val": [[c, str(p.relative_to(PROJECT_ROOT))] for c, p in split["val"]],
    }), encoding="utf-8")
    print(f"  split manifest cached in {cache}")
    return split


def balanced_sample(items, per_class, seed=SEED, tag="sample"):
    """`per_class` images from each of the 36 classes, deterministically."""
    by_class = {}
    for class_id, path in items:
        by_class.setdefault(class_id, []).append(path)
    picked = []
    for class_id in range(api.N_CLASSES):
        paths = sorted(by_class.get(class_id, []))
        if not paths:
            print(f"  warning: class {class_id} has no images in the original split")
            continue
        rng = random.Random(f"{seed}|{tag}|{class_id}")
        picked += [(class_id, p) for p in rng.sample(paths, min(per_class, len(paths)))]
    return picked


# ---------------------------------------------------------------------------
# 3. Model, freezing, metrics
# ---------------------------------------------------------------------------
def load_checkpoint(path):
    """A SignCNN with the given weights, loaded strict=True."""
    model = api.SignCNN(n_classes=api.N_CLASSES, dropout=api.DROPOUT,
                        base_filters=api.BASE_FILTERS)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE)


def conv_blocks(features):
    """Indices of each conv block in the flat features Sequential (split at pools)."""
    blocks, current = [], []
    for i, layer in enumerate(features):
        current.append(i)
        if isinstance(layer, nn.MaxPool2d):
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def set_trainable(model, unfreeze_last_block):
    """Freeze the feature extractor; optionally re-open its last conv block.

    BatchNorm weights are left frozen even in stage 2, and the whole features
    module is put in eval mode during training, so the running statistics
    learned from 37k images are not rewritten by a few dozen webcam frames.
    """
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    opened = []
    if unfreeze_last_block:
        for i in conv_blocks(model.features)[-1]:
            layer = model.features[i]
            if isinstance(layer, nn.Conv2d):
                for p in layer.parameters():
                    p.requires_grad = True
                opened.append(f"features[{i}] {layer.__class__.__name__}")
    return opened


def trainable_report(model):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return names, n_train, n_total


@torch.no_grad()
def predict_all(model, X, batch=64):
    """Predicted class and its probability for a uint8 NHWC array."""
    model.eval()
    preds, confs = [], []
    for i in range(0, len(X), batch):
        logits = model(to_device_batch(X[i:i + batch]))
        probs = torch.softmax(logits, dim=1)
        top_conf, top_idx = probs.max(dim=1)
        preds.append(top_idx.cpu().numpy())
        confs.append(top_conf.cpu().numpy())
    if not preds:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    return np.concatenate(preds), np.concatenate(confs)


def confusion(y_true, y_pred, n_classes=api.N_CLASSES):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics(y_true, y_pred, n_classes=api.N_CLASSES):
    """Accuracy plus per-class precision/recall/F1 and their macro averages.

    The macro averages are taken over the classes that actually occur in
    y_true: with a 5-class pilot, averaging over all 36 would silently divide
    by 36 and report a number that means nothing.
    """
    cm = confusion(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    pred_tot = cm.sum(axis=0).astype(float)
    true_tot = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    recall = np.divide(tp, true_tot, out=np.zeros_like(tp), where=true_tot > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = np.nonzero(true_tot)[0]
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_precision": float(precision[present].mean()) if len(present) else 0.0,
        "macro_recall": float(recall[present].mean()) if len(present) else 0.0,
        "macro_f1": float(f1[present].mean()) if len(present) else 0.0,
        "per_class": {int(c): {"support": int(true_tot[c]),
                               "precision": round(float(precision[c]), 4),
                               "recall": round(float(recall[c]), 4),
                               "f1": round(float(f1[c]), 4)} for c in present},
        "confusion": cm,
        "classes_present": [int(c) for c in present],
    }


def format_metrics(title, m):
    lines = [f"{title}",
             f"  images            : {m['n']}",
             f"  accuracy          : {m['accuracy'] * 100:6.2f}%",
             f"  macro precision   : {m['macro_precision'] * 100:6.2f}%",
             f"  macro recall      : {m['macro_recall'] * 100:6.2f}%",
             f"  macro F1          : {m['macro_f1'] * 100:6.2f}%"]
    return "\n".join(lines)


def format_per_class(m):
    lines = ["  class            support  precision  recall    F1"]
    for c, row in m["per_class"].items():
        lines.append(f"  {c:>2} {api.ROMAN_MAP[c]:<12} {row['support']:>5}"
                     f"   {row['precision'] * 100:7.2f}  {row['recall'] * 100:6.2f}"
                     f"  {row['f1'] * 100:6.2f}")
    return "\n".join(lines)


def format_confusion(m, y_pred):
    """Confusion matrix over the classes that appear as a true or predicted label."""
    cm = m["confusion"]
    shown = sorted(set(m["classes_present"]) | set(int(p) for p in np.unique(y_pred)))
    header = "         " + "".join(f"{api.ROMAN_MAP[c][:4]:>6}" for c in shown)
    lines = ["  rows = true, columns = predicted", header]
    for r in shown:
        row = "".join(f"{cm[r, c]:>6}" for c in shown)
        marker = " " if r in m["classes_present"] else "*"
        lines.append(f"  {api.ROMAN_MAP[r][:6]:<6}{marker}{row}")
    if any(r not in m["classes_present"] for r in shown):
        lines.append("  (* a class with no held-out images, shown because it was "
                     "predicted at least once)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Fine-tuning
# ---------------------------------------------------------------------------
def train_stage(model, X_train, y_train, X_val, y_val, *, lr, max_epochs,
                patience, batch_size, unfreeze_last_block, label):
    """One conservative stage. Returns (best_state, history, best_accuracy)."""
    opened = set_trainable(model, unfreeze_last_block)
    names, n_train_params, n_total = trainable_report(model)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    print(f"\n=== {label} ===")
    print(f"  learning rate     : {lr}")
    print(f"  trainable tensors : {len(names)}  "
          f"({n_train_params:,} of {n_total:,} parameters)")
    print("  training          : classifier" +
          (" + last conv block (" + ", ".join(opened) + ")" if opened else " only"))
    print("  BatchNorm         : frozen, eval mode (running stats untouched)")
    print(f"  epochs            : up to {max_epochs}, early stop patience {patience}")

    base_pred, _ = predict_all(model, X_val)
    best_acc = float((base_pred == y_val).mean()) if len(y_val) else 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    print(f"  epoch  0 (before this stage)         webcam val acc {best_acc * 100:6.2f}%")

    history = [{"epoch": 0, "train_loss": None, "train_acc": None,
                "val_acc": best_acc}]
    stale = 0
    generator = np.random.default_rng(SEED)

    for epoch in range(1, max_epochs + 1):
        model.train()
        model.features.eval()          # keep BatchNorm's running stats fixed
        order = generator.permutation(len(X_train))
        running_loss, seen, correct = 0.0, 0, 0

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = to_device_batch(X_train[idx])
            yb = torch.from_numpy(y_train[idx]).to(DEVICE).long()
            xb = train_augment(xb)                      # no horizontal flip
            optimiser.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * len(idx)
            seen += len(idx)
            correct += int((logits.argmax(dim=1) == yb).sum())

        val_pred, _ = predict_all(model, X_val)
        val_acc = float((val_pred == y_val).mean()) if len(y_val) else 0.0
        history.append({"epoch": epoch, "train_loss": running_loss / max(1, seen),
                        "train_acc": correct / max(1, seen), "val_acc": val_acc})

        note = ""
        if val_acc > best_acc:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            note = "  <- best so far (kept)"
        else:
            stale += 1
        print(f"  epoch {epoch:>2}  train loss {running_loss / max(1, seen):.4f}  "
              f"train acc {correct / max(1, seen) * 100:6.2f}%  "
              f"webcam val acc {val_acc * 100:6.2f}%{note}")

        if stale >= patience:
            print(f"  early stop: no webcam-validation improvement for "
                  f"{patience} epochs")
            break

    model.load_state_dict(best_state)
    print(f"  best webcam validation accuracy {best_acc * 100:.2f}% "
          f"(epoch {best_epoch}{' - this stage changed nothing' if best_epoch == 0 else ''})")
    return best_state, history, best_acc, best_epoch


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------
def comparison_report(models, X_val, y_val, X_orig, y_orig, context):
    """Both checkpoints on the held-out webcam images, and on original data."""
    chunks = []
    results = {}

    chunks.append("=" * 74)
    chunks.append("HELD-OUT WEBCAM VALIDATION  (never seen by the optimiser)")
    chunks.append("=" * 74)
    for name, model in models.items():
        pred, conf = predict_all(model, X_val)
        m = metrics(y_val, pred)
        results[name] = {"webcam": {k: v for k, v in m.items() if k != "confusion"},
                         "webcam_mean_confidence": float(conf.mean()) if len(conf) else 0.0}
        chunks.append("")
        chunks.append(format_metrics(f"{name}", m))
        chunks.append(f"  mean confidence   : {conf.mean() * 100:6.2f}%"
                      if len(conf) else "")
        chunks.append(format_per_class(m))
        chunks.append("")
        chunks.append(format_confusion(m, pred))

    if len(X_orig):
        chunks.append("")
        chunks.append("=" * 74)
        chunks.append("ORIGINAL-DATA CHECK  (sample of the notebook's VALIDATION split;")
        chunks.append("the original test split is never read by this script)")
        chunks.append("=" * 74)
        for name, model in models.items():
            pred, _ = predict_all(model, X_orig)
            m = metrics(y_orig, pred)
            results[name]["original"] = {k: v for k, v in m.items()
                                         if k not in ("confusion", "per_class")}
            chunks.append("")
            chunks.append(format_metrics(f"{name}", m))

    names = list(models)
    if len(names) == 2:
        a, b = names
        wa = results[a]["webcam"]["accuracy"]
        wb = results[b]["webcam"]["accuracy"]
        fa = results[a]["webcam"]["macro_f1"]
        fb = results[b]["webcam"]["macro_f1"]
        chunks.append("")
        chunks.append("=" * 74)
        chunks.append("VERDICT")
        chunks.append("=" * 74)
        chunks.append(f"  webcam accuracy : {a} {wa * 100:.2f}%  ->  {b} {wb * 100:.2f}%"
                      f"   ({(wb - wa) * 100:+.2f} points)")
        chunks.append(f"  webcam macro F1 : {a} {fa * 100:.2f}%  ->  {b} {fb * 100:.2f}%"
                      f"   ({(fb - fa) * 100:+.2f} points)")
        if "original" in results[a]:
            oa = results[a]["original"]["accuracy"]
            ob = results[b]["original"]["accuracy"]
            chunks.append(f"  original accuracy: {a} {oa * 100:.2f}%  ->  "
                          f"{b} {ob * 100:.2f}%   ({(ob - oa) * 100:+.2f} points)")
        n_val = results[a]["webcam"]["n"]
        chunks.append("")
        chunks.append(f"  These are {n_val} held-out webcam images. Early stopping also "
                      "used them,")
        chunks.append("  so treat the fine-tuned figure as slightly optimistic; the real "
                      "test is")
        chunks.append("  the live demo. main.py still serves low_lr.pt and is unchanged.")

    report = "\n".join(c for c in chunks if c is not None)
    return report, results


def write_report(text, payload):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    (REPORT_DIR / f"{stamp}_webcam_finetune.txt").write_text(text, encoding="utf-8")
    (REPORT_DIR / f"{stamp}_webcam_finetune.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return REPORT_DIR / f"{stamp}_webcam_finetune.txt"


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Conservatively fine-tune a copy of low_lr.pt on webcam images.")
    ap.add_argument("--dataset-dir", default=str(collector.DATASET_DIR),
                    help="webcam dataset root (default live_demo/webcam_dataset)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="where to save the fine-tuned copy "
                         "(default models/live_finetuned.pt)")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="stage 1 learning rate (default 1e-4)")
    ap.add_argument("--epochs", type=int, default=12,
                    help="stage 1 epoch ceiling (default 12)")
    ap.add_argument("--patience", type=int, default=3,
                    help="early-stopping patience (default 3)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--stage2", action="store_true",
                    help="after stage 1, also unfreeze the LAST conv block")
    ap.add_argument("--stage2-lr", type=float, default=1e-5,
                    help="stage 2 learning rate (default 1e-5)")
    ap.add_argument("--stage2-epochs", type=int, default=8)
    ap.add_argument("--replay-per-class", type=int, default=8,
                    help="original TRAINING images per class mixed in each epoch, "
                         "all 36 classes (default 8; 0 disables replay)")
    ap.add_argument("--orig-check-per-class", type=int, default=10,
                    help="original VALIDATION images per class for the "
                         "forgetting check (default 10; 0 disables it)")
    ap.add_argument("--eval-only", action="store_true",
                    help="only compare the two checkpoints, train nothing")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="recompute the original split manifest from scratch")
    ap.add_argument("--allow-split-mismatch", action="store_true",
                    help="proceed even if the reproduced original split does not "
                         "match the notebook's sizes")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out).resolve()

    # --- never touch the academic checkpoint -------------------------------
    if not ORIGINAL_CKPT.exists():
        raise SystemExit(f"Original checkpoint missing: {ORIGINAL_CKPT}")
    if out_path == ORIGINAL_CKPT.resolve() or out_path.name.lower() == "low_lr.pt":
        raise SystemExit("Refusing to write to low_lr.pt: that is the original "
                         "academic result. Choose another --out.")
    original_md5 = md5_file(ORIGINAL_CKPT)

    set_seed(SEED)
    print(f"device: {DEVICE}")
    print(f"original checkpoint: {ORIGINAL_CKPT.name}  md5 {original_md5[:12]}... "
          f"(read-only)")
    print(f"fine-tuned copy    : {out_path}")

    # --- webcam data and its split ----------------------------------------
    print("\nWebcam dataset")
    items = webcam_items(args.dataset_dir)
    train_items, val_items = split_webcam(items)
    present = sorted({c for c, _ in items})
    print(f"  root            : {args.dataset_dir}")
    print(f"  images          : {len(items)} over {len(present)} class(es)")
    print(f"  split (seed {SEED}): {len(train_items)} fine-tuning / "
          f"{len(val_items)} validation  (~{VAL_FRACTION:.0%} held out)")
    for class_id in present:
        n_t = sum(1 for c, _ in train_items if c == class_id)
        n_v = sum(1 for c, _ in val_items if c == class_id)
        flag = "" if n_t + n_v >= 20 else "   (few images)"
        print(f"    {class_id:>2} {api.ROMAN_MAP[class_id]:<6} "
              f"train {n_t:>3}  val {n_v:>3}{flag}")
    if len(present) < api.N_CLASSES:
        print(f"  NOTE: this is a {len(present)}-class pilot, not a 36-class webcam "
              f"classifier.\n        Everything below is evidence about these "
              f"{len(present)} classes only.")

    X_wtrain = load_images([p for _, p in train_items])
    y_wtrain = np.array([c for c, _ in train_items], dtype=np.int64)
    X_wval = load_images([p for _, p in val_items])
    y_wval = np.array([c for c, _ in val_items], dtype=np.int64)

    # --- original data: replay sample and forgetting check ----------------
    X_replay = np.zeros((0, api.IMG_SIZE, api.IMG_SIZE, 3), dtype=np.uint8)
    y_replay = np.zeros(0, dtype=np.int64)
    X_ocheck, y_ocheck = X_replay, y_replay
    need_original = (not args.eval_only and args.replay_per_class > 0) \
        or args.orig_check_per_class > 0

    if need_original:
        print("\nOriginal dataset (Section 5 split, reproduced)")
        split = original_split(args.rebuild_cache, args.allow_split_mismatch)
        if not args.eval_only and args.replay_per_class > 0:
            replay = balanced_sample(split["train"], args.replay_per_class, tag="replay")
            X_replay, y_replay = (load_images([p for _, p in replay]),
                                  np.array([c for c, _ in replay], dtype=np.int64))
            print(f"  replay sample   : {len(replay)} images from the TRAIN split "
                  f"({args.replay_per_class}/class, "
                  f"{len(set(y_replay.tolist()))} classes)")
        if args.orig_check_per_class > 0:
            check = balanced_sample(split["val"], args.orig_check_per_class, tag="check")
            X_ocheck, y_ocheck = (load_images([p for _, p in check]),
                                  np.array([c for c, _ in check], dtype=np.int64))
            print(f"  forgetting check: {len(check)} images from the VAL split "
                  f"({args.orig_check_per_class}/class)")

    # --- the original model, for comparison -------------------------------
    baseline = load_checkpoint(ORIGINAL_CKPT)
    baseline.eval()

    history = []
    stages = []
    if args.eval_only:
        if not out_path.exists():
            raise SystemExit(f"--eval-only needs an existing {out_path}; run the "
                             f"fine-tuning first.")
        tuned = load_checkpoint(out_path)
        print(f"\nLoaded the fine-tuned checkpoint for evaluation only: {out_path.name}")
    else:
        X_ft = np.concatenate([X_wtrain, X_replay]) if len(X_replay) else X_wtrain
        y_ft = np.concatenate([y_wtrain, y_replay]) if len(y_replay) else y_wtrain
        print(f"\nFine-tuning set : {len(X_ft)} images "
              f"({len(X_wtrain)} webcam + {len(X_replay)} original) over "
              f"{len(set(y_ft.tolist()))} classes")
        if len(X_wval) == 0:
            raise SystemExit("No held-out webcam images: collect at least two per class.")

        tuned = load_checkpoint(ORIGINAL_CKPT)
        state, hist, acc, best_epoch = train_stage(
            tuned, X_ft, y_ft, X_wval, y_wval, lr=args.lr, max_epochs=args.epochs,
            patience=args.patience, batch_size=args.batch_size,
            unfreeze_last_block=False,
            label="Stage 1: classifier only, conv features frozen")
        history += [{"stage": 1, **h} for h in hist]
        stages.append({"stage": 1, "lr": args.lr, "best_val_acc": acc,
                       "best_epoch": best_epoch, "unfroze_last_conv_block": False})

        if args.stage2:
            state, hist, acc, best_epoch = train_stage(
                tuned, X_ft, y_ft, X_wval, y_wval, lr=args.stage2_lr,
                max_epochs=args.stage2_epochs, patience=args.patience,
                batch_size=args.batch_size, unfreeze_last_block=True,
                label="Stage 2: classifier + last conv block, lower learning rate")
            history += [{"stage": 2, **h} for h in hist]
            stages.append({"stage": 2, "lr": args.stage2_lr, "best_val_acc": acc,
                           "best_epoch": best_epoch, "unfroze_last_conv_block": True})
        elif acc < 0.90:
            print(f"\n  Stage 1 reached {acc * 100:.2f}% on held-out webcam images. "
                  f"If that is not\n  enough, re-run with --stage2 to additionally "
                  f"unfreeze the last conv block\n  at lr {args.stage2_lr}.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tuned.state_dict(), out_path)
        print(f"\nSaved the best webcam-validation weights to {out_path}")

    # --- comparison --------------------------------------------------------
    print()
    report, results = comparison_report(
        {"low_lr.pt (original)": baseline, f"{out_path.name} (fine-tuned)": tuned},
        X_wval, y_wval, X_ocheck, y_ocheck, context=vars(args))
    print(report)

    payload = {
        "run": _dt.datetime.now().isoformat(timespec="seconds"),
        "device": str(DEVICE),
        "mode": "eval-only" if args.eval_only else "fine-tune",
        "arguments": vars(args),
        "original_checkpoint": {"path": str(ORIGINAL_CKPT), "md5": original_md5},
        "finetuned_checkpoint": str(out_path),
        "webcam_dataset": {
            "images": len(items), "classes_present": present,
            "train": len(train_items), "val": len(val_items),
            "val_fraction": VAL_FRACTION, "split_seed": SEED,
        },
        "replay_images": int(len(X_replay)),
        "original_check_images": int(len(X_ocheck)),
        "stages": stages,
        "history": history,
        "results": results,
    }
    path = write_report(report, payload)
    print(f"\nReport written to {path}")

    # --- prove the academic checkpoint is untouched ------------------------
    after = md5_file(ORIGINAL_CKPT)
    if after != original_md5:
        raise SystemExit(f"FATAL: {ORIGINAL_CKPT} changed during this run "
                         f"({original_md5} -> {after}).")
    print(f"low_lr.pt unchanged (md5 {after[:12]}... verified after the run)")
    print("main.py still loads low_lr.pt; nothing about the live API was changed.")


if __name__ == "__main__":
    main()
