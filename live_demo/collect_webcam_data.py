"""
Collect labelled webcam images of one sign class, for webcam fine-tuning.

    python collect_webcam_data.py ka
    python collect_webcam_data.py chha --target 50

Why this exists
---------------
The Week 4 CNN reaches ~100% on the original dataset but mispredicts live
webcam frames: the webcam is a different camera, a different signer and a
different room, so its images land in the wrong place in the CNN's feature
space. That is a domain shift, not a preprocessing bug, so the fix is labelled
data from *this* webcam. This script is only the collector; nothing is trained
here and no checkpoint is touched.

What is saved
-------------
Exactly the pixels the CNN receives in the live demo:

  * the RAW, un-mirrored frame (the mirrored picture is for the signer only);
  * the same square ROI geometry as webcam_demo.py, imported from it rather
    than re-typed, so the two can never drift apart;
  * no horizontal flip, no brightness normalisation, no other preprocessing -
    the resize to 128x128 happens later, in the same code path as inference.

Layout (one directory per class, label taken from the numeric prefix):

    live_demo/webcam_dataset/00_ka/ka_20261006_143012_481.jpg
    live_demo/webcam_dataset/06_chha/...

The prefix is there on purpose. Windows paths are case-insensitive, so the
mapping's 'Ta' (class 10) and 'ta' (class 15) - and five more pairs like them -
would collide in a directory named after the roman label alone.

Controls
--------
    SPACE   capture one frame
    A       toggle automatic capture (one frame every ~0.3 s)
    Q       quit
"""

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The 36-class mapping comes from the live API module, never a second copy of
# it; the ROI geometry comes from the live webcam client for the same reason.
import main as api            # noqa: E402
import webcam_demo as demo    # noqa: E402

DATASET_DIR = HERE / "webcam_dataset"
META_NAME = "_capture_meta.json"

JPEG_QUALITY = 95             # cv2's default, i.e. what the live demo POSTs
TARGET_DEFAULT = 40           # 30-50 per class is the useful range
AUTO_INTERVAL_DEFAULT = 0.30  # seconds between automatic captures
AUTO_INTERVAL_MIN = 0.10

WINDOW_NAME = "Collect webcam data"
GREEN, WHITE, YELLOW, RED, GREY = demo.GREEN, demo.WHITE, demo.YELLOW, demo.RED, demo.GREY
FONT = demo.FONT
CONTROLS = "SPACE: capture   A: auto   Q: quit"

# Rotated through as the count grows, so the 40 images of a class are not 40
# copies of one pose. The sign itself must stay correct throughout.
VARIATION_HINTS = [
    "normal pose, centred in the box",
    "hand slightly left, then slightly right",
    "a little nearer, then a little further",
    "tilt the hand slightly, both ways",
    "let the lighting vary a little",
    "free mix of all of the above",
]

NAME_TO_ID = {name: cid for cid, name in api.ROMAN_MAP.items()}


# ---------------------------------------------------------------------------
# Class names and directories
# ---------------------------------------------------------------------------
def class_dir_name(class_id):
    """'ka' (class 0) -> '00_ka'. Unique even on a case-insensitive filesystem."""
    return f"{class_id:02d}_{api.ROMAN_MAP[class_id]}"


def resolve_class(token):
    """Validate a command-line class against the 36-class mapping.

    Accepts the exact roman label ('ka', 'Ta') or the numeric id ('0', '10').
    Case is significant: 'ta' and 'Ta' are two different characters in this
    dataset, so a case-only mismatch is rejected instead of guessed.
    """
    if token in NAME_TO_ID:
        cid = NAME_TO_ID[token]
        return cid, api.ROMAN_MAP[cid]

    if token.isdigit() and int(token) in api.ROMAN_MAP:
        cid = int(token)
        return cid, api.ROMAN_MAP[cid]

    near = sorted(n for n in NAME_TO_ID if n.lower() == token.lower())
    if near:
        spelled = ", ".join(f"{n} (id {NAME_TO_ID[n]})" for n in near)
        raise SystemExit(
            f"'{token}' is not a class name, but these differ from it only in "
            f"case: {spelled}.\nThese are different Nepali characters, so spell "
            f"the one you mean exactly, or pass its numeric id."
        )

    listing = "  ".join(f"{cid}:{name}" for cid, name in sorted(api.ROMAN_MAP.items()))
    raise SystemExit(f"Unknown class '{token}'. The 36 classes are:\n  {listing}")


def existing_count(class_dir):
    return len(sorted(class_dir.glob("*.jpg")))


def write_meta(class_dir, class_id, roman, frame_shape, box, count):
    """Record what geometry these images were captured with, for later checks."""
    x1, y1, x2, y2 = box
    meta = {
        "class_id": class_id,
        "roman": roman,
        "devanagari": api.CHAR_MAP[class_id],
        "images": count,
        "last_updated": _dt.datetime.now().isoformat(timespec="seconds"),
        "frame_size": [int(frame_shape[1]), int(frame_shape[0])],
        "roi": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "size_px": x2 - x1,
                "fraction": demo.ROI_FRACTION,
                "x_offset": demo.ROI_X_OFFSET,
                "y_offset": demo.ROI_Y_OFFSET},
        "mirrored": False,
        "brightness_normalisation": False,
        "jpeg_quality": JPEG_QUALITY,
    }
    (class_dir / META_NAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")


def check_geometry(class_dir, box):
    """Warn if this session's ROI differs from the one already in the folder.

    Mixing two ROI sizes inside one class is allowed (the loader resizes
    everything to 128x128), but it is worth knowing about, because a changed
    camera resolution also changes how much of the frame the hand fills.
    """
    meta_path = class_dir / META_NAME
    if not meta_path.exists():
        return
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    old_size = old.get("roi", {}).get("size_px")
    if old_size and old_size != box[2] - box[0]:
        print(f"  note: earlier images in this class used a {old_size}px ROI, "
              f"this camera gives {box[2] - box[0]}px")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_status(display, roman, class_id, captured, target, auto, interval,
                roi_left, hint, hints):
    """Counts and instructions, kept clear of the ROI so nothing covers the hand."""
    pad = 14
    lines = [
        f"Class: {roman}  (id {class_id})",
        f"Captured: {captured}",
        f"Target: {target}",
        f"AUTO: {'ON' if auto else 'OFF'}" + (f"  every {interval:.2f}s" if auto else ""),
    ]
    max_w = max(150, roi_left - 2 * pad - 10)
    scale = 0.62
    for line in lines:
        scale = min(scale, demo.fit_scale(line, max_w, 0.62, 2))
    text_w = max(cv2.getTextSize(t, FONT, scale, 2)[0][0] for t in lines)
    panel_w = min(max_w, text_w) + 2 * pad
    panel_h = 30 * len(lines) + 56

    display = demo.draw_panel(display, pad, pad, panel_w, panel_h)
    y = pad + 28
    for i, line in enumerate(lines):
        colour = WHITE
        if i == 1 and captured >= target:
            colour = GREEN
        elif i == 3 and auto:
            colour = YELLOW
        cv2.putText(display, line, (pad + 12, y), FONT, scale, colour, 2)
        y += 30

    cv2.putText(display, "Nepali:", (pad + 12, y + 4), FONT, 0.5, WHITE, 1)
    display = demo.draw_devanagari(display, api.CHAR_MAP[class_id],
                                   (pad + 74, y - 18), WHITE)
    y += 34
    vary = f"vary: {hint}"
    cv2.putText(display, vary, (pad + 12, y), FONT,
                demo.fit_scale(vary, max_w, 0.42, 1, min_scale=0.32), GREY, 1)

    for warning in hints[:2]:
        y += 20
        cv2.putText(display, f"! {warning}", (pad + 12, y), FONT,
                    demo.fit_scale(warning, max_w, 0.42, 1, min_scale=0.32), RED, 1)
    return display, pad + panel_h


def draw_preview(display, crop, top_y, roi_left):
    """The un-mirrored crop, i.e. exactly the image being written to disk."""
    pad, side = 14, 150
    if roi_left - pad < side + 20 or display.shape[0] - top_y < side + 60:
        return display
    display = demo.draw_panel(display, pad, top_y + 10, side + 20, side + 36, alpha=0.55)
    preview = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
    display[top_y + 20:top_y + 20 + side, pad + 10:pad + 10 + side] = preview
    cv2.rectangle(display, (pad + 10, top_y + 20),
                  (pad + 10 + side, top_y + 20 + side), YELLOW, 1)
    cv2.putText(display, "saved crop (not mirrored)", (pad + 10, top_y + side + 38),
                FONT, 0.38, YELLOW, 1)
    return display


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Capture webcam images of one sign class for fine-tuning.")
    ap.add_argument("sign_class", help="roman label ('ka', 'Ta') or class id (0-35)")
    ap.add_argument("--target", type=int, default=TARGET_DEFAULT,
                    help=f"images wanted for this class (default {TARGET_DEFAULT})")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--auto-interval", type=float, default=AUTO_INTERVAL_DEFAULT,
                    help="seconds between automatic captures "
                         f"(default {AUTO_INTERVAL_DEFAULT})")
    ap.add_argument("--dataset-dir", default=str(DATASET_DIR),
                    help="where the class folders live "
                         "(default live_demo/webcam_dataset)")
    args = ap.parse_args()

    class_id, roman = resolve_class(args.sign_class)
    interval = max(AUTO_INTERVAL_MIN, args.auto_interval)
    class_dir = Path(args.dataset_dir) / class_dir_name(class_id)
    class_dir.mkdir(parents=True, exist_ok=True)

    captured = existing_count(class_dir)
    # Only the roman label is printed: a redirected Windows console uses the
    # locale codepage, which cannot encode Devanagari. The character itself is
    # drawn in the window, where Pillow renders it properly.
    print(f"Class {class_id} ({roman})")
    print(f"  folder   : {class_dir}")
    print(f"  already  : {captured} image(s)")
    print(f"  target   : {args.target}")
    print(f"  controls : SPACE capture   A auto ({interval:.2f}s)   Q quit")
    print("  saved un-mirrored, with the live demo's ROI, never horizontally flipped")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, demo.WINDOW_WIDTH, demo.WINDOW_HEIGHT)
    window_sized = False

    auto = False
    last_auto = 0.0
    flash_until = 0.0
    geometry_checked = False
    session_saved = 0
    box = None
    frame_shape = None

    while True:
        ok, raw = cap.read()
        if not ok:
            print("Dropped frame from the camera; stopping.")
            break

        h, w = raw.shape[:2]
        frame_shape = raw.shape
        box = demo.roi_box(w, h)
        x1, y1, x2, y2 = box
        crop = raw[y1:y2, x1:x2].copy()      # RAW, un-mirrored: what the CNN sees

        if not window_sized:
            scale = min(demo.WINDOW_WIDTH / w, demo.WINDOW_HEIGHT / h)
            cv2.resizeWindow(WINDOW_NAME, int(w * scale), int(h * scale))
            window_sized = True
        if not geometry_checked:
            check_geometry(class_dir, box)
            geometry_checked = True

        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()

        if key in (ord("a"), ord("A")):
            auto = not auto
            last_auto = now
            print(f"  auto capture {'ON' if auto else 'OFF'}")

        take = key == 32                                    # SPACE
        if auto and now - last_auto >= interval:
            take = True
            last_auto = now

        if take:
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = class_dir / f"{roman}_{stamp}.jpg"
            if cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                captured += 1
                session_saved += 1
                flash_until = now + 0.12
                write_meta(class_dir, class_id, roman, raw.shape, box, captured)
                if captured % 10 == 0 or captured == args.target:
                    print(f"  {captured}/{args.target} images")
            else:
                print(f"  could not write {path}")

        # --- display: mirrored, so the signer sees a natural picture ---
        display = cv2.flip(raw, 1)
        mx1, my1, mx2, my2 = demo.mirror_box(box, w)
        metrics = demo.hand_metrics(crop)
        cv2.rectangle(display, (mx1, my1), (mx2, my2),
                      WHITE if now < flash_until else GREEN,
                      4 if now < flash_until else 2)

        hint_index = captured * len(VARIATION_HINTS) // max(1, args.target)
        hint = VARIATION_HINTS[min(len(VARIATION_HINTS) - 1, hint_index)]
        display, bottom = draw_status(display, roman, class_id, captured, args.target,
                                      auto, interval, mx1, hint,
                                      demo.framing_hints(metrics))
        display = draw_preview(display, crop, bottom, mx1)
        cv2.putText(display, CONTROLS, (14, h - 14), FONT, 0.45, GREY, 1)
        cv2.imshow(WINDOW_NAME, display)

        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    if box is not None and frame_shape is not None:
        write_meta(class_dir, class_id, roman, frame_shape, box, captured)

    print(f"\nSaved {session_saved} image(s) this session; {captured} in total for "
          f"class {class_id} ({roman}).")
    print(f"  {class_dir}")
    if captured < 30:
        print("  30-50 images per class is the useful range; this class has fewer.")


if __name__ == "__main__":
    main()
