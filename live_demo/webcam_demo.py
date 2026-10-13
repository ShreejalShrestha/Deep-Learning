"""
OpenCV webcam client for the Nepali Sign Language demo.

Orientation note (the important part)
-------------------------------------
The training images show the hand as the camera saw it, un-mirrored: a right
hand looks like a right hand. The notebook also deliberately excluded
horizontal flipping from the augmentation, because mirroring a sign can turn it
into a different character. The trained model is therefore strongly orientation
sensitive - feeding it mirrored dataset images drops accuracy from 100% to 56%.

So this client keeps two versions of every frame:

    camera frame --+-- raw, un-mirrored --> ROI crop --> FastAPI --> CNN
                   +-- mirrored copy ------> on-screen display for the signer

The signer sees a natural mirror image, while the CNN sees the orientation it
was trained on. Sign with the hand that makes the ROI look like the dataset
images, which for this dataset is a right hand.

Start the API first (python -m uvicorn main:app --reload), then run this file.
"""

import datetime as _dt
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

API_URL = "http://127.0.0.1:8000/predict"
WINDOW_NAME = "Nepali Sign Language - live demo"

CONFIDENCE_THRESHOLD = 0.70   # below this the demo says "Uncertain"
PREDICT_EVERY = 3             # send 1 frame in 3, so the window stays responsive
SMOOTH_WINDOW = 7             # keep the last 7 predictions
MIN_AGREEMENT = 4             # ...and show a class only if 4 of them agree
REQUEST_TIMEOUT = 2.0         # seconds
STALE_AFTER = 1.5             # seconds without a reply before the history is dropped

# --- ROI placement -------------------------------------------------------
# ROI_FRACTION is the box side as a fraction of the smaller frame dimension.
# The offsets are fractions of the frame size, measured on the MIRRORED display
# (the picture you actually see): positive X moves the box to the right, positive
# Y moves it down. A right-handed signer's hand appears on the right of the
# mirrored image, so a small positive X offset puts the box where the hand
# already is. The raw-frame crop is derived from this, so the two always agree.
ROI_FRACTION = 0.40
ROI_X_OFFSET = 0.055
ROI_Y_OFFSET = 0.00

# --- matching the training-image statistics --------------------------------
# Measured on real captures vs the training set:
#   hand bbox area  39%  vs  ~75%   -> crop tighter (0.48 -> 0.40) and stand closer
#   hand centre x   0.55 vs  ~0.34  -> the hand must sit LEFT inside the raw crop,
#                                      which is the RIGHT of the mirrored display
# These targets drive the debug-mode framing hints only; they never change what
# the API does with the crop.
TARGET_HAND_FRACTION = (0.60, 0.80)   # share of the ROI the hand should fill
TARGET_CENTRE_X = (0.30, 0.48)        # hand centre inside the RAW crop

# Brightness: OFF by default, on purpose.
# The dataset's own mean brightness is 177 (p10-p90 124-233) and it varies
# enormously *by class*: cha=121, jha=122 at one end, Da=237, Na=235 at the
# other. The "~128" figure from the diagnostic was the mean of class chha (126)
# - the class that was wrongly predicted - not of the dataset. Normalising every
# crop to 130 therefore drags it towards the darkest classes: measured on 36
# training images it takes accuracy from 100% to 58%, and the new errors land on
# cha/ja/tra, the darkest classes of all. Target 177 scores 97%, off scores 100%.
# Turn this on only if your room is genuinely far from the dataset's lighting.
ENABLE_BRIGHTNESS_NORMALIZATION = False
TARGET_MEAN_BRIGHTNESS = 177           # the dataset's actual mean, not 130
BRIGHTNESS_GAIN_LIMITS = (0.65, 1.35)  # never push the image further than this
BRIGHTNESS_STRENGTH = 0.7              # 1.0 = go all the way to the target

WINDOW_WIDTH = 1280           # initial window size; the window stays resizable
WINDOW_HEIGHT = 720

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 215, 255)
RED = (0, 0, 255)
GREY = (190, 190, 190)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CONTROLS = "Q: Quit   D: Debug   SPACE: Predict   C: Capture"
CAPTURE_DIR = Path(__file__).resolve().parent / "debug_captures"

# OpenCV's built-in fonts have no Devanagari glyphs, so the Nepali character is
# drawn with Pillow using a system font that has them. If no such font is found
# the demo still runs and shows the roman transliteration only.
DEVANAGARI_FONT = None
for _candidate in ("C:/Windows/Fonts/Nirmala.ttc", "C:/Windows/Fonts/mangal.ttf"):
    if Path(_candidate).exists():
        DEVANAGARI_FONT = ImageFont.truetype(_candidate, 34)
        break


def draw_devanagari(frame, text, position, colour):
    """Draw Devanagari text onto a BGR OpenCV frame using Pillow."""
    if DEVANAGARI_FONT is None or not text:
        return frame
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(position, text, font=DEVANAGARI_FONT,
                               fill=colour[::-1])  # BGR tuple -> RGB
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def roi_box(frame_w, frame_h, fraction=ROI_FRACTION,
            x_offset=ROI_X_OFFSET, y_offset=ROI_Y_OFFSET):
    """The classifier ROI in RAW (un-mirrored) frame coordinates.

    Square on purpose: the API resizes it to 128x128, and a non-square crop
    would stretch the hand in a way the training images never did. The offsets
    are given in display coordinates, so the box is positioned where the signer
    sees it and then converted back to the raw frame (display x = width - raw x).
    The result is always clamped inside the frame.
    """
    size = int(min(frame_w, frame_h) * fraction)
    size = max(32, min(size, frame_w, frame_h))

    centre_x_display = frame_w / 2 + x_offset * frame_w
    centre_x_raw = frame_w - centre_x_display      # undo the horizontal mirror
    centre_y = frame_h / 2 + y_offset * frame_h

    x1 = int(round(centre_x_raw - size / 2))
    y1 = int(round(centre_y - size / 2))
    x1 = max(0, min(x1, frame_w - size))           # keep the box inside the frame
    y1 = max(0, min(y1, frame_h - size))
    return x1, y1, x1 + size, y1 + size


def mirror_box(box, frame_w):
    """The same ROI expressed in the coordinates of the horizontally flipped frame.

    Flipping maps column x to (frame_w - 1 - x), so the left and right edges
    swap. This is what gets drawn, so the rectangle on screen always frames
    exactly the pixels that were sent to the CNN.
    """
    x1, y1, x2, y2 = box
    return frame_w - x2, y1, frame_w - x1, y2


def normalise_brightness(roi):
    """Scale the ROI gently towards TARGET_MEAN_BRIGHTNESS.

    A single multiplicative gain, partially applied and clamped, so it is
    deterministic and cannot wreck the picture. Contrast is left alone: no
    histogram equalisation, no per-channel stretching. Returns (roi, info).
    """
    mean = float(roi.mean())
    info = {"enabled": ENABLE_BRIGHTNESS_NORMALIZATION,
            "mean_before": round(mean, 1), "gain": 1.0,
            "mean_after": round(mean, 1)}
    if not ENABLE_BRIGHTNESS_NORMALIZATION or mean < 1.0:
        return roi, info

    wanted = TARGET_MEAN_BRIGHTNESS / mean
    gain = 1.0 + (wanted - 1.0) * BRIGHTNESS_STRENGTH   # only part of the way
    gain = float(np.clip(gain, *BRIGHTNESS_GAIN_LIMITS))
    if abs(gain - 1.0) < 0.02:                          # already close enough
        return roi, info

    out = np.clip(roi.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    info["gain"] = round(gain, 3)
    info["mean_after"] = round(float(out.mean()), 1)
    return out, info


def hand_metrics(roi):
    """Rough skin-based framing stats for the crop, matching the diagnostic's rule."""
    a = roi.astype(np.int16)
    b, g, r = a[:, :, 0], a[:, :, 1], a[:, :, 2]        # OpenCV is BGR
    mx, mn = a.max(2), a.min(2)
    mask = ((r > 95) & (g > 40) & (b > 20) & ((mx - mn) > 15) &
            (np.abs(r - g) > 15) & (r > g) & (r > b))
    h, w = mask.shape
    out = {"hand_fraction": float(mask.mean()), "brightness": float(roi.mean()),
           "centre_x": None, "bbox_fraction": None, "edges": []}
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return out
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    out["centre_x"] = float(xs.mean() / w)
    out["bbox_fraction"] = float(((x2 - x1 + 1) * (y2 - y1 + 1)) / (w * h))
    k = max(2, h // 64)
    out["edges"] = [n for n, hit in (("L", mask[:, :k].mean() > 0.02),
                                     ("R", mask[:, -k:].mean() > 0.02),
                                     ("T", mask[:k, :].mean() > 0.02),
                                     ("B", mask[-k:, :].mean() > 0.02)) if hit]
    return out


def framing_hints(metrics):
    """Short warnings, only when a measurement is actually out of range.

    Left/right is phrased as the signer sees it on the mirrored display, which
    is the opposite of the raw crop the CNN receives.
    """
    hints = []
    bbox = metrics.get("bbox_fraction")
    if bbox is not None and bbox < TARGET_HAND_FRACTION[0]:
        hints.append("Move hand closer")
    cx = metrics.get("centre_x")
    if cx is not None:
        # cx is measured in the RAW crop; the display is mirrored, so a hand too
        # far right in the crop has to move right on screen to get left in the crop.
        if cx > TARGET_CENTRE_X[1]:
            hints.append("Move hand right (on screen)")
        elif cx < TARGET_CENTRE_X[0]:
            hints.append("Move hand left (on screen)")
    if metrics.get("brightness", 0) > TARGET_MEAN_BRIGHTNESS + 45:
        hints.append("Too bright")
    if metrics.get("edges"):
        hints.append("Hand touches edge: " + "".join(metrics["edges"]))
    return hints


def classify(roi):
    """JPEG-encode the ROI in memory and POST it as multipart/form-data."""
    ok, buf = cv2.imencode(".jpg", roi)
    if not ok:
        return None, "JPEG encoding failed"
    files = {"file": ("roi.jpg", buf.tobytes(), "image/jpeg")}
    try:
        response = requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, f"API error: {type(exc).__name__}"


def smooth(history):
    """Majority vote over the recent predictions.

    Returns (label, devanagari, confidence, is_stable). A class is reported
    only when it wins the vote, appears at least MIN_AGREEMENT times in the
    window, and its average confidence clears CONFIDENCE_THRESHOLD.
    """
    if not history:
        return "...", "", 0.0, False
    majority, votes = Counter(r for r, _, _ in history).most_common(1)[0]
    matching = [(d, c) for r, d, c in history if r == majority]
    devanagari = matching[0][0]
    confidence = sum(c for _, c in matching) / len(matching)
    if votes >= MIN_AGREEMENT and confidence >= CONFIDENCE_THRESHOLD:
        return majority, devanagari, confidence, True
    return "Uncertain", "", confidence, False


def fit_scale(text, max_width, base_scale, thickness=2, min_scale=0.4):
    """Largest font scale <= base_scale that keeps `text` within max_width."""
    width = cv2.getTextSize(text, FONT, base_scale, thickness)[0][0]
    if width <= max_width or width == 0:
        return base_scale
    return max(min_scale, base_scale * max_width / width)


def draw_panel(frame, x, y, w, h, alpha=0.45):
    """Darken a rectangle so overlay text stays readable over any background."""
    x, y = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x2 <= x or y2 <= y:
        return frame
    region = frame[y:y2, x:x2]
    frame[y:y2, x:x2] = cv2.addWeighted(region, 1 - alpha,
                                        np.zeros_like(region), alpha, 0)
    return frame


def draw_readout(display, label, nepali, confidence, stable, roi_left):
    """Prediction / Nepali / Confidence, in the top-left corner.

    The panel is kept clear of `roi_left` (the left edge of the ROI on the
    mirrored display) so the text never covers the hand.
    """
    pad = 14
    max_w = max(150, roi_left - 2 * pad - 10)
    lines = [f"Prediction: {label}", f"Confidence: {confidence * 100:.1f}%"]
    scale = min(fit_scale(lines[0], max_w, 0.72), fit_scale(lines[1], max_w, 0.72))
    text_w = max(cv2.getTextSize(t, FONT, scale, 2)[0][0] for t in lines)
    panel_w = min(max_w, text_w) + 2 * pad
    panel_h = 118

    display = draw_panel(display, pad, pad, panel_w, panel_h)
    colour = WHITE if stable else YELLOW
    cv2.putText(display, lines[0], (pad + 12, pad + 30), FONT, scale, colour, 2)
    cv2.putText(display, "Nepali:", (pad + 12, pad + 66), FONT, 0.6, colour, 1)
    display = draw_devanagari(display, nepali if nepali else "-",
                              (pad + 86, pad + 42), colour)
    cv2.putText(display, lines[1], (pad + 12, pad + 100), FONT, scale * 0.85, colour, 2)
    return display, pad + panel_h


def draw_debug(display, roi, box, top3, history, stable, stable_count, top_y,
               roi_left, metrics=None, brightness=None):
    """Debug overlay on the left: the exact crop sent, plus the raw numbers.

    Kept on the left-hand side because the ROI sits to the right, so nothing
    here covers the hand. The panel is sized to the space actually available
    between the readout above, the ROI to the right and the controls line
    below; if that leaves no room for the crop preview, the numbers are shown
    on their own.
    """
    h, w = display.shape[:2]
    pad = 14
    x1, y1, x2, y2 = box
    y0 = top_y + 12

    metrics = metrics or {}
    bbox = metrics.get("bbox_fraction")
    cx = metrics.get("centre_x")
    votes = Counter(r for r, _, _ in history)
    top_votes = votes.most_common(1)[0][1] if votes else 0

    measured = []
    measured.append(f"fill {bbox * 100:.0f}%" if bbox is not None else "fill n/a")
    measured.append(f"centre x {cx:.2f}" if cx is not None else "centre x n/a")

    # Most useful first: whatever does not fit is dropped from the end.
    lines = [
        f"Hand size target: fill ~{TARGET_HAND_FRACTION[0] * 100:.0f}-"
        f"{TARGET_HAND_FRACTION[1] * 100:.0f}% of ROI",
        f"Position target: left of centre "
        f"{TARGET_CENTRE_X[0]:.2f}-{TARGET_CENTRE_X[1]:.2f}",
        "measured: " + "   ".join(measured),
    ]
    if brightness:
        lines.append(f"brightness {brightness.get('mean_before', 0):.0f} ->"
                     f" {brightness.get('mean_after', 0):.0f}"
                     f"  gain x{brightness.get('gain', 1.0):.2f}"
                     f"{'' if brightness.get('enabled') else '  (OFF)'}")
    lines += [f"! {hint}" for hint in framing_hints(metrics)]
    lines.append("top 3 (last API reply):")
    for i, item in enumerate(top3):
        lines.append(f"  {i + 1}. {item['roman']:<5} id {item['class_id']:>2}"
                     f"  {item['confidence'] * 100:6.2f}%")
    if not top3:
        lines.append("  (no reply yet)")
    lines.append(f"votes {top_votes}/{len(history)} (need {MIN_AGREEMENT})  "
                 f"{'STABLE' if stable else 'UNCERTAIN'}")
    lines.append(f"ROI {x2 - x1}x{y2 - y1}px  x{x1}-{x2} y{y1}-{y2}")
    lines.append(f"stable predictions: {stable_count}")

    # Width: never reach the ROI. Shrink the text until it fits that budget.
    avail_w = roi_left - pad - 12
    avail_h = (h - 34) - y0
    line_h = 17
    if avail_w < 120 or avail_h < 6 * line_h:
        # Too small a frame to show anything useful without covering the hand.
        msg = "debug: no room"
        cv2.putText(display, msg, (pad, y0 + 14), FONT,
                    fit_scale(msg, max(40, avail_w), 0.4, 1), YELLOW, 1)
        return display

    scale = 0.44
    for line in lines:
        scale = min(scale, fit_scale(line, avail_w - 20, 0.44, 1, min_scale=0.32))

    # Give the crop preview whatever is left once the key lines are placed,
    # then drop any trailing lines that still do not fit.
    side = min(170, avail_w - 20, max(0, avail_h - 9 * line_h - 26))
    if side < 60:
        side = 0
    max_lines = max(4, (avail_h - (side + 14 if side else 0) - 18) // line_h)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + ["..."]

    text_w = max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t in lines)
    panel_w = min(avail_w, max(side, text_w) + 20)
    panel_h = (side + 14 if side else 0) + line_h * len(lines) + 16
    display = draw_panel(display, pad, y0, panel_w, panel_h, alpha=0.55)

    px, py = pad + 10, y0 + 10
    if side:
        # The preview is deliberately NOT mirrored: it is exactly what the CNN
        # sees, so it can be compared against the dataset images side by side.
        preview = cv2.resize(roi, (side, side), interpolation=cv2.INTER_AREA)
        display[py:py + side, px:px + side] = preview
        cv2.rectangle(display, (px, py), (px + side, py + side), YELLOW, 1)

    ty = py + (side + 20 if side else 8)
    for i, line in enumerate(lines):
        colour = YELLOW
        if line.startswith("smoothing:"):
            colour = GREEN if stable else RED
        elif line.startswith("!"):
            colour = RED
        cv2.putText(display, line, (px, ty), FONT, scale, colour, 1)
        ty += line_h
    return display


def save_capture(roi, box, result, label, confidence, stable,
                 brightness=None, metrics=None):
    """Write the exact crop that was sent to the API, plus what it predicted.

    The image saved is the un-mirrored ROI after brightness normalisation but
    *before* the API's resize to 128x128 - i.e. byte-for-byte the classifier
    input - so a diagnostic run later sees exactly what the model saw. The gain
    that was applied is recorded in the JSON beside it.
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    img_path = CAPTURE_DIR / f"capture_{stamp}.jpg"
    meta_path = CAPTURE_DIR / f"capture_{stamp}.json"
    x1, y1, x2, y2 = box

    if not cv2.imwrite(str(img_path), roi):
        print(f"  could not write {img_path}")
        return None

    meta = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "image": img_path.name,
        "roi": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1,
                "fraction": ROI_FRACTION,
                "x_offset": ROI_X_OFFSET, "y_offset": ROI_Y_OFFSET},
        "mirrored": False,
        "brightness_normalisation": brightness or {"enabled": False, "gain": 1.0},
        "framing": metrics or {},
        "smoothed": {"label": label, "confidence": round(confidence, 4),
                     "stable": stable},
        "last_api_reply": result,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    shown = result.get("roman") if result else "no reply"
    print(f"  captured {img_path.name}  ({x2 - x1}x{y2 - y1} px, API said {shown})")
    return img_path


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam (device 0).")
        return

    # A normal, manually resizable window. WINDOW_KEEPRATIO letterboxes the
    # frame instead of stretching it, so the camera's own aspect ratio survives
    # both the initial size and any later resize.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    window_sized = False

    history = deque(maxlen=SMOOTH_WINDOW)   # (roman, devanagari, confidence)
    frame_count = 0
    error_text = None
    debug = False
    last_top3 = []
    last_result = None        # the full last API reply, saved with a capture
    last_roi = None           # the normalised crop, i.e. exactly what was POSTed
    last_metrics = {}         # framing stats of that crop, measured before the gain
    last_bright = {"enabled": ENABLE_BRIGHTNESS_NORMALIZATION, "gain": 1.0}
    stable_count = 0          # how many times a new sign settled into "stable"
    last_stable_label = None
    last_success = None       # time.monotonic() of the last good reply
    api_failing = False

    print("Webcam demo running.  Q = quit   D = debug   SPACE = predict now   C = capture ROI")

    while True:
        ok, raw = cap.read()
        if not ok:
            print("Dropped frame from the camera; stopping.")
            break

        h, w = raw.shape[:2]
        box = roi_box(w, h)
        x1, y1, x2, y2 = box

        # Once the real camera resolution is known, shrink the window to that
        # aspect ratio inside the 1280x720 box (e.g. 960x720 for a 4:3 camera),
        # so the picture is never stretched to fill a 16:9 window.
        if not window_sized:
            scale = min(WINDOW_WIDTH / w, WINDOW_HEIGHT / h)
            cv2.resizeWindow(WINDOW_NAME, int(w * scale), int(h * scale))
            window_sized = True

        key = cv2.waitKey(1) & 0xFF
        frame_count += 1

        if key in (ord("d"), ord("D")):
            debug = not debug
            history.clear()        # the view changed; start the vote fresh

        # --- classifier path: crop the RAW, un-mirrored frame ---
        if frame_count % PREDICT_EVERY == 0 or key == ord(" "):
            crop = raw[y1:y2, x1:x2].copy()
            last_metrics = hand_metrics(crop)          # measured before any gain
            roi, last_bright = normalise_brightness(crop)
            last_roi = roi                             # exactly what is POSTed
            result, error_text = classify(roi)
            if result is None:
                api_failing = True
            else:
                if api_failing:
                    history.clear()   # reconnected: drop everything from before
                    api_failing = False
                history.append((result["roman"], result["devanagari"],
                                result["confidence"]))
                last_top3 = result["top3"]
                last_result = result
                last_success = time.monotonic()

        # A long gap without a successful reply makes the window stale.
        if last_success is not None and time.monotonic() - last_success > STALE_AFTER:
            history.clear()
            last_success = None

        label, nepali, confidence, stable = smooth(history)

        # C: save the exact crop the API just saw, for offline diagnosis.
        if key in (ord("c"), ord("C")):
            if last_roi is None:
                print("  nothing to capture yet - wait for the first prediction")
            else:
                save_capture(last_roi, box, last_result, label, confidence,
                             stable, last_bright, last_metrics)

        if stable:
            if label != last_stable_label:
                stable_count += 1
                last_stable_label = label
        else:
            last_stable_label = None

        # --- display path: mirrored copy, so the signer sees themselves ---
        display = cv2.flip(raw, 1)
        mx1, my1, mx2, my2 = mirror_box(box, w)

        cv2.rectangle(display, (mx1, my1), (mx2, my2), GREEN, 2)
        display, panel_bottom = draw_readout(display, label, nepali, confidence,
                                             stable, mx1)

        if debug and last_roi is not None:
            display = draw_debug(display, last_roi, box, last_top3, history,
                                 stable, stable_count, panel_bottom, mx1,
                                 last_metrics, last_bright)

        cv2.putText(display, CONTROLS, (14, h - 14), FONT, 0.45, GREY, 1)
        if error_text:
            cv2.putText(display, error_text[:48], (14, h - 36), FONT, 0.45, RED, 1)

        cv2.imshow(WINDOW_NAME, display)

        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
