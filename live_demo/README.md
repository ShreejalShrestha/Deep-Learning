# Nepali Sign Language — live demo

A FastAPI service that serves the original Week 4 CNN, plus an OpenCV webcam
client that sends it crops from the laptop camera.

Nothing is trained here. The service only loads the existing checkpoint
`../models/low_lr.pt` (the "Low LR" experiment, the model selected in
`Shreejal.Shrestha.ipynb`). The architecture, the preprocessing and the 36
class IDs are copied from that notebook unchanged.

## Files

| File | What it does |
| --- | --- |
| `main.py` | FastAPI app: loads the CNN once, exposes `/`, `/health`, `/predict` |
| `webcam_demo.py` | OpenCV client: webcam → ROI crop → `/predict` → on-screen result |
| `requirements.txt` | Packages the demo needs |

## Setup (once)

From the project root, create a virtual environment, activate it, and install
the dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r live_demo/requirements.txt
```

`requirements.txt` is the single source of dependencies for the demo. Both
halves — the API and the webcam client — run from this one interpreter, so
activate the same environment in both terminals below.

## How to run

Open two terminals, activate the virtual environment in each, and `cd live_demo`.

**Terminal 1 — the API**

```
python -m uvicorn main:app --reload
```

Swagger documentation (useful for testing without the webcam):
<http://127.0.0.1:8000/docs>

**Terminal 2 — the webcam client**

```
python webcam_demo.py
```

The window opens at about 1280x720 and can be resized. Controls: `Q` quits,
`D` toggles debug mode, `SPACE` forces an immediate prediction.

## What happens, step by step

1. FastAPI loads `models/low_lr.pt` once, when the server starts.
2. OpenCV opens the laptop webcam.
3. You place your hand inside the green Region Of Interest box.
4. OpenCV crops just that box — not the whole frame.
5. The crop is taken from the **raw, un-mirrored** frame and POSTed to
   `/predict` as `multipart/form-data`, roughly every 3rd frame. Only a
   display copy is mirrored, so you see yourself naturally while the CNN sees
   the orientation it was trained on — mirrored input costs about 44 accuracy
   points, because the training data was never flipped.
6. FastAPI applies the exact original preprocessing: convert to RGB, resize to
   128×128 with bilinear interpolation, convert to a float32 CHW tensor, divide
   by 255, add the batch dimension, move it to the model's device. No
   augmentation and no ImageNet normalisation — the notebook used neither at
   inference time.
7. The CNN predicts one of the 36 classes under `torch.no_grad()`, and softmax
   turns the logits into probabilities.
8. FastAPI returns the predicted class ID, the roman transliteration, the
   Devanagari character, the confidence, and the top-3 predictions.
9. OpenCV displays the result live in a resizable window. Predictions are
   smoothed over the last 7 responses: a class is shown only if it is the
   majority, appears in at least 4 of those 7, and its average confidence clears
   `CONFIDENCE_THRESHOLD = 0.70`. Otherwise the display reads `Uncertain`.

## Why only the ROI is sent

The model was trained on 300×300 images that are almost entirely hand. Sending
a full camera frame would put the hand in a small corner of the input, which
does not resemble the training data. The fixed box keeps the live input close
to what the CNN saw during training.

## Settings you may want to change

Both near the top of `webcam_demo.py`:

- `CONFIDENCE_THRESHOLD` — raise it if the demo looks over-confident on
  nonsense, lower it if it rarely commits to a class.
- `ROI_FRACTION` — the ROI side as a fraction of the smaller frame dimension
  (default `0.65`); make it match how far you sit from the camera. The box
  stays square and centred, so the API's resize to 128×128 never distorts it.

`main.py` finds the checkpoint relative to its own file location, so both
commands work from any directory and no path needs editing.

## Running on the GPU (optional)

`main.py` already picks the device by itself:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

so **no code change is needed**. The only thing to change is which PyTorch
build is installed inside `.venv`. Everything below happens in that virtual
environment only — Anaconda and the system Python installation are not touched.

Checked on this machine: NVIDIA GeForce RTX 4060 Laptop GPU, driver 617.14
(CUDA 13.4), `.venv` on Python 3.13.7 with `torch 2.14.1+cpu`.

**1. Activate the virtual environment** (from the project root):

```
.venv\Scripts\activate
```

Every prompt after this should start with `(.venv)`. If it does not, the
commands below would hit the wrong Python.

**2. Remove the CPU-only build from this venv:**

```
python -m pip uninstall torch torchvision torchaudio -y
```

`torchaudio` is not currently installed and the demo does not need it; pip will
simply say so and move on.

**3. Install the CUDA build.** The official PyTorch command selects the build
through an index URL. CUDA 13.0 is the current stable default and matches this
driver:

```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

This resolves to `torch 2.14.1+cu130` and `torchvision 0.29.1+cu130` — the same
versions as the CPU build being replaced, so nothing else in the demo changes.
If that build ever gives trouble, `https://download.pytorch.org/whl/cu126` also
publishes matching wheels for this Python and GPU. The current command for any
other setup is generated at <https://pytorch.org/get-started/locally/>.

**4. Verify that PyTorch can see the GPU:**

```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output:

```
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

If it prints `False`, the CPU wheel is probably still installed — check that
`python -c "import torch; print(torch.__version__)"` ends in `+cu130` rather
than `+cpu`, and that the venv is activated.

**5. Start FastAPI as usual:**

```
cd live_demo
python -m uvicorn main:app --reload
```

The startup line prints the device it loaded the checkpoint on.

**6. Confirm the API is on the GPU** at <http://127.0.0.1:8000/health>, which
should now report:

```json
{"status": "ready", "model": "SignCNN", "checkpoint": "low_lr.pt",
 "device": "cuda", "classes": 36, "input_size": [128, 128]}
```

Restart the server after switching builds — the device is chosen once, when
the module is imported.

Note that `requirements.txt` still lists plain `torch`/`torchvision`, so a
*fresh* venv built from it gets the CPU build again; repeat steps 2–3 if you
rebuild the environment.

## Notes

- `torch` in `requirements.txt` installs the default build from PyPI, which on
  Windows is CPU-only. That is fine for this demo — one 128×128 image per
  request takes a few milliseconds — but see
  [Running on the GPU](#running-on-the-gpu-optional) to switch the venv to a
  CUDA build.
- Press `D` during the demo to see the exact crop being sent to the CNN, the
  raw top-3 with confidences, and whether smoothing currently calls the result
  stable. The preview is shown un-mirrored, so it can be compared directly
  with the dataset images.
- The Devanagari character is drawn with Pillow using a system font
  (`Nirmala.ttc` or `mangal.ttf` on Windows), because OpenCV's built-in fonts
  cannot render Devanagari. If no such font is found the demo still runs and
  shows the roman transliteration only.

## Example `/predict` response

```json
{
  "class_id": 1,
  "roman": "kha",
  "devanagari": "ख",
  "confidence": 0.9874,
  "top3": [
    {"class_id": 1, "roman": "kha", "devanagari": "ख", "confidence": 0.9874},
    {"class_id": 0, "roman": "ka",  "devanagari": "क", "confidence": 0.0081},
    {"class_id": 2, "roman": "ga",  "devanagari": "ग", "confidence": 0.0021}
  ]
}
```
