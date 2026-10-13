# Nepali Sign Language Recognition

University Neural Networks & Deep Learning coursework: classifying static
hand signs of the Nepali Sign Language (NSL) alphabet — 36 classes, one per
Devanagari consonant — from photographs, using exploratory data analysis, a
CNN, and an RNN/LSTM comparison, plus a live webcam demo.

## Dataset

[*Nepali Sign Language Character Dataset*](https://www.kaggle.com/datasets/biratpoudelrocks/nepali-sign-language-character-dataset)
(Kaggle), documented in Poudel, Ghimire, Bhattarai & Bhandari,
*KEC Journal of Science and Engineering* ([arXiv:2510.11243](https://arxiv.org/abs/2510.11243)).

- 36 classes (Devanagari consonants), 1,500 images each, 54,000 images total — perfectly balanced.
- Two capture conditions per class: `Plain Background/` (1,000 img/class) and `Random Background/` (500 img/class).
- All images 300×300 RGB JPEG; none corrupt.
- ~1.2% of images are exact-duplicate burst frames within the same class.
- No signer IDs are recorded, so signer-independent evaluation isn't possible with this dataset — noted explicitly as a limitation rather than assumed.
- Not committed to this repo. Expected locally at `archive/NSL/NSL/` (folders `0`–`35`, one per class).

The data is downloaded separately and is not included in version control
(see `.gitignore`).

## Approach

1. **EDA (Week 2).** Dataset structure, class balance, image-quality checks
   (corruption, duplicates), per-class and per-background pixel-intensity
   analysis, and a leakage-aware train/validation/test split (MD5 de-dup,
   stratified, 70/15/15, seed 0 → 37,330 / 7,999 / 8,000 images).
2. **CNN (Week 4).** A small convolutional classifier trained over several
   learning-rate configurations on the fixed split; the best-validation
   checkpoint ("Low LR") is selected and never retrained afterward.
3. **Final project.** The same split and preprocessing are reused to train
   an RNN and an LSTM on a 128-timestep × 384-feature row-sequence view of
   each image, for a direct three-way comparison against the CNN: accuracy,
   generalisation-gap analysis, a near-duplicate/leakage investigation, and
   error analysis.

## Results (test set, 8,000 images)

| Model | Accuracy | Precision | Recall | F1 | Params |
| --- | --- | --- | --- | --- | --- |
| RNN | 0.398 | 0.385 | 0.401 | 0.355 | 70,436 |
| LSTM | 0.876 | 0.885 | 0.877 | 0.875 | 267,812 |
| **CNN** (selected) | **0.9998** | **0.9998** | **0.9998** | **0.9998** | 1,249,220 |

Ranking: CNN > LSTM > RNN. None of the three models show classical
overfitting (train/val gaps are flat or negative); the RNN instead
underfits, with validation performance degrading late in training —
caught by early stopping.

**Caveat on the near-perfect CNN accuracy.** 36% of the test set has a
confirmed near-duplicate (dHash + high pixel correlation) in the training
set. The split itself is unbiased — this redundancy is spread evenly across
train and test, not concentrated on one side — but the dataset is
pervasively redundant at the image level (burst-captured frames of the same
hand pose). This inflates absolute accuracy for *all three* models and is
a dataset-density property, not data leakage or overfitting. A
signer/session-disjoint split would be needed to remove it, which this
dataset's lack of signer IDs makes impossible.

## Repository layout

```
Shreejal.Shrestha.ipynb   the graded submission — staged Week 2 / Week 4 / Final Project sections
Shreejal.ipynb            the same project as one continuous, re-runnable notebook
TECH_405_Week5_RNN_...    a separate, unrelated assignment (RNN on sklearn digits)
live_demo/                live webcam demo (FastAPI + OpenCV) — see live_demo/README.md
archive/                  dataset (gitignored, not committed)
models/                   saved checkpoints, e.g. models/low_lr.pt (gitignored)
figures/                  notebook-generated plots (gitignored)
```

## Environment

The notebooks and `live_demo/` run from a user-managed Python virtual
environment (`.venv`) — not Anaconda, not the system Python install.

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow
```

`live_demo/requirements.txt` is the dependency list for the demo
specifically (see below).

## Live demo

`live_demo/` is a deployment track kept strictly separate from the graded
result: a FastAPI service serves the Week 4 CNN checkpoint
(`models/low_lr.pt`) live from a webcam feed via an OpenCV client.
`low_lr.pt` is never overwritten; any webcam fine-tuning writes to
`models/live_finetuned.pt` instead, so the graded notebook result can't be
affected by the demo. Full setup and usage: [`live_demo/README.md`](live_demo/README.md).

## Limitations

- No signer IDs in the dataset — can't test signer-independent generalisation.
- Significant train/test image-level redundancy (near-duplicate burst frames) inflates reported accuracy for all models; see Results above.
- The RNN/LSTM treat each image as a flattened row-sequence rather than a true temporal signal — there is no video/time dimension in the source data, so this is a fixed-length sequence-model comparison, not a sign-*language* (motion) recognition task.
- The live demo faces webcam/new-signer domain shift relative to the training distribution, addressed by separate fine-tuning rather than by changing the graded model.

## References

- Poudel, B., Ghimire, S., Bhattarai, P., Bhandari, S. (2025/2026). *Nepali Sign Language Character Dataset*. KEC Journal of Science and Engineering. [arXiv:2510.11243](https://arxiv.org/abs/2510.11243)
- Dataset: [kaggle.com/datasets/biratpoudelrocks/nepali-sign-language-character-dataset](https://www.kaggle.com/datasets/biratpoudelrocks/nepali-sign-language-character-dataset)
