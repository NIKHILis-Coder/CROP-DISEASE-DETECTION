# Crop Disease Detection

A deep-learning web app that identifies plant diseases from a photo of a leaf.

Upload a leaf image and the app returns the predicted disease along with a
confidence score. The model is a small **Convolutional Neural Network (CNN)**
built from scratch with TensorFlow/Keras and trained on the
[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
dataset from Kaggle. It is served through a lightweight **Flask** web app.

> **Status:** Phase 2 of 7 complete — dataset acquired and explored. Later
> phases add preprocessing, the trained model, evaluation and the web app.
> Sections marked *Coming soon* below get filled in as those phases land.

### Why this project?

Crop disease is a major cause of yield loss, and early identification usually
depends on an expert being physically present. A phone camera plus an image
classifier makes a first-pass diagnosis available to anyone. This repository is
a compact, end-to-end demonstration of that idea: **raw data → preprocessing →
model → evaluation → deployed app.**

---

## Setup

### Prerequisites

- **Python 3.10–3.12.** TensorFlow does not publish builds for Python 3.13+ yet,
  so a newer Python will fail at `pip install`.
- A [Kaggle account](https://www.kaggle.com/) for downloading the dataset
  (needed from Phase 2 onward).

### 1. Clone the repository

```bash
git clone https://github.com/NIKHILis-Coder/CROP-DISEASE-DETECTION.git
cd CROP-DISEASE-DETECTION
```

### 2. Create and activate a virtual environment

A virtual environment keeps this project's libraries separate from every other
Python project on your machine, so versions can't clash.

```bash
# Windows (PowerShell) -- the "py -3.12" part picks a TensorFlow-compatible Python
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3.12 -m venv venv
source venv/bin/activate
```

Your prompt should now start with `(venv)`.

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle API access *(needed from Phase 2)*

1. On Kaggle: **Account → Settings → API → Create New Token.** This downloads a
   `kaggle.json` file containing your username and API key.
2. Move it to the location the Kaggle library looks in:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - macOS / Linux: `~/.kaggle/kaggle.json` (then run `chmod 600 ~/.kaggle/kaggle.json`)

`kaggle.json` is a **secret** — it is listed in `.gitignore` and must never be
committed.

---

## Project structure

```
CROP-DISEASE-DETECTION/
│
├── data/                  # All dataset files (contents ignored by Git)
│   ├── raw/               # Untouched PlantVillage images as downloaded
│   └── processed/         # Resized / split images ready for training
│
├── notebooks/             # Jupyter notebooks for exploration and experiments
│   ├── 01_eda.ipynb       # Exploratory data analysis of the tomato subset
│   └── figures/           # Plots saved by the notebooks (embedded in this README)
│
├── src/                   # Reusable Python code (download, preprocess, train, evaluate)
│   └── download_data.py   # Fetches PlantVillage and extracts the tomato classes
│
├── models/                # Saved trained models (.keras / .h5 — ignored by Git)
│
├── app/                   # The Flask web application
│   ├── templates/         # HTML pages rendered by Flask
│   └── static/            # CSS, images, and user-uploaded files
│
├── requirements.txt       # Python dependencies
├── .gitignore             # Files Git should not track
├── INTERVIEW_PREP.md      # Design decisions + Q&A, written up phase by phase
└── README.md              # This file
```

**Why this layout?** Data, code, models and the app each live in their own
folder, so any one of them can be changed without touching the others. It is the
conventional structure for a small ML project (a trimmed-down version of the
widely used *Cookiecutter Data Science* template), which means another developer
can find their way around it immediately.

`data/raw` is kept strictly read-only: every transformation writes into
`data/processed` instead. That way the original download never has to be
repeated, and any preprocessing bug can be traced back to a pristine source.

---

## Dataset

**Source:** the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
on Kaggle — 54,305 lab photographs of healthy and diseased leaves covering 38
classes across 14 crops, released for public research use.

**Scope used here:** the **tomato subset only** — **18,160 colour images across
10 classes** (nine diseases plus healthy). Download and extraction are scripted
in [`src/download_data.py`](src/download_data.py); the full exploration lives in
[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

| Class | Images | Share |
|-------|-------:|------:|
| Tomato Yellow Leaf Curl Virus | 5,357 | 29.5% |
| Bacterial spot | 2,127 | 11.7% |
| Late blight | 1,909 | 10.5% |
| Septoria leaf spot | 1,771 | 9.8% |
| Spider mites (two-spotted) | 1,676 | 9.2% |
| Healthy | 1,591 | 8.8% |
| Target spot | 1,404 | 7.7% |
| Early blight | 1,000 | 5.5% |
| Leaf mold | 952 | 5.2% |
| Tomato mosaic virus | 373 | 2.1% |
| **Total** | **18,160** | **100%** |

![Class distribution](notebooks/figures/class_distribution.png)

### Why subset to tomato?

**Training time.** The model here is a small CNN trained **from scratch on a
CPU** — no pre-trained weights, no GPU. Cutting the data from 54k images to 18k
cuts every training epoch to roughly a third, which is the difference between
iterating on the architecture several times an evening and waiting overnight per
run. Fast iteration matters more than raw scale for a project whose point is to
demonstrate the pipeline end to end.

**A sharper, more honest problem.** All 10 classes are the same crop, so the
model must separate *diseases of tomato* rather than the much easier task of
telling an apple leaf from a grape leaf. Distinguishing early blight from late
blight is genuinely hard and visually subtle; distinguishing corn from tomato is
not. A high score on the tomato subset therefore means considerably more than
the same score across all 38 classes.

**Tomato is the natural choice** for the subset: it is the single best-
represented crop in PlantVillage, with the widest range of diseases, so it gives
10 classes without any of them being uselessly small.

### What the data looks like

![Sample images](notebooks/figures/sample_images.png)

**Key findings from the EDA:**

- **Perfectly uniform format.** Every one of the 18,160 images is **256×256 RGB
  JPEG** — 100% square, a single aspect ratio. Resizing for the CNN is therefore
  a clean downscale with no distortion or cropping decisions.
- **Zero corrupt files.** All 18,160 passed a `PIL.Image.verify()` integrity
  check, so the data loader needs no error handling for unreadable images.
- **Moderate class imbalance — 14.4×** between the largest class (Yellow Leaf
  Curl Virus, 5,357) and the smallest (Tomato mosaic virus, 373). This sets the
  **majority-class baseline at 29.5%**: a model that ignores the image entirely
  and always guesses the biggest class would score that. Any reported accuracy
  has to be read against that number.
- **Consequences for later phases:** the train/val/test split must be
  **stratified** so rare classes appear in every split, and evaluation must
  report a **confusion matrix and per-class recall**, not accuracy alone.

**Known limitation:** PlantVillage images are laboratory photographs of single
detached leaves on plain backgrounds. A model trained on them will not
automatically work on a phone photo taken in a field, where soil, sky and
overlapping foliage fill the frame. This *domain gap* is the honest caveat about
the finished app.

### Reproducing the dataset locally

```bash
python src/download_data.py               # download + extract the tomato subset
python src/download_data.py --delete-zip  # ...and remove the 2 GB archive after
```

## Model

*Coming soon (Phase 4).* Will cover: the CNN architecture layer by layer, why
each layer is there, parameter count, and training configuration.

## Results

*Coming soon (Phase 5).* Will cover: test accuracy, confusion matrix,
per-class precision/recall, and examples the model gets wrong.

## Usage

*Coming soon (Phase 6).* Will cover: how to run the Flask app locally and how
to classify your own leaf photo.

---

## Roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Project structure, dependencies, README | ✅ Done |
| 2 | Dataset download + exploratory data analysis | ✅ Done |
| 3 | Preprocessing and augmentation pipeline | ⬜ Not started |
| 4 | Build and train the CNN | ⬜ Not started |
| 5 | Evaluation: accuracy, confusion matrix, error analysis | ⬜ Not started |
| 6 | Flask web app | ⬜ Not started |
| 7 | Final README polish and screenshots | ⬜ Not started |

## Tech stack

| Tool | Role |
|------|------|
| TensorFlow / Keras | Define and train the CNN |
| NumPy / pandas | Numerical arrays and tabular summaries |
| matplotlib / seaborn | Plots and the confusion-matrix heatmap |
| scikit-learn | Evaluation metrics |
| Pillow | Image loading and resizing |
| Flask | Web app serving the predictions |
| Kaggle API | Scripted dataset download |
