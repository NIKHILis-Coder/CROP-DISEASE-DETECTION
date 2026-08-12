# Crop Disease Detection

A deep-learning web app that identifies plant diseases from a photo of a leaf.

Upload a leaf image and the app returns the predicted disease along with a
confidence score. The model is a small **Convolutional Neural Network (CNN)**
built from scratch with TensorFlow/Keras and trained on the
[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
dataset from Kaggle. It is served through a lightweight **Flask** web app.

> **Status:** Phase 1 of 7 complete — project scaffolding. Later phases add the
> dataset, preprocessing, the trained model, evaluation and the web app.
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
│
├── src/                   # Reusable Python code (download, preprocess, train, evaluate)
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

*Coming soon (Phase 2).* Will cover: source and licence, number of images,
number of classes, class distribution, and image size statistics.

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
| 2 | Dataset download + exploratory data analysis | ⬜ Not started |
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
