# Interview Prep

A running log of **what was built, why it was built that way, and the questions
an interviewer is likely to ask** about it. One section per phase, appended as
each phase is completed.

---

## Phase 1 — Project structure

### What was built

The empty skeleton of the project — no data, no model, no app code yet, just
the scaffolding everything else drops into:

- **Folder tree:** `data/raw`, `data/processed`, `notebooks`, `src`, `models`,
  `app/templates`, `app/static`.
- **`requirements.txt`** listing every dependency with a comment explaining why
  it is there.
- **`.gitignore`** excluding datasets, trained models, `__pycache__/`, the
  virtual environment and secrets — while keeping the empty folders visible via
  `.gitkeep` placeholder files.
- **`README.md`** with the project description, setup instructions, the folder
  layout, and placeholder sections for Dataset / Model / Results / Usage.

### Why — design decisions

**Why separate `data/raw` from `data/processed`?**
`raw` is treated as read-only: it is exactly what was downloaded from Kaggle and
never modified. Every transformation writes to `processed`. This means (a) the
slow download never has to be repeated, and (b) if a preprocessing step turns
out to be buggy, the pipeline can be re-run from a known-good source instead of
from data that has already been silently corrupted.

**Why keep data and models out of Git?**
Git stores a complete new copy of a file every time it changes, and it is built
for text (where it can store just the differences), not for large binaries.
PlantVillage is over a gigabyte, and a saved Keras model is tens of megabytes.
Committing them would make the repo slow to clone, and GitHub hard-rejects any
file over 100 MB. Both are *reproducible* — the dataset from Kaggle and the
model from the training script — so committing the code that produces them is
enough.

**Why `.gitkeep` files?**
Git tracks *files*, not *folders*, so an empty folder simply does not exist as
far as Git is concerned. Someone cloning the repo would get no `data/raw` at
all, and the download script would crash. A zero-byte `.gitkeep` file gives Git
something to track so the folder survives the clone. (The name is a convention,
not a Git feature — any filename would work.)

**Why `>=` version ranges instead of exact `==` pins?**
For a portfolio project that other people will clone onto unknown machines,
minimum versions keep installation from breaking over trivial version skew.
Production systems do the opposite and pin exactly, so that a deploy today
installs precisely what was tested last week. The trade-off is *reproducibility
vs. flexibility*, and the right answer depends on which one costs more when it
goes wrong.

**Why Flask rather than Django or FastAPI?**
The app is two routes: show an upload form, and return a prediction. Django
brings an ORM, a migration system and an admin panel that would all sit unused.
FastAPI would be the better pick for a pure JSON API, but this app renders an
HTML page, which is Flask's natural home.

**Why write `INTERVIEW_PREP.md` as the project goes rather than at the end?**
The reasoning behind a decision is sharpest at the moment it is made. Recording
it immediately avoids reconstructing a plausible-sounding rationale weeks later.

### Key metrics

None yet — no data or model exists at this phase. First metrics arrive in
Phase 2 (dataset statistics) and Phase 4 (training accuracy).

### Likely interview questions

**Q: Why did you structure the project this way instead of putting everything in one script?**
A: Separating data, source code, models and the app means each part can change
independently. I can rewrite the preprocessing without touching the Flask app,
or swap the model file without touching preprocessing. It also makes the project
readable to someone who has never seen it: the layout is a trimmed-down version
of the standard *Cookiecutter Data Science* template, so the folder names are
already familiar. For a single throwaway experiment one script is genuinely
fine — the structure earns its keep once there are multiple stages that get
re-run at different times.

**Q: What would happen if you committed the dataset and the trained model to Git?**
A: The repo would balloon to over a gigabyte, cloning would take minutes, and
GitHub would reject any single file above 100 MB. Worse, because Git snapshots
a whole new copy of a binary on every change, retraining the model a few times
would permanently bloat the history — and deleting the file later does not
shrink it, since the old versions stay in the history. The standard fixes are
either to ignore the artefacts and regenerate them (what I did here), or to use
Git LFS / DVC, which store large files outside the repo and keep only a pointer
in Git.

**Q: Why is a virtual environment worth the extra step?**
A: It gives this project its own isolated copy of its libraries. Without one,
`pip install` puts packages system-wide, so a different project needing an older
NumPy would break this one. It also makes `requirements.txt` honest — I can be
confident the file lists everything needed, because the environment started
empty.

**Q: You put `.env` and `kaggle.json` in `.gitignore` — why does that matter?**
A: `kaggle.json` holds an API key tied to my Kaggle account. If it were pushed
to a public repo, anyone could use it — and secret-scanning bots find leaked
keys within minutes. It also cannot be un-leaked by deleting the file in a later
commit, because the key is still readable in the Git history. The only real fix
after a leak is to revoke and reissue the key, so the important thing is not to
commit it in the first place.

---

## Phase 2 — Dataset download and EDA

### What was built

- **`src/download_data.py`** — a scripted, repeatable download of the
  PlantVillage archive from Kaggle that extracts **only** the 10 colour tomato
  classes directly out of the zip.
- **`notebooks/01_eda.ipynb`** — exploratory analysis covering class
  distribution, sample images per class, image size/format statistics, and a
  corrupt-file check.
- **README "Dataset" section** — source, scope, the per-class table, the
  reasoning for subsetting, and the EDA findings.

### Why — design decisions

**Why script the download instead of clicking "Download" on Kaggle?**
A script is reproducible. Anyone cloning the repo runs one command and gets
byte-identical data, with no written instructions to misread. It also documents
*exactly* which dataset was used — a manual download leaves no record of the
version or the source.

**Why extract from inside the zip rather than unzipping everything?**
The archive holds 162,916 files: 38 classes across 14 crops, each stored three
times (color / grayscale / segmented). We need 18,160 of them. Streaming the
matching members straight out of the zip means the other ~145,000 files are
never written to disk — a meaningful saving in both time and space on Windows,
where creating many small files is slow.

**Why the *colour* variant and not grayscale or segmented?**
Several tomato diseases are distinguished by the *hue* of the lesion — yellow
mottling for mosaic virus versus brown concentric rings for early blight —
information grayscale throws away. The "segmented" variant has the background
removed, which would make the task artificially easier than a real photo and
would not match what the Flask app receives at inference time.

**Why keep `data/raw` untouched and extract into a subfolder?**
The extraction writes to `data/raw/tomato/<class>/`, preserving the "folder name
= label" convention. Keras' `image_dataset_from_directory` reads exactly that
layout in Phase 3, so no separate CSV of labels is needed and there is no risk
of labels drifting out of sync with files.

**Why check for corrupt files before training rather than during?**
A truncated JPEG raises an exception the moment the data loader touches it,
which kills a training run mid-epoch — potentially 30 minutes in. A five-minute
integrity pass up front turns a random future failure into a known quantity.
(Result: zero corrupt files, so no guard clause is needed.)

### Key metrics / EDA findings

| Finding | Value |
|---|---|
| Total images | **18,160** |
| Classes | **10** (9 diseases + healthy) |
| Largest class | Tomato Yellow Leaf Curl Virus — 5,357 (29.5%) |
| Smallest class | Tomato mosaic virus — 373 (2.1%) |
| **Imbalance ratio** | **14.4×** |
| **Majority-class baseline accuracy** | **29.5%** |
| Resolution | 256×256 for **100%** of images |
| Colour mode / format | RGB / JPEG for 100% of images |
| Aspect ratios present | 1.0 only (all square) |
| Corrupt / unreadable files | **0** |

**Class balance.** Imbalance is real but moderate — 14.4× between the extremes,
not 1000×. The single most useful number here is the **29.5% majority baseline**:
it is what a model scores by ignoring the image entirely, and it is the floor any
real result must clear.

**Image uniformity.** Total consistency (256×256 RGB JPEG, no exceptions) is
unusual and convenient: no aspect-ratio distortion, no cropping decisions, no
mixed colour modes. It is also a *tell* about provenance — these are controlled
lab photographs of detached leaves on plain backgrounds, not field photos.

### Decisions this drives in later phases

1. **Stratified splitting (Phase 3).** A plain random split could leave the
   373-image mosaic-virus class with very few validation images, making its score
   statistical noise. Stratifying preserves class proportions in every split.
2. **Metrics beyond accuracy (Phase 5).** With a 29.5% baseline and a 14.4×
   imbalance, headline accuracy hides per-class failure. A confusion matrix and
   per-class recall are the honest reporting.
3. **No resampling yet.** Establish a baseline first, then fix what the confusion
   matrix actually shows is broken. Class weights are the obvious lever if the
   rare classes underperform.

### Likely interview questions

**Q: Why didn't you use the full 54,000-image dataset?**
A: Two reasons, one practical and one about problem quality. Practically, I train
a CNN from scratch on a CPU, so 18k images versus 54k is roughly a 3× difference
in epoch time — that is what makes it feasible to iterate on the architecture
several times rather than once. More interestingly, the full dataset is in some
ways an *easier* problem: much of it is telling apples from grapes from corn,
which is a crop-identification task the network can solve from overall leaf
shape and colour. Restricting to one crop forces the model to discriminate
between diseases of the same species — early blight versus late blight versus
Septoria leaf spot — which is the genuinely hard, and genuinely useful, part. A
92% on tomato-only means more than 92% across all 38 classes.

**Q: You have a 14× class imbalance. How did you handle it?**
A: In Phase 2 I measured it and let it inform the design rather than immediately
"fixing" it. Three concrete responses: **(1)** I quote the 29.5% majority-class
baseline alongside any accuracy figure, so nobody reads 85% as impressive without
knowing that guessing gets 29.5%. **(2)** The train/val/test split is stratified,
so the 373-image mosaic-virus class is proportionally represented everywhere —
otherwise its validation score would be noise. **(3)** Evaluation reports a
confusion matrix and per-class recall, since aggregate accuracy is dominated by
the large classes. I deliberately did *not* resample or reweight for the first
run: the point of a baseline is to find out what actually breaks. If mosaic virus
shows poor recall in Phase 5, class weights are the first lever — they penalise
mistakes on rare classes more heavily without discarding data the way
undersampling does, or overfitting to duplicates the way naive oversampling can.

**Q: What did the EDA tell you that changed your plan?**
A: Three things. The image uniformity — every file 256×256 RGB — meant I could
drop the aspect-ratio handling I had expected to need; resizing is a clean
downscale. Zero corrupt files meant no defensive error handling in the data
loader. And the imbalance figure is what pushed me to stratify the split and to
plan the evaluation around a confusion matrix rather than accuracy. EDA is
cheap and each of those findings removed either work or a future failure.

**Q: What is the biggest weakness of this dataset?**
A: The domain gap. PlantVillage images are lab photographs of single detached
leaves on a plain, uniform background. A real user photographs a leaf still on
the plant, with soil, sky and other foliage in frame, in uncontrolled lighting.
A model trained on this data can score very well on its own test split and still
degrade badly on real phone photos, because the test split shares the same
unrealistic conditions as the training data. High test accuracy here proves the
pipeline works; it does not prove field readiness. Closing that gap would need
either field-collected images or aggressive augmentation (random backgrounds,
lighting and scale) — and honestly reporting the limitation matters more than
hiding it behind a good number.
