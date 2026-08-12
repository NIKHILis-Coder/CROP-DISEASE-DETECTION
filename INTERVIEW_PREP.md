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

---

## Phase 3 — Preprocessing and augmentation pipeline

### What was built

- **`src/data_loader.py`** — the full path from folders of JPEGs to batched
  tensors: manifest building, stratified 70/15/15 splitting, persisted split
  CSVs, `tf.data` pipelines (decode → resize 128×128 → scale to [0,1]), and
  Keras augmentation layers applied to training data only.
- **`notebooks/02_pipeline_check.ipynb`** — six explicit PASS/FAIL checks run
  before any training, covering shape, pixel range, label alignment,
  stratification, augmentation behaviour, and val/test purity.
- **README "Preprocessing" section** documenting each choice.

### Why — design decisions

**Why in-graph preprocessing rather than writing processed images to disk?**
Three reasons, and the first is the decisive one:

1. **Augmentation has to be on the fly to work at all.** The point of
   augmentation is that the model sees a *differently* distorted copy of each
   image every epoch. Writing augmented copies to disk freezes a fixed set of
   variants — the model just memorises 5 versions instead of 1, and most of the
   regularisation benefit disappears.
2. **No stale derived data.** Changing the input size means changing one
   constant. With a preprocessed copy on disk, there is a second dataset that can
   silently disagree with the code that produced it — and nothing warns you.
3. **Cost.** No second 18k-image copy to write, store or keep in sync.

The trade-off is CPU work per epoch: decoding and resizing happen every time
rather than once. `prefetch` and parallel decoding hide most of that cost, and
for an 18k-image dataset it is clearly the right side of the trade. At a scale
where decoding genuinely bottlenecks the model, the answer is
`dataset.cache()` or a pre-built TFRecord — not a folder of JPEGs.

**Why persist the split manifest, when nothing else is persisted?**
Because the split must *never* change, while everything else must be free to.
If the split were recomputed with a different shuffle between training and
evaluation, images the model trained on would land in the test set and inflate
the score. Saving three CSVs of paths (a few hundred KB) fixes the split
permanently, and a fixed `random_state=42` makes it reproducible from scratch.

**Why stratify?** With a 14.4× imbalance, a plain random split could deal the
373-image mosaic-virus class an unlucky hand — leaving it with, say, 30 test
images, where a handful of errors swings its recall by 10 points and the number
means nothing. Stratifying forces every split to mirror the overall
distribution. Measured result: the largest class-share drift between any two
splits is **0.1 percentage points**.

**Why these three augmentations specifically?** Each models a variation that
genuinely occurs when photographing a leaf, and none of them changes the correct
answer:

- `RandomFlip` (horizontal + vertical) — a leaf has no inherent "up"; a flipped
  leaf is the same disease. Free, completely safe variety.
- `RandomRotation(0.1)` (≈±36°) — the camera not held square to the leaf.
- `RandomZoom(0.1)` — the camera slightly nearer or further.

**Why not more aggressive augmentation?** This is the more interesting half, and
it is specific to this domain:

- **No colour/brightness/hue jitter.** Diagnosis here depends on hue — yellow
  mottling means mosaic virus, brown concentric rings mean early blight. Shifting
  colours attacks the exact signal the model needs, and can push an image towards
  the appearance of a *different* disease while keeping its original label. That
  is actively harmful: it teaches the model something false.
- **No shear or perspective warping.** These are flat, square-on lab photographs
  of detached leaves. Simulating extreme perspective invents a distribution that
  exists in neither the training nor the test data, spending model capacity on
  variation that never occurs.
- **No random cropping.** A lesion can be anywhere on the leaf. A crop may remove
  the only diseased region while keeping the "diseased" label — again, a
  wrong-label image.
- **Rotation/zoom capped at ±10%** because larger rotations pad the corners with
  empty pixels, and a network will happily learn to read that padding artefact
  instead of the leaf.

The general principle: augment along axes that genuinely vary (orientation,
distance) and leave alone the axes that carry the diagnostic signal (colour).

### Key metrics

| Split | Images | Share | Imbalance ratio |
|-------|-------:|------:|----------------:|
| Train | 12,712 | 70.0% | 14.4× |
| Validation | 2,724 | 15.0% | 14.3× |
| Test | 2,724 | 15.0% | 14.4× |

**Stratification verification — largest class-share drift between any two
splits: 0.1 percentage points.** Per-class shares match the full-dataset
distribution across all three splits (e.g. Yellow Leaf Curl Virus 29.5% / 29.5%
/ 29.5%; mosaic virus 2.1% / 2.1% / 2.1%).

**Pipeline check results — all six PASS:**

| Check | Result |
|---|---|
| Batch shape | `(32, 128, 128, 3)` float32 for all splits |
| Pixel range | min 0.000, max ≈0.95–0.97 → normalised exactly once |
| Label validity | indices 0–9, all 10 classes seen across 320 samples |
| Augmentation alters images | mean abs. diff from original 0.121 |
| Augmentation is random | mean abs. diff between two calls 0.094 |
| Val/test unaugmented | two passes bit-for-bit identical (max diff 0.0000000000) |

### Likely interview questions

**Q: Why do you augment only the training data and not validation or test?**
A: Because they answer different questions. Training data exists to teach the
model, so distorting it is useful — it forces the model to learn the underlying
pattern rather than memorise specific photographs. Validation and test data exist
to *measure* the model on realistic inputs. If I augmented them, I would be
measuring performance on randomly distorted images that no user will ever submit,
and worse, the score would jitter between runs as the random distortions changed
— so I could not tell whether a change to the model helped or the dice just fell
differently. There is a subtle trap here too: Keras preprocessing layers are
deliberately no-ops when called with `training=False`, so if you put augmentation
*inside* the model rather than in the input pipeline, it automatically disables
itself at evaluation time. I kept it in the `tf.data` pipeline and applied it
explicitly with `training=True`, then wrote a test asserting that two passes over
the validation set are bit-for-bit identical — I would rather assert it than
assume it.

**Q: Why those augmentation types and not others?**
A: The rule I used is that an augmentation must change the image without ever
changing the correct label, and it should model a variation that actually occurs.
Flips, small rotations and small zooms all pass: a leaf photographed upside down
or from slightly further away is still the same disease. Colour jitter fails,
and that is domain-specific reasoning — several of these diseases are identified
*by* colour, so shifting hue can make a mosaic-virus leaf look like early blight
while the label still says mosaic virus. That does not regularise the model, it
poisons the labels. Random cropping fails for the same reason: crop out the only
lesion and you have a picture of a healthy leaf labelled "diseased". This is the
part people get wrong by reaching for a standard augmentation recipe — the right
set depends entirely on what carries the signal in your data.

**Q: How do you prevent data leakage between splits?**
A: Four things. First, the split is on *file paths*, made once, before any image
is loaded — so a single physical image can only ever land in one split. Second,
the split is written to CSV and read back, rather than recomputed; a fresh
shuffle between training and evaluation is the classic way test images silently
end up in training. Third, `random_state` is fixed, so the split is identical on
every machine and every run. Fourth — the one people forget — **all preprocessing
statistics must come from training data only.** Here that is trivially satisfied
because normalisation is a fixed divide by 255, not a mean/std computed from the
data. If I had used per-channel standardisation, I would have had to compute the
mean and standard deviation on the training split alone and apply those same
constants to val and test; computing them over the whole dataset would leak
information about the test set into the model's inputs.

**Q: Why 128×128 rather than keeping the native 256×256?**
A: Compute. Halving each dimension quarters the pixel count, so roughly quarters
the convolution work per image. Training from scratch on a CPU, that is the
difference between iterating on the architecture several times in an evening and
managing one run overnight — and the number of experiments you can afford is
usually what determines the final result. I did not go smaller than 128 because
at 96×96 the fine speckling separating Septoria leaf spot from Target Spot starts
to smear, and no model can learn a feature that is no longer present in its
input. If accuracy on those two classes turns out to be the weak point in
Phase 5, raising the input size is one of the first things I would try.

---

## Phase 4 (part 1) — Architecture and benchmark

*Training results follow after the full run; this section covers the design and
the timed benchmark only.*

### What was built

- **`src/model.py`** — `build_model()` returning the compiled CNN: three
  Conv→BatchNorm→MaxPool blocks (32/64/128 filters), GlobalAveragePooling,
  Dense(128) + Dropout(0.5), Dense(10, softmax). **111,946 parameters.**
- **`src/train.py`** — loads the Phase 3 pipelines, computes inverse-frequency
  class weights from the training split, and runs either a **one-epoch timed
  benchmark** (default) or the full run with early stopping (`--full`).
- **Split manifests committed.** `data/processed/*.csv` is now un-ignored, so the
  exact split is reproducible from the repository rather than from a seed.

### Architecture

| Layer | Output shape | Params |
|---|---|---:|
| Input | (128, 128, 3) | 0 |
| conv1 — Conv2D(32, 3×3, relu, same) | (128, 128, 32) | 896 |
| bn1 — BatchNormalization | (128, 128, 32) | 128 |
| pool1 — MaxPooling2D(2×2) | (64, 64, 32) | 0 |
| conv2 — Conv2D(64, 3×3, relu, same) | (64, 64, 64) | 18,496 |
| bn2 — BatchNormalization | (64, 64, 64) | 256 |
| pool2 — MaxPooling2D(2×2) | (32, 32, 64) | 0 |
| conv3 — Conv2D(128, 3×3, relu, same) | (32, 32, 128) | 73,856 |
| bn3 — BatchNormalization | (32, 32, 128) | 512 |
| pool3 — MaxPooling2D(2×2) | (16, 16, 128) | 0 |
| gap — GlobalAveragePooling2D | (128) | 0 |
| dense1 — Dense(128, relu) | (128) | 16,512 |
| dropout — Dropout(0.5) | (128) | 0 |
| output — Dense(10, softmax) | (10) | 1,290 |
| **Total** | | **111,946** |

Trainable 111,498 · non-trainable 448 (BatchNorm moving averages).

### Why — design decisions

**Why three conv blocks, not two or five?**
Each block halves the spatial size: 128 → 64 → 32 → 16. Three blocks land on a
16×16 map, which is the sweet spot — small enough that global pooling summarises
it meaningfully, large enough that spatial information has not been thrown away.
Two blocks would stop at 32×32 with only 64 filters, leaving the deepest features
still fairly local — they would describe textures, not lesion-scale structure.
Five blocks would take it to 4×4, which for 128×128 inputs starts discarding
detail, and would add parameters to a model that already has to be regularised
hard against 12,712 training images. Depth is also not free here: the benchmark
says 8.6 minutes per epoch on CPU, and a fourth block would add meaningfully to
that. Three is the point where the receptive field covers a useful fraction of
the leaf without either wasting compute or overfitting.

**Why 32 → 64 → 128 filters?**
Doubling the filter count each time the spatial size halves is the standard CNN
trade-off, and it is a genuine trade rather than a convention. Early layers see
small patches and only need a few kinds of pattern — edges, colour transitions —
so 32 filters suffice. Deeper layers see larger regions, and the number of
*distinct* things a large region can contain is much greater, so more filters are
needed to describe them. Doubling filters while quartering the pixel count also
keeps the computational cost per block roughly balanced rather than
front-loading or back-loading the network. Starting at 32 rather than 64 halves
the cost of the most expensive layer — conv1 runs at full 128×128 resolution.

**Why GlobalAveragePooling2D instead of Flatten?**
This is the single highest-leverage choice in the architecture. The final feature
map is 16×16×128. `Flatten` would produce a 32,768-element vector, and connecting
that to `Dense(128)` costs **32,768 × 128 ≈ 4.2 million parameters** — turning a
112k-parameter model into a 4.3M-parameter one, where **97% of the weights sit in
a single layer**. With 12,712 training images that layer would overfit almost
immediately; it has enough capacity to memorise the training set outright.

`GlobalAveragePooling2D` instead averages each of the 128 feature maps down to
one number, giving a 128-element vector and a 16,512-parameter dense layer — a
**254× reduction**. It also gives something Flatten cannot: translation
invariance. The average of a feature map does not depend on *where* in the leaf
the feature appeared, which is exactly right here, because a lesion in the top
left and the same lesion in the bottom right mean the same diagnosis. Flatten
would force the model to learn that equivalence from data instead.

**Why BatchNorm after each Conv?**
As a network trains, the distribution of each layer's inputs keeps shifting
because the layers below it are still changing. BatchNorm renormalises each
batch's activations back to a stable scale, which (a) lets training use a higher
learning rate without diverging, (b) speeds up convergence — worth a lot when an
epoch costs 8.6 minutes — and (c) provides a mild regularising effect, since each
example's normalisation depends on the other examples in its batch.

Worth knowing for follow-up questions: I placed BatchNorm *after* the ReLU
activation (`Conv(relu) → BN → Pool`). The original paper puts it *before* the
activation (`Conv → BN → ReLU`), and that ordering is marginally more common in
modern code. In practice the difference is small and empirically contested; the
important thing is to know that both exist and that the choice is deliberate.

**Why Dropout 0.5 on the dense layer only, and not after the conv layers?**
Dropout randomly zeroes units during training so the model cannot lean on any
single feature. It belongs where the parameters are, and after global pooling the
dense layers hold 16,512 + 1,290 of the weights in the most overfit-prone part of
the network. 0.5 is the rate the original Dropout paper recommends for fully
connected layers and is a sensible default.

Conv layers are a different case. They already have far fewer parameters
(weight sharing means one 3×3 filter is 9 weights regardless of image size), and
they are already regularised by BatchNorm, by pooling, and by the input
augmentation from Phase 3. Dropping random pixels from a feature map is also less
effective than it sounds, because neighbouring pixels are highly correlated — the
dropped information is usually still present next door. If the model still
overfits, `SpatialDropout2D` (which drops entire feature maps) is the right tool,
not plain Dropout.

**Why `class_weight` rather than oversampling or focal loss?**
All three address the 14.4× imbalance; they differ in cost and side-effects.

- **`class_weight` (chosen)** multiplies each example's contribution to the loss
  by its class weight, so every *class* contributes roughly equally rather than
  every *image*. It costs nothing — no extra data, no change to epoch time — and
  it is a single argument to `fit()`, so it is trivial to ablate: run with and
  without, compare. That matters when each experiment costs 8.6 min/epoch.
- **Oversampling** duplicates rare-class images. The 373-image mosaic-virus class
  would need ~14× duplication, and the model then sees the *same* 373 images over
  and over, which invites memorising them specifically. It also lengthens every
  epoch by inflating the dataset.
- **Focal loss** down-weights easy examples to focus on hard ones. It is designed
  for extreme imbalance (object detection, where background outnumbers objects
  ~1000:1) and introduces a second hyperparameter, γ, to tune. At 14.4× it is
  reaching for a heavier tool than the problem needs.

The honest framing: at moderate imbalance, class weighting gets most of the
benefit for none of the cost. If Phase 5's confusion matrix shows mosaic virus
still failing, focal loss or targeted data collection become worth the extra
complexity.

Computed weights (from the training split only, so nothing leaks from val/test):

| Class | Train images | Share | Weight |
|---|---:|---:|---:|
| Tomato Yellow Leaf Curl Virus | 3,750 | 29.5% | **0.339** |
| Bacterial spot | 1,489 | 11.7% | 0.854 |
| Late blight | 1,336 | 10.5% | 0.951 |
| Septoria leaf spot | 1,240 | 9.8% | 1.025 |
| Spider mites (two-spotted) | 1,173 | 9.2% | 1.084 |
| Healthy | 1,114 | 8.8% | 1.141 |
| Target spot | 983 | 7.7% | 1.293 |
| Early blight | 700 | 5.5% | 1.816 |
| Leaf mold | 666 | 5.2% | 1.909 |
| Tomato mosaic virus | 261 | 2.1% | **4.870** |

Weight ratio 0.339 → 4.870 = **14.4×**, exactly mirroring the class imbalance,
which is the arithmetic working as intended: `weight = n_samples / (n_classes ×
class_count)`, so a class holding exactly 1/10th of the data gets weight 1.0.

**Why early-stop on `val_loss` rather than `val_accuracy`?**
Because with a 14.4× imbalance, accuracy is a blunt instrument. A model can raise
accuracy simply by getting better at Yellow Leaf Curl Virus — 29.5% of the data —
while getting *worse* at mosaic virus at 2.1%. Accuracy would go up and the model
would be worse at the thing that is hard.

Loss is more sensitive in two ways that matter here. It is computed on the
predicted *probabilities*, not just the arg-max, so it registers a model becoming
less confident about correct answers — a genuine early warning of overfitting
that accuracy is blind to, because the arg-max does not change until confidence
has already collapsed. And because `class_weight` is applied to the loss, the
monitored quantity is the *class-balanced* loss, which is exactly the objective
being optimised. Accuracy is also a step function of the arg-max, so it moves in
discrete jumps and plateaus — poor signal for a patience-based rule, which needs
to detect small consistent degradation.

### Benchmark results (one timed epoch)

| Measurement | Value |
|---|---|
| Time for 1 epoch | **514 s (8.6 min)** — 398 steps at ~1.3 s/step |
| Estimated 30 epochs | **~4.3 hours** (conservative — epoch 1 includes graph tracing) |
| Training loss / accuracy | **1.176 / 61.21%** |
| Validation loss / accuracy | **3.794 / 23.94%** |
| Random-guess loss, ln(10) | 2.303 |
| Majority-class baseline | 29.5% |

**Reading these numbers.** Training loss of 1.176 against a random baseline of
2.303, with 61% training accuracy after a single pass, says the model is learning
real structure — this is the check the benchmark exists for.

The validation numbers are *worse* than random-ish, and that is worth explaining
rather than glossing over. Validation loss (3.79) far above training loss (1.18)
after one epoch is the classic **BatchNorm warm-up** signature. During training,
BatchNorm normalises using each batch's own statistics; at inference it uses
moving averages accumulated across training. After one epoch those averages have
only had 398 updates at momentum 0.99 and are still far from the true
activation statistics, so the model behaves quite differently in inference mode
than in training mode. This typically resolves within a few epochs as the moving
averages converge. It is a known artefact of measuring after exactly one epoch,
not evidence that the architecture is broken — but the way to *confirm* that is
the full run, not assertion.

### Likely interview questions

**Q: Your model has 112k parameters. Isn't that far too small to be useful?**
A: It is small deliberately, and the constraint that sets the size is the data,
not the ambition. There are 12,712 training images; a model with millions of free
parameters can memorise that outright, and the validation curve would diverge
from training within a few epochs. The parameter count is also where the
GlobalAveragePooling choice pays off — swapping it for Flatten would take the
model from 112k to 4.3M parameters, with 97% of them in one dense layer, purely
to preserve spatial position information the task does not need. Small also means
fast: 8.6 minutes per epoch on CPU, so the whole run fits in an afternoon and I
can afford to try things. If the model turns out to *underfit* — training accuracy
plateauing well below what the task allows — then adding capacity is the right
response, and I would add a fourth conv block before I would touch the dense head.

**Q: You measured validation loss of 3.79 after one epoch, far worse than
training loss of 1.18. Isn't that overfitting already?**
A: It is very unlikely to be overfitting after a single pass through the data —
a 112k-parameter model has not had the opportunity to memorise 12,712 images in
398 gradient steps. The much more probable cause is BatchNorm's train/inference
mismatch. In training mode BatchNorm normalises using the current batch's mean
and variance; in inference mode it uses moving averages accumulated during
training, which after one epoch are still poorly estimated. So the training
number is measured with good statistics and the validation number with bad ones,
and they are not yet comparable. The distinguishing test is simply to keep
training: BatchNorm warm-up closes over the next few epochs, whereas genuine
overfitting shows validation loss *rising* while training loss keeps falling,
after both have first come down together. If the gap persisted past five or six
epochs I would suspect something structural — and I would check the augmentation
was not accidentally applied to validation, which Phase 3 already tests for.

**Q: Why run a one-epoch benchmark at all instead of just training?**
A: Two reasons, and the second is the one people underrate. First, it converts an
unknown into a number: 8.6 minutes per epoch means 30 epochs is 4.3 hours, which
is a decision I can now make deliberately instead of discovering three hours in.
Second, it is the cheapest possible smoke test for a broken pipeline. If labels
were misaligned or the learning rate were wildly wrong, training loss would sit
near ln(10) = 2.30 and I would know within nine minutes rather than at the end of
a long run. Getting 1.176 tells me the whole chain — manifest, split, decode,
resize, normalise, label mapping, loss function — is wired up correctly. That
is a lot of things confirmed for very little time.
