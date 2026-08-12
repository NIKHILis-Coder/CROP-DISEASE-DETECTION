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
