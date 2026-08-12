# Flask App — Crop Disease Detection

Web interface for the trained tomato-leaf disease classifier. Upload a leaf
photo, get the top three predictions with confidence scores.

## Prerequisites

A trained model must exist at `models/best_model.keras`. If it doesn't:

```bash
python src/download_data.py      # fetch the dataset (once)
python src/train.py --full       # train the model
```

## Run

From the repository root:

```bash
# activate the virtual environment first
venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate         # macOS / Linux

cd app
python app.py
```

Then open **http://localhost:5000**.

The model loads once at startup, so the first request is as fast as the rest.
Startup prints the class list it loaded — check that it shows 10 classes.

## Routes

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Upload page |
| `POST` | `/predict` | Accepts an image file field named `file`; returns top-3 predictions as JSON |
| `GET` | `/health` | `{"status": "ok", ...}` — liveness check; returns 503 if the model failed to load |

### Example

```bash
curl -s http://localhost:5000/health

curl -s -X POST http://localhost:5000/predict \
     -F "file=@../data/raw/tomato/Tomato___healthy/some_leaf.JPG"
```

```json
{
  "predictions": [
    {"class": "healthy", "label": "Healthy", "confidence": 0.87},
    {"class": "Late_blight", "label": "Late Blight", "confidence": 0.06},
    {"class": "Leaf_Mold", "label": "Leaf Mold", "confidence": 0.03}
  ],
  "top_prediction": "Healthy",
  "confidence": 0.87
}
```

## Error handling

Every failure returns a JSON message and an appropriate status code — never a
stack trace:

| Situation | Status | Response |
|---|---|---|
| No file in the request | 400 | "No file was uploaded…" |
| Empty filename | 400 | "No file was selected." |
| Wrong extension (e.g. `.pdf`) | 400 | "Unsupported file type…" |
| Empty or corrupt image | 400 | "That file could not be read as an image…" |
| File larger than 8 MB | 413 | Flask's `MAX_CONTENT_LENGTH` limit |
| Model failed to load at startup | 503 | "Model is not loaded…" |
| Inference raised | 500 | Generic message; the traceback goes to the server log only |

## Notes

- **Preprocessing must match training exactly.** The app imports `IMAGE_SIZE`
  from `src/data_loader.py` rather than hardcoding it, so a retrain at a
  different resolution cannot silently desynchronise the two.
- **Class order matters.** Class names come from `sorted(unique labels)` of the
  training manifest — the same expression training used. Any other ordering
  would mislabel every prediction while looking entirely plausible.
- `debug=False` deliberately: the reloader would load the model twice, and the
  interactive debugger should never be exposed by a demo left running.
