# PetVision AI

End-to-end MLOps platform for pet image classification (cat vs dog).

This project tells a complete production story:

> I started with a **CNN baseline**, compared it against **transfer learning** and **fine-tuning** on a pretrained backbone, **optimized hyperparameters with Optuna**, **tracked everything in MLflow**, **registered the champion model**, **exposed inference via FastAPI**, **explained predictions with Grad-CAM**, and shipped a full application with **Streamlit** and **Docker**.

## Stack

| Layer | Technologies |
|-------|--------------|
| Modeling | PyTorch, TorchVision, Albumentations, MLflow, Optuna |
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| Infra | Docker, Docker Compose |
| Dev tooling | UV, Ruff, Pytest, Pre-commit |

## Deployment architecture

```
GitHub
├── FastAPI ──────────► Render (Docker + MODEL_URL)
├── Streamlit ────────► Streamlit Cloud (PETVISION_API_URL)
├── MLflow ───────────► local (treino + registry offline)
├── Docker
└── GitHub Actions ───► ruff + pytest (+ deploy hook opcional)
```

| Target | Platform | Dependencies |
|--------|----------|----------------|
| API | Render (Docker) | `[project]` only — lean image, no CUDA |
| Frontend | Streamlit Cloud | `frontend` group (`requirements-frontend.txt`) |
| Training / MLOps | Local | `training` group |
| Model artifact | GitHub Release | `best_model.pth` (gitignored, baked into Docker at build) |

## Project structure

```
petvision-ai/
├── app/
│   ├── api/main.py                 # FastAPI REST API
│   ├── frontend/streamlit_app.py   # Streamlit UI
│   ├── inference/
│   │   ├── predictor.py
│   │   ├── grad_cam.py             # Grad-CAM explainability
│   │   └── model_registry.py       # Local + MLflow model loading
│   ├── observability/              # Structured logs + inference store
│   ├── models/                     # Trained checkpoints (.pth)
│   └── schemas/prediction.py
├── training/
│   ├── train.py                    # Training with MLflow tracking
│   ├── tune.py                     # Optuna HPO (Exp 3)
│   ├── compare.py                  # Final evaluation + champion selection
│   ├── register.py                 # MLflow Model Registry
│   ├── plots.py                    # Comparison charts
│   ├── optim.py                    # Optimizer and scheduler builders
│   ├── dataset.py
│   ├── transforms.py
│   ├── evaluate.py
│   ├── config.py
│   └── models/registry.py          # Architecture registry
├── mlruns/                         # MLflow experiment artifacts
├── tests/
├── Dockerfile                      # API image (production deps only)
├── Dockerfile.frontend             # Local Streamlit image
├── docker-compose.yml
└── pyproject.toml
```

## Quick start

### 1. Install dependencies

```bash
# API only (same as Docker / Render)
uv sync

# Full local workflow (training + frontend + dev tooling)
uv sync --all-groups
uv run pre-commit install
```

### 2. Prepare the dataset

Organize images in per-class folders:

```
data/
├── cat/
│   └── *.jpg
└── dog/
    └── *.jpg
```

## MLOps workflow

### Step 1 — CNN baseline (from scratch)

Train a simple CNN without pretrained weights:

```powershell
uv run python -m training.train `
  --model simple_cnn `
  --run-name exp1-cnn-baseline `
  --epochs 10 `
  --batch-size 64
```

### Step 2 — Transfer learning experiments (EfficientNet-B0)

Compare three fine-tuning strategies on a pretrained backbone:

```powershell
# Exp 2 — Feature extraction (frozen backbone, train classifier only)
uv run python -m training.train `
  --model efficientnet_b0 `
  --freeze-strategy head_only `
  --run-name exp2-efficientnet-head-only `
  --epochs 10 `
  --batch-size 32

# Exp 3 — Partial fine-tuning (last 2 feature blocks + classifier)
uv run python -m training.train `
  --model efficientnet_b0 `
  --freeze-strategy partial `
  --run-name exp3-efficientnet-partial `
  --epochs 10 `
  --batch-size 32

# Exp 4 — Full fine-tuning (all weights trainable)
uv run python -m training.train `
  --model efficientnet_b0 `
  --freeze-strategy full `
  --run-name exp4-efficientnet-full `
  --epochs 10 `
  --batch-size 32
```

Freeze strategies (`--freeze-strategy`):

| Value | Description |
|-------|-------------|
| `head_only` | Backbone frozen; only the classifier head trains |
| `partial` | Early feature blocks frozen; last 2 blocks + classifier train |
| `full` | All weights are trainable |

Track experiments in MLflow UI:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Checkpoints are saved as `app/models/{model}_{strategy}_best.pth` (e.g. `efficientnet_b0_partial_best.pth`).
Experiment metadata lives in `mlflow.db` (SQLite).

### Step 3 — Hyperparameter optimization (Optuna, Exp 3)

Automated search to improve partial fine-tuning (`efficientnet_b0` + `partial`):

```powershell
# 10 trials × 5 epochs (metrics in MLflow, tag hpo=true, no checkpoints)
uv run python -m training.tune --n-trials 10 --epochs 5

# Final training with exported best hyperparameters
uv run python -m training.train `
  --model efficientnet_b0 `
  --freeze-strategy partial `
  --epochs 10 `
  --run-name exp3-optimized `
  --from-hpo app/models/hpo/exp3_best_params.json
```

Tuned hyperparameters: `learning_rate`, `batch_size`, `dropout`, `weight_decay`, `optimizer`, `scheduler`.
MLflow runs: parent `hpo-exp3-study` + nested `trial_001` … `trial_010`.
Optuna study persisted in `optuna_exp3.db`.

### Step 4 — Final evaluation and champion model

Compare all trained checkpoints and promote the best model by highest `val_acc`:

```powershell
uv run python -m training.compare --data-dir data
```

Per-model metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**, **ROC-AUC**, and **val_acc**.
Comparison plots are logged to MLflow (`final-model-comparison`) and saved under `reports/final_comparison/`.
The champion is promoted to `app/models/best_model.pth` (default for the API).
Manifest exported to `app/models/champion.json`.

### Step 5 — Model Registry and production serving

Full MLOps path: Baseline → Experiments → HPO → Compare → Register → Production.

```powershell
# 1. Compare models and select champion
uv run python -m training.compare --data-dir data

# 2. Register champion in MLflow Model Registry (Production stage)
uv run python -m training.register

# 3. Serve from Registry
$env:PETVISION_MODEL_SOURCE="registry"
uv run uvicorn app.api.main:app --reload --port 8000
```

Locally, the API loads checkpoints by default (`PETVISION_MODEL_SOURCE=local`).
Docker Compose is preconfigured to load `models:/petvision-classifier/Production`.

## Deploy

Render does not support Docker volume mounts (`-v ./app/models:...`). Checkpoints are gitignored (`*.pth`), so the model is **downloaded at Docker build time** via `MODEL_URL`.

### 1. Publish the model (GitHub Release)

```powershell
# Train and promote champion locally
uv run python -m training.compare --data-dir data

# Create a GitHub Release (e.g. v1.0.0) and attach app/models/best_model.pth
# Copy the public asset URL, e.g.:
# https://github.com/<user>/pet-classifier/releases/download/v1.0.0/best_model.pth
```

### 2. API on Render

1. Connect the GitHub repo to Render (or use [`render.yaml`](render.yaml) Blueprint).
2. Runtime: **Docker** (`Dockerfile`), port `8000`.
3. Set environment variables:

| Variable | Value |
|----------|-------|
| `MODEL_URL` | GitHub Release asset URL for `best_model.pth` |
| `PETVISION_MODEL_SOURCE` | `local` |
| `REQUIRE_MODEL` | `true` |
| `PETVISION_LOG_LEVEL` | `INFO` |

Render passes `MODEL_URL` as a Docker build arg; the Dockerfile downloads the checkpoint into `app/models/best_model.pth`.

Optional: add `RENDER_DEPLOY_HOOK` as a GitHub secret — CI triggers redeploy after tests pass on `main`.

```bash
# Verify locally (simulates Render build)
docker build --build-arg MODEL_URL=<release-url> -t petvision-api .
docker run -p 8000:8000 petvision-api
curl http://localhost:8000/health
```

**Future:** switch to `PETVISION_MODEL_SOURCE=registry` with a remote MLflow server (Postgres + artifact store).

### 3. Frontend on Streamlit Cloud

1. Point Streamlit Cloud at this repo.
2. Main file: `app/frontend/streamlit_app.py`.
3. Dependencies: `requirements-frontend.txt`.
4. Secrets: `PETVISION_API_URL=https://your-api.onrender.com`

### 4. GitHub Actions

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every PR and push:

- `ruff check`
- `pytest`
- On `main`: optional Render deploy hook (`RENDER_DEPLOY_HOOK` secret)

### 5. Local Docker Compose

`docker compose` still uses MLflow Registry and volume mounts for local full-stack testing — see `docker-compose.yml`.

## Serving and testing

### Run the API locally

```powershell
uv run uvicorn app.api.main:app --reload --port 8000
```

**Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | API and model status |
| `GET` | `/models` | Available architectures and checkpoints |
| `GET` | `/monitoring/inferences` | Persisted inference history |
| `POST` | `/predict` | Image classification |
| `POST` | `/explain` | Classification + Grad-CAM heatmap (base64 PNG) |

**Test with curl** (replace the image path):

```powershell
# Health check
curl.exe http://localhost:8000/health

# Predict
curl.exe -X POST "http://localhost:8000/predict?top_k=1" `
  -F "file=@data/cat/exemplo.jpeg"

# Explain with Grad-CAM
curl.exe -X POST "http://localhost:8000/explain?top_k=1" `
  -F "file=@data/cat/exemplo.jpeg"

# Explain a specific class
curl.exe -X POST "http://localhost:8000/explain?target_label=cat" `
  -F "file=@data/cat/exemplo.jpeg"

# Inference monitoring
curl.exe http://localhost:8000/monitoring/inferences?limit=20
```

### Run the Streamlit frontend

```powershell
# Terminal 1 — API
uv run uvicorn app.api.main:app --reload --port 8000

# Terminal 2 — Streamlit (set API URL to http://localhost:8000 in the sidebar)
uv run streamlit run app/frontend/streamlit_app.py
```

The UI shows the top prediction, confidence bar, and optional Grad-CAM overlay.

Supported Grad-CAM architectures: `simple_cnn`, `resnet18`, `resnet50`, `efficientnet_b0`.

### Docker Compose (full stack)

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Streamlit | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

## Observability

Every `/predict` and `/explain` call produces:

- **Structured JSON logs** on stdout (`timestamp`, `prediction`, `probability`, `latency_ms`, `model_version`, …)
- **SQLite persistence** in `inference_monitoring.db` with:
  - `timestamp`, `filename`, `prediction`, `probability`, `latency_ms`, `model_version`

| Variable | Default | Description |
|----------|---------|-------------|
| `PETVISION_INFERENCE_DB` | `inference_monitoring.db` | Monitoring database path (ephemeral on Render) |
| `PETVISION_LOG_LEVEL` | `INFO` | Structured log level |
| `PETVISION_MODEL_SOURCE` | `local` | `local` or `registry` |
| `MODEL_URL` | — | Docker build arg: URL to download `best_model.pth` |
| `REQUIRE_MODEL` | `false` | If `true`, API fails startup when model is missing |
| `PETVISION_API_URL` | `http://localhost:8000` | Streamlit default API endpoint |

## Development

```bash
# Lint
uv run ruff check .
uv run ruff format .

# Tests (unit + integration)
uv run pytest

# With coverage
uv run pytest --cov=app --cov=training
```

## Extending the platform

- **Architectures**: add models in `training/models/registry.py`.
- **Experiments**: each `training.train` run logs params, metrics, and artifacts to MLflow.
- **Checkpoints**: store additional `.pth` files in `app/models/` and load via `ModelRegistry`.
- **Model Registry**: after `training.register`, the champion is available as `petvision-classifier@Production`.

## License

MIT
