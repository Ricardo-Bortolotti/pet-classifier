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

## Architecture

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
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Quick start

### 1. Install dependencies

```bash
uv sync
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
| `PETVISION_INFERENCE_DB` | `inference_monitoring.db` | Monitoring database path |
| `PETVISION_LOG_LEVEL` | `INFO` | Structured log level |
| `PETVISION_MODEL_SOURCE` | `local` | `local` or `registry` |

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
