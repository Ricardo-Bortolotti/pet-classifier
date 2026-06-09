# PetVision AI

Production-grade image classification platform using PyTorch, FastAPI, Docker and MLOps practices.

## Stack

| Camada | Tecnologias |
|--------|-------------|
| Modelagem | PyTorch, TorchVision, Albumentations, MLflow |
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| Infra | Docker, Docker Compose |
| Desenvolvimento | UV, Ruff, Pytest, Pre-commit |

## Arquitetura

```
petvision-ai/
├── app/
│   ├── api/main.py              # FastAPI REST API
│   ├── frontend/streamlit_app.py
│   ├── inference/
│   │   ├── predictor.py
│   │   └── model_registry.py    # Carrega múltiplos checkpoints
│   ├── models/                  # Checkpoints treinados (.pth)
│   └── schemas/prediction.py
├── training/
│   ├── train.py                 # Treino com MLflow
│   ├── tune.py                  # HPO com Optuna (Exp 3)
│   ├── compare.py               # Avaliação final e champion model
│   ├── register.py              # Registro no MLflow Model Registry
│   ├── plots.py                 # Gráficos comparativos
│   ├── optim.py                 # Optimizer e scheduler
│   ├── dataset.py
│   ├── transforms.py
│   ├── evaluate.py
│   ├── config.py
│   └── models/registry.py       # Registro de arquiteturas
├── mlruns/                      # Experimentos MLflow
├── tests/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Início rápido

### 1. Instalar dependências

```bash
uv sync
```

### 2. Pre-commit

```bash
uv run pre-commit install
```

### 3. Treinar um modelo

Organize o dataset em subpastas por classe:

```
data/
├── cat/
│   └── *.jpg
└── dog/
    └── *.jpg
```

Sequência de experimentos para comparar no MLflow:

```powershell
# Exp 1 — CNN Baseline (do zero)
uv run python -m training.train --model simple_cnn --run-name exp1-cnn-baseline --epochs 10 --batch-size 64

# Exp 2 — Transfer Learning (feature extraction, backbone congelado)
uv run python -m training.train --model efficientnet_b0 --freeze-strategy head_only --run-name exp2-efficientnet-head-only --epochs 10 --batch-size 32

# Exp 3 — Fine-Tuning Parcial (últimos 2 blocos + classifier)
uv run python -m training.train --model efficientnet_b0 --freeze-strategy partial --run-name exp3-efficientnet-partial --epochs 10 --batch-size 32

# Exp 4 — Fine-Tuning Completo (todos os pesos)
uv run python -m training.train --model efficientnet_b0 --freeze-strategy full --run-name exp4-efficientnet-full --epochs 10 --batch-size 32
```

Estratégias de congelamento (`--freeze-strategy`):
| Valor | Descrição |
|-------|-----------|
| `head_only` | `features` congelado, só `classifier` treina |
| `partial` | `features[:-2]` congelado, `features[-2:]` + `classifier` treinam |
| `full` | Todos os pesos treináveis |

Compare runs no MLflow UI:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Checkpoints: `app/models/{modelo}_{estrategia}_best.pth` (ex: `efficientnet_b0_partial_best.pth`).
Experimentos ficam em `mlflow.db` (SQLite).

#### HPO — Otimização de hiperparâmetros (Exp 3)

Busca automática com Optuna para melhorar o fine-tuning parcial (`efficientnet_b0` + `partial`):

```powershell
# 10 trials × 5 épocas (métricas no MLflow, tag hpo=true, sem checkpoints)
uv run python -m training.tune --n-trials 10 --epochs 5

# Treino final com os melhores hiperparâmetros exportados
uv run python -m training.train --model efficientnet_b0 --freeze-strategy partial `
  --epochs 10 --run-name exp3-optimized --from-hpo app/models/hpo/exp3_best_params.json
```

Hiperparâmetros buscados: `learning_rate`, `batch_size`, `dropout`, `weight_decay`, `optimizer`, `scheduler`.
Runs no MLflow: run pai `hpo-exp3-study` + nested runs `trial_001` … `trial_010`.
Estudo Optuna persistido em `optuna_exp3.db`.

#### Etapa 7 — Avaliação Final e Champion Model

Compare todos os checkpoints treinados e selecione automaticamente o melhor modelo pelo maior `val_acc`:

```powershell
uv run python -m training.compare --data-dir data
```

Métricas calculadas por modelo: **Accuracy**, **Precision**, **Recall**, **F1-score**, **ROC-AUC** e **val_acc**.
Gráficos comparativos registrados no MLflow (run `final-model-comparison`) e salvos em `reports/final_comparison/`.
O champion é promovido para `app/models/best_model.pth` (usado pela API por padrão).
Manifesto exportado em `app/models/champion.json`.

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

#### Etapa 8 — Model Registry e Produção

Fluxo MLOps completo: Baseline → Experimentos → HPO → Compare → Register → Produção.

```powershell
# 1. Comparar modelos e eleger champion
uv run python -m training.compare --data-dir data

# 2. Registrar champion no MLflow Model Registry (stage Production)
uv run python -m training.register

# 3. API em modo produção (carrega automaticamente do Registry)
$env:PETVISION_MODEL_SOURCE="registry"
uv run uvicorn app.api.main:app --reload --port 8000
```

Em desenvolvimento local, a API usa checkpoints locais por padrão (`PETVISION_MODEL_SOURCE=local`).
No Docker Compose, a API já está configurada para carregar `models:/petvision-classifier/Production`.

### 4. Rodar a API localmente

```bash
uv run uvicorn app.api.main:app --reload --port 8000
```

Endpoints:
- `GET /health` — status da API
- `GET /models` — arquiteturas e checkpoints disponíveis
- `GET /monitoring/inferences` — histórico de inferências persistidas
- `POST /predict` — classificação de imagem
- `POST /explain` — classificação + Grad-CAM (heatmap base64)

#### Etapa 11 — Grad-CAM (Explainability)

Visualize as regiões da imagem que mais influenciaram a predição:

```powershell
# Via API (retorna overlay PNG em base64)
curl -X POST "http://localhost:8000/explain?top_k=3" -F "file=@cat.jpg"

# Explicar uma classe específica
curl -X POST "http://localhost:8000/explain?target_label=cat" -F "file=@cat.jpg"
```

No Streamlit, marque **Mostrar Grad-CAM** na barra lateral para ver o overlay após a classificação.
Arquiteturas suportadas: `simple_cnn`, `resnet18`, `resnet50`, `efficientnet_b0`.

### 5. Rodar o frontend Streamlit

```bash
uv run streamlit run app/frontend/streamlit_app.py
```

### 6. Docker Compose (stack completa)

```bash
docker compose up --build
```

| Serviço | URL |
|---------|-----|
| API | http://localhost:8000 |
| Streamlit | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

#### Etapa 12 — Testes e Observabilidade

Cada inferência (`/predict` e `/explain`) gera:

- **Log estruturado JSON** no stdout (`timestamp`, `prediction`, `probability`, `latency_ms`, `model_version`, etc.)
- **Persistência SQLite** em `inference_monitoring.db` com os campos:
  - `timestamp`, `filename`, `prediction`, `probability`, `latency_ms`, `model_version`

Consultar histórico de inferências:

```powershell
curl http://localhost:8000/monitoring/inferences?limit=20
```

Variáveis de ambiente:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `PETVISION_INFERENCE_DB` | `inference_monitoring.db` | Caminho do banco de monitoramento |
| `PETVISION_LOG_LEVEL` | `INFO` | Nível dos logs estruturados |

## Desenvolvimento

```bash
# Lint
uv run ruff check .
uv run ruff format .

# Testes (unitários + integração)
uv run pytest

# Com cobertura
uv run pytest --cov=app --cov=training
```

## Múltiplos modelos e experimentos

- **Arquiteturas**: adicione novos modelos em `training/models/registry.py`.
- **Experimentos**: cada execução de `training.train` cria um run no MLflow com parâmetros, métricas e artefatos.
- **Checkpoints**: salve modelos adicionais em `app/models/` e carregue via `ModelRegistry`.
- **Model Registry**: após `training.register`, o champion fica em `petvision-classifier@Production` no MLflow.

## Licença

MIT
