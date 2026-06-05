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

Treine com a arquitetura desejada:

```bash
# ResNet18 (padrão)
uv run python -m training.train --data-dir data --epochs 10

# ResNet50
uv run python -m training.train --model resnet50 --data-dir data

# EfficientNet-B0
uv run python -m training.train --model efficientnet_b0 --data-dir data --run-name efficientnet-exp1
```

Modelos disponíveis: `resnet18`, `resnet50`, `efficientnet_b0`.

O melhor checkpoint é salvo em `app/models/best_model.pth` e os experimentos em `mlruns/`.

### 4. Rodar a API localmente

```bash
uv run uvicorn app.api.main:app --reload --port 8000
```

Endpoints:
- `GET /health` — status da API
- `GET /models` — arquiteturas e checkpoints disponíveis
- `POST /predict` — classificação de imagem

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

## Desenvolvimento

```bash
# Lint
uv run ruff check .
uv run ruff format .

# Testes
uv run pytest

# Com cobertura
uv run pytest --cov=app --cov=training
```

## Múltiplos modelos e experimentos

- **Arquiteturas**: adicione novos modelos em `training/models/registry.py`.
- **Experimentos**: cada execução de `training.train` cria um run no MLflow com parâmetros, métricas e artefatos.
- **Checkpoints**: salve modelos adicionais em `app/models/` e carregue via `ModelRegistry`.

## Licença

MIT
