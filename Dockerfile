FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen

COPY app/ app/

ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ]; then \
      mkdir -p app/models && \
      python -c "import urllib.request; urllib.request.urlretrieve('${MODEL_URL}', 'app/models/best_model.pth')"; \
    fi

ENV PYTHONPATH=/app
ENV REQUIRE_MODEL=true

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
