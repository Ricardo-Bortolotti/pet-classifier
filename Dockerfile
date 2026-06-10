FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen

COPY app/ app/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
