FROM python:3.13-slim

WORKDIR /app

ENV POETRY_VERSION=2.5.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH=/app

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION" \
    && poetry install --without dev --no-interaction --no-ansi \
    && apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "src/step7_pipeline/pipeline_cv.py"]
