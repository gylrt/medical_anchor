FROM python:3.11-slim

RUN useradd -m -u 1000 user

RUN apt-get update && apt-get install -y \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

WORKDIR /app

# Dependencies: rebuild only when lock/dependency files change.
COPY --chown=user pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction

# Runtime data directories (assets can be downloaded from HF at startup).
RUN mkdir -p /app/data/chroma /app/data/processed

# App code: invalidates on code changes only.
COPY --chown=user app/ ./app/
COPY --chown=user supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir -p /var/log/supervisor /var/run && \
    chown -R user:user /var/log/supervisor /var/run /etc/supervisor /app/data

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_DOWNLOAD_IF_MISSING=true \
    HF_DATASET_REPO_ID=gylrt/medical-anchor-dataset

EXPOSE 7860

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
