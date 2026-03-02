FROM python:3.11-slim

RUN useradd -m -u 1000 user

RUN apt-get update && apt-get install -y \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app
# Dependencies — only invalidates when pyproject.toml/poetry.lock change
COPY --chown=user pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction

# Data — only invalidates when data/ changes
COPY --chown=user data/ ./data/

# Code — invalidates on every code change, but layers above are cached
COPY --chown=user app/ ./app/
COPY --chown=user supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY --chown=user . .

RUN mkdir -p /var/log/supervisor /var/run && \
    chown -R user:user /var/log/supervisor /var/run /etc/supervisor

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN chown user:user /etc/supervisor/conf.d/supervisord.conf

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]