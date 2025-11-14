# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && \
    apt-get install --yes --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --format requirements.txt --locked --output-file requirements.txt && \
    uv pip install --system --requirements requirements.txt && \
    rm requirements.txt

COPY train.py transformer.py ./

CMD ["python", "train.py"]
