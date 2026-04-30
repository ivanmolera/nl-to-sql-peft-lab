FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/opt/hf-cache
ENV TRANSFORMERS_CACHE=/opt/hf-cache/hub
ENV HF_DATASETS_CACHE=/opt/hf-cache/datasets
ENV APP_RUNTIME=docker
ENV APP_DOCKER_IMAGE=europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/ml:latest

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY sample_results ./sample_results
COPY web ./web

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip install --no-cache-dir -e .

RUN python3 scripts/download_hf_assets.py --config configs/hf_assets.yaml --only models
RUN python3 scripts/download_hf_assets.py --config configs/hf_assets.yaml --only datasets

COPY benchmark_results ./benchmark_results

EXPOSE 8080

CMD ["uvicorn", "peft_lab.web_app:app", "--host", "0.0.0.0", "--port", "8080"]
