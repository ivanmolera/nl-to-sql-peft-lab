FROM python:3.11-slim

ENV APP_DOCKER_IMAGE=europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/app:latest
ENV APP_RUNTIME=docker
ENV APP_VERSION=v0.1.0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.web.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY sample_results ./sample_results
COPY web ./web
COPY benchmark_results ./benchmark_results

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.web.txt \
    && python -m pip install --no-cache-dir --no-deps -e .

EXPOSE 8080

CMD ["uvicorn", "peft_lab.web_app:app", "--host", "0.0.0.0", "--port", "8080"]
