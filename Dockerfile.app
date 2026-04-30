ARG RUNTIME_IMAGE=europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/runtime:latest
FROM ${RUNTIME_IMAGE}

ENV APP_DOCKER_IMAGE=europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/app:latest

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY sample_results ./sample_results
COPY web ./web
COPY benchmark_results ./benchmark_results

RUN python3 -m pip install --no-cache-dir --no-deps -e .

EXPOSE 8080

CMD ["uvicorn", "peft_lab.web_app:app", "--host", "0.0.0.0", "--port", "8080"]
