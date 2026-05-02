# NL-to-SQL PEFT Lab

Parameter-Efficient Fine-Tuning technique comparison for NL-to-SQL on small
generative models.

NL-to-SQL PEFT Lab is a reproducible experimentation platform for evaluating how
parameter-efficient fine-tuning techniques affect small language models on the
WikiSQL text-to-SQL task. It combines training pipelines, benchmark scripts, stored
evaluation artifacts, and a web dashboard where users can compare results and test
trained variants on real WikiSQL examples.

The model suite includes `google-t5/t5-small`, `openai-community/gpt2`,
`HuggingFaceTB/SmolLM2-135M-Instruct`, and
`Qwen/Qwen2.5-Coder-0.5B-Instruct`. The lab is designed to apply the same
evaluation flow to the full selected model set. The benchmark dataset is
`Salesforce/wikisql`, with a project-specific WikiSQL NL-to-SQL benchmark runner
designed to be compatible with a Hugging Face LightEval custom task. The PEFT
comparison covers QLoRA, BitFit, Prefix Tuning, and IA3.

The web app is designed as both an analysis dashboard and an interactive playground:
it displays benchmark metrics such as exact match, SQL validity, execution accuracy,
latency, BLEU, ROUGE-L, and Token F1, while also allowing users to generate SQL from
selected WikiSQL natural-language questions using either base models or available
fine-tuned PEFT adapters.

## Academic Background

This project is based on the work carried out by Antoni Carrasco Martinez and Ivan
Molera Gomez for their Master's Final Project in Artificial Intelligence at UNIR. The
original work uses a different set of base models, while this repository adapts the
experimental direction to small Hugging Face models and a web-based PEFT comparison
lab for NL-to-SQL.

Original Master's Final Project record:
https://reunir.unir.net/handle/123456789/19416

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

QLoRA requires a CUDA-capable environment for `bitsandbytes` 4-bit training. The scripts
fail fast with a clear error if CUDA is not available.

## Download Hugging Face Assets

The selected models and WikiSQL dataset are declared in `configs/hf_assets.yaml`.

Download them into the current machine's Hugging Face cache:

```bash
python scripts/download_hf_assets.py --config configs/hf_assets.yaml
```

The script downloads:

- `google-t5/t5-small`
- `openai-community/gpt2`
- `HuggingFaceTB/SmolLM2-135M-Instruct`
- `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- `Salesforce/wikisql` splits: `train`, `validation`, `test`

It also writes a manifest to `$HF_HOME/asset_manifest.json`.

You can also download only one asset family:

```bash
python scripts/download_hf_assets.py --config configs/hf_assets.yaml --only models
python scripts/download_hf_assets.py --config configs/hf_assets.yaml --only datasets
```

## Docker Images

The project intentionally separates the heavy ML runtime from the lightweight web
dashboard. This keeps normal UI iterations fast and avoids rebuilding or redownloading
model assets when only benchmark presentation, copy, or frontend behavior changes.

The active Google Cloud images are:

| Image | Dockerfile | Purpose | Rebuild when |
| --- | --- | --- | --- |
| `europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/runtime:latest` | `Dockerfile.runtime` | Heavy CUDA/Python ML base image. It installs the ML stack and caches the selected Hugging Face models and WikiSQL dataset under `/opt/hf-cache`. | CUDA/Python base, ML dependencies, selected model list, dataset cache logic, or Hugging Face asset preload changes. |
| `europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/train:latest` | `Dockerfile.train` | Vertex AI training image. It inherits from the runtime image and adds this repository's training scripts, benchmark runners, configs, and GCS artifact upload helper. | Training code, benchmark code, experiment configs, or pipeline wrapper scripts change. |
| `europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/ml-api:latest` | `Dockerfile.ml-api` | Cloud Run inference service for the WikiSQL playground. It inherits from the runtime image and loads base models plus available PEFT adapters through `peft_lab.ml_api`. | Live inference code, prompt generation, model loading, adapter loading, or packaged adapter artifacts change. |
| `europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/app:latest` | `Dockerfile.app` | Lightweight Cloud Run web app. It serves the dashboard, static UI, benchmark JSON files, runtime metadata, and proxy endpoints that call the ML API. | Frontend code, dashboard formatting, benchmark JSON files, web API proxy behavior, or README/package metadata used by the web image changes. |

There is also a legacy monolithic image:

| Image | Dockerfile | Status |
| --- | --- | --- |
| `europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/ml:latest` | `Dockerfile.ml` | Kept for compatibility only. It combines dependencies, cached assets, code, web files, and benchmark results in one image, so it is no longer the recommended deployment path. |

### Build Commands

Rebuild the runtime image only when dependencies, selected models, dataset cache logic,
or the CUDA/Python base image change:

```bash
gcloud builds submit \
  --config=cloudbuild.runtime.yaml \
  --project=nl-sql-peft-lab-ivan-0429
```

Rebuild the training image when training code, benchmark runners, configs, or Vertex AI
pipeline wrappers change:

```bash
gcloud builds submit \
  --config=cloudbuild.train.yaml \
  --project=nl-sql-peft-lab-ivan-0429
```

Rebuild the ML API image when live inference code, model loading, prompt generation,
or adapter-loading logic changes:

```bash
gcloud builds submit \
  --config=cloudbuild.ml-api.yaml \
  --project=nl-sql-peft-lab-ivan-0429
```

Rebuild the web app image for normal dashboard iteration over `web/`, benchmark JSON,
result formatting, copy, layout, and the lightweight API proxy:

```bash
gcloud builds submit \
  --config=cloudbuild.app.yaml \
  --project=nl-sql-peft-lab-ivan-0429
```

### Cloud Run Services

Deploy the ML API image to Cloud Run:

```bash
gcloud run deploy nl-to-sql-peft-lab-ml \
  --image=europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/ml-api:latest \
  --region=europe-west1 \
  --project=nl-sql-peft-lab-ivan-0429 \
  --platform=managed \
  --allow-unauthenticated \
  --memory=8Gi \
  --cpu=4 \
  --timeout=900 \
  --concurrency=1 \
  --max-instances=2 \
  --port=8080
```

Deploy the web app image to Cloud Run and point it at the ML API URL:

```bash
gcloud run deploy nl-to-sql-peft-lab \
  --image=europe-west1-docker.pkg.dev/nl-sql-peft-lab-ivan-0429/nl-to-sql-peft-lab/app:latest \
  --region=europe-west1 \
  --project=nl-sql-peft-lab-ivan-0429 \
  --platform=managed \
  --allow-unauthenticated \
  --memory=512Mi \
  --cpu=1 \
  --timeout=300 \
  --concurrency=20 \
  --max-instances=5 \
  --port=8080 \
  --set-env-vars=ML_API_BASE_URL=https://<ml-api-service-url>,ML_API_TIMEOUT_SECONDS=900
```

The web app service does not load Hugging Face models. Its `/api/examples` and
`/api/generate` routes proxy to the ML API service so the browser can keep using same
origin API paths.

### Local Docker Testing

For local testing, build the runtime once, then build the ML API and web app images:

```bash
docker build -f Dockerfile.runtime -t nl-to-sql-peft-lab-runtime .
docker build -f Dockerfile.ml-api \
  --build-arg RUNTIME_IMAGE=nl-to-sql-peft-lab-runtime \
  -t nl-to-sql-peft-lab-ml-api .
docker build -f Dockerfile.app \
  -t nl-to-sql-peft-lab-app .
```

Run both services locally:

```bash
docker run --rm -p 8081:8080 nl-to-sql-peft-lab-ml-api
docker run --rm -p 8080:8080 \
  -e ML_API_BASE_URL=http://host.docker.internal:8081 \
  nl-to-sql-peft-lab-app
```

On Apple Silicon this builds `arm64` images locally. For Google Cloud GPU/Cloud Run,
build on Cloud Build or explicitly target `linux/amd64`.

The runtime image uses:

- CUDA 12.1 runtime with `torch==2.4.1+cu121`, pinned for Vertex AI T4 driver compatibility.
- `HF_HOME=/opt/hf-cache`
- `TRANSFORMERS_CACHE=/opt/hf-cache/hub`
- `HF_DATASETS_CACHE=/opt/hf-cache/datasets`

This split avoids downloading model weights or regenerating the dataset cache every time
the dashboard layout changes, and it also keeps normal UI deployments independent from
the large ML runtime image.

## Smoke Training Run

The default config intentionally uses a small subset so the pipeline can be tested before
launching longer experiments.

```bash
python -m peft_lab.train_t5_qlora --config configs/t5_small_wikisql_qlora.yaml
```

Outputs are written under `outputs/t5-small-wikisql-qlora/`.

## Zero-Shot Baseline

```bash
python -m peft_lab.evaluate_zero_shot --config configs/zero_shot_wikisql_baseline.yaml
```

This evaluates:

- `google-t5/t5-small`
- `HuggingFaceTB/SmolLM2-135M-Instruct`
- `Qwen/Qwen2.5-Coder-0.5B-Instruct`

The generated JSON is written to `outputs/baselines/zero_shot_wikisql.json` and is shaped
for a future dashboard: leaderboard metrics, latency, SQL validity, execution accuracy,
and sample predictions.

For UI work before the real models are evaluated, use
`sample_results/zero_shot_wikisql.demo.json`. Its values are placeholders, but the schema
matches the evaluator output.

For a quick local check, reduce the workload:

```bash
python -m peft_lab.evaluate_zero_shot \
  --config configs/zero_shot_wikisql_baseline.yaml \
  --max-examples 8
```

## Zero-Shot Benchmark

Use the benchmark script for the comparative baseline that feeds the dashboard:

```bash
python -m peft_lab.benchmark_zero_shot --config configs/zero_shot_wikisql_benchmark.yaml
```

The current runner is a project-specific benchmark script so the first dashboard
can capture NL-to-SQL metrics that are not standard multiple-choice metrics.
The planned framework integration is:

- Primary: [Hugging Face LightEval](https://huggingface.co/docs/lighteval/en/index), using a custom `wikisql_nl_to_sql` task and custom metrics.
- Research baseline alternative: [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), also with a custom WikiSQL task.
- Heavier suite alternatives for later comparison: [HELM](https://github.com/stanford-crfm/helm) and [OpenCompass](https://github.com/open-compass/opencompass).

Recommended sample sizes:

- `30-50` examples for smoke tests.
- `300` examples for fast iteration.
- `1000` examples for a stable portfolio comparison.
- Full validation split (`8421`) only for a final, publication-style run.

The benchmark records per-example predictions and aggregated metrics:

- Exact match
- BLEU
- ROUGE-L
- Token F1
- SQL validity
- Execution accuracy
- Failure, empty-output, and non-SQL rates
- Model load time
- Mean, p50, and p95 generation latency
- Evaluation latency
- Throughput
- Runtime metadata: Docker image, platform, Python, PyTorch, CUDA/device, CPU and RAM.
- Benchmark metadata: task, dataset split, sample size, calls per model, total model calls, sampling seed, prompt length, generation limits, and metric definitions.

It writes one file per model plus an index consumed by the web app:

```text
benchmark_results/zero_shot/zero_shot_wikisql_t5-small.json
benchmark_results/zero_shot/zero_shot_wikisql_gpt2.json
benchmark_results/zero_shot/zero_shot_wikisql_smollm2-135m-instruct.json
benchmark_results/zero_shot/zero_shot_wikisql_qwen2.5-coder-0.5b-instruct.json
benchmark_results/zero_shot/zero_shot_wikisql_index.json
```

Run a single model:

```bash
python -m peft_lab.benchmark_zero_shot \
  --config configs/zero_shot_wikisql_benchmark.yaml \
  --model-id t5-small
```

## Web App

Run the interactive baseline app from the ML image:

```bash
docker run --rm -p 8080:8080 nl-to-sql-peft-lab-app
```

Then open http://localhost:8080.

The app provides:

- A result selector for `zero-shot`, `QLoRA`, `BitFit`, `Prefix Tuning`, and `IA3` benchmark runs.
- A leaderboard for the selected benchmark metrics.
- Benchmark charts for execution accuracy, SQL validity, and latency.
- Benchmark details: sample size, calls per model, total calls, dataset split, seed, and generation limits.
- Runtime reproducibility details for the Docker or Cloud Run environment.
- A WikiSQL playground where users choose a validation example, select a model, generate SQL, and compare it with the expected WikiSQL SQL.

The first generation request for each model loads that model into memory in the ML API
service, so it can take longer than later requests. The web app reads real benchmark results from
`benchmark_results/<mode>/*_wikisql_index.json`. For the zero-shot dashboard it
falls back to `sample_results/zero_shot_wikisql.demo.json` if no real run exists.

## Project Shape

```text
configs/                 Experiment configurations
web/                     Static frontend served by FastAPI
src/peft_lab/            Training, data preparation, SQL rendering, metrics
benchmark_results/       Benchmark JSON files consumed by the dashboard
outputs/                 Local model/adapters/results output directory
```

## Current Experiment Matrix

Models:

- [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small): T5-small, 60M parameters, encoder-decoder text-to-text baseline.
- [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct): 135M parameter decoder-only instruct model.
- [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct): 0.49B parameter decoder-only code model.

Dataset:

- [`Salesforce/wikisql`](https://huggingface.co/datasets/Salesforce/wikisql): WikiSQL text-to-SQL dataset with 80,654 annotated question/query examples over 24,241 Wikipedia tables.

PEFT techniques:

- QLoRA: 4-bit quantization plus LoRA adapters, implemented with `bitsandbytes`, `transformers`, and `peft`.
- BitFit: bias-only fine-tuning baseline.
- Prefix Tuning: trainable virtual prefix vectors while freezing the base model.
- IA3: learned activation scaling vectors over attention/feed-forward activations.

## References

- Carrasco Martinez, Antoni, and Molera Gomez, Ivan. Master's Final Project in
  Artificial Intelligence, UNIR: https://reunir.unir.net/handle/123456789/19416
- T5-small model card: https://huggingface.co/google-t5/t5-small
- SmolLM2-135M-Instruct model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- Qwen2.5-Coder-0.5B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct
- WikiSQL dataset card: https://huggingface.co/datasets/Salesforce/wikisql
- Hugging Face PEFT documentation: https://huggingface.co/docs/peft/index
- PEFT quantization and QLoRA guide: https://huggingface.co/docs/peft/developer_guides/quantization
- Transformers bitsandbytes / QLoRA documentation: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
- BitFit paper: https://huggingface.co/papers/2106.10199
- Prefix Tuning reference: https://huggingface.co/docs/peft/main/en/package_reference/prefix_tuning
- IA3 task guide: https://huggingface.co/docs/peft/task_guides/ia3
