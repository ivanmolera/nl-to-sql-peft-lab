# NL-to-SQL PEFT Lab

Comparativa de tecnicas PEFT para NL-to-SQL en modelos generativos pequenos.

This repository starts with a minimal, reproducible pipeline for:

- Model: `google-t5/t5-small`
- Dataset: `Salesforce/wikisql`
- Technique: QLoRA, implemented as 4-bit quantization plus LoRA adapters

The first goal is to validate the full loop: load WikiSQL, build canonical SQL targets,
fine-tune with PEFT, evaluate exact-match, and save the adapter/checkpoint artifacts.

Before fine-tuning, the project can also run a zero-shot baseline over the selected
models. That lets the web app show the comparative dashboard and example-level analysis
without waiting for PEFT jobs.

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
- `HuggingFaceTB/SmolLM2-135M-Instruct`
- `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- `Salesforce/wikisql` splits: `train`, `validation`, `test`

It also writes a manifest to `$HF_HOME/asset_manifest.json`.

You can also download only one asset family:

```bash
python scripts/download_hf_assets.py --config configs/hf_assets.yaml --only models
python scripts/download_hf_assets.py --config configs/hf_assets.yaml --only datasets
```

## Docker Image With Cached Models

For Google Cloud, the recommended shape is:

- A lightweight web/API image for the app.
- A separate ML/inference image with the Hugging Face assets predownloaded.

This repository includes `Dockerfile.ml`, which installs the Python stack and runs the
asset downloader during image build. This avoids downloading model weights at Cloud Run
startup time.

```bash
docker build -f Dockerfile.ml -t nl-to-sql-peft-lab-ml .
```

On Apple Silicon this builds an `arm64` image. For Google Cloud GPU/Cloud Run, build on
Cloud Build or explicitly target `linux/amd64`:

```bash
docker build --platform linux/amd64 -f Dockerfile.ml -t nl-to-sql-peft-lab-ml .
```

The image uses:

- `HF_HOME=/opt/hf-cache`
- `TRANSFORMERS_CACHE=/opt/hf-cache/hub`
- `HF_DATASETS_CACHE=/opt/hf-cache/datasets`

For production on Google Cloud, build this image with Cloud Build and push it to Artifact
Registry. The future web app can call this ML service or read precomputed results from a
database/object storage.

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

## Web App

Run the interactive baseline app from the ML image:

```bash
docker run --rm -p 8080:8080 nl-to-sql-peft-lab-ml
```

Then open http://localhost:8080.

The app provides:

- A leaderboard for the zero-shot baseline metrics.
- Benchmark charts for execution accuracy, SQL validity, and latency.
- A WikiSQL playground where users choose a validation example, select a model, generate SQL, and compare it with the expected WikiSQL SQL.

The first generation request for each model loads that model into memory, so it can take
longer than later requests. The app reads real benchmark results from
`outputs/baselines/zero_shot_wikisql.json` when present, otherwise it uses
`sample_results/zero_shot_wikisql.demo.json`.

## Project Shape

```text
configs/                 Experiment configurations
web/                     Static frontend served by FastAPI
src/peft_lab/            Training, data preparation, SQL rendering, metrics
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
