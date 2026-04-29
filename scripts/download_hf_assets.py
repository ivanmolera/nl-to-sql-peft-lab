"""Download Hugging Face models and datasets into the configured HF cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download


MODEL_ALLOW_PATTERNS = [
    "*.bin",
    "*.json",
    "*.model",
    "*.py",
    "*.safetensors",
    "*.txt",
    "README.md",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.*",
    "tokenizer_config.json",
    "vocab.*",
]

MODEL_IGNORE_PATTERNS = [
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.ot",
    "*.tflite",
]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    manifest_path = hf_home / "asset_manifest.json"
    manifest: dict[str, Any] = load_manifest(manifest_path, hf_home)

    if args.only in {"all", "models"}:
        for model in config.get("models", []):
            print(f"Downloading model: {model['name']}")
            local_path = snapshot_download(
                repo_id=model["name"],
                allow_patterns=MODEL_ALLOW_PATTERNS,
                ignore_patterns=MODEL_IGNORE_PATTERNS,
            )
            manifest["models"].append(
                {
                    "id": model["id"],
                    "name": model["name"],
                    "kind": model["kind"],
                    "local_path": local_path,
                }
            )

    if args.only in {"all", "datasets"}:
        for dataset in config.get("datasets", []):
            for split in dataset.get("splits", []):
                print(f"Downloading dataset: {dataset['name']} [{split}]")
                loaded = load_dataset(
                    dataset["name"],
                    split=split,
                    trust_remote_code=True,
                )
                manifest["datasets"].append(
                    {
                        "name": dataset["name"],
                        "split": split,
                        "rows": len(loaded),
                    }
                )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved asset manifest to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML file with HF assets.")
    parser.add_argument(
        "--only",
        choices=["all", "models", "datasets"],
        default="all",
        help="Limit the download to models or datasets.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_manifest(path: Path, hf_home: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "hf_home": str(hf_home),
        "models": [],
        "datasets": [],
    }


if __name__ == "__main__":
    main()
