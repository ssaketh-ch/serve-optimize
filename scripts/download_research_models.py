"""Download the pinned publication model snapshots into a Hugging Face cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = (
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"),
    ("ibm-granite/granite-3.3-2b-instruct", "707f574c62054322f6b5b04b6d075f0a8f05e0f0"),
    ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"),
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    ("ibm-granite/granite-3.3-8b-instruct", "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"),
    ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
    ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    ("Qwen/Qwen3.8-27B-FP8", "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"),
    ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    for model, revision in MODELS:
        print(f"START {model} {revision}", flush=True)
        path = snapshot_download(
            repo_id=model,
            revision=revision,
            cache_dir=args.cache_dir,
            max_workers=8,
        )
        print(f"DONE {model} {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
