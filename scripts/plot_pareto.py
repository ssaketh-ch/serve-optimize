#!/usr/bin/env python
"""Plot a throughput-energy Pareto scatter from Serve Optimize JSONL results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=Path("pareto.png"))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install plotting dependencies with: pip install -e '.[plot]'") from exc

    rows = [json.loads(line) for line in args.results_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No rows found.")

    x = [row["joules_per_token"] for row in rows]
    y = [row["throughput_tok_s"] for row in rows]
    labels = [row["config"]["quantization"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    for quant in sorted(set(labels)):
        indices = [index for index, label in enumerate(labels) if label == quant]
        ax.scatter([x[index] for index in indices], [y[index] for index in indices], label=quant, alpha=0.8)
    ax.set_xlabel("Joules per token")
    ax.set_ylabel("Throughput tokens/sec")
    ax.set_title("Serve Optimize throughput-energy scatter")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

