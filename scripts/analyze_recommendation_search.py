from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from serve_optimize.research_analysis import compare_equal_budget, load_measured_pool, replay_ablations


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, default=20)
    args = parser.parse_args()
    if args.seed_count < 1:
        raise SystemExit("--seed-count must be at least one.")
    run_dirs = sorted({path.parent for root in args.roots for path in root.rglob("managed_run.json")})
    pools = [load_measured_pool(path) for path in run_dirs]
    seeds = range(args.seed_count)
    baselines = [row for pool in pools for row in compare_equal_budget(pool, seeds=seeds)]
    ablations = [row for pool in pools for row in replay_ablations(pool, seeds=seeds)]
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "equal_budget_baselines.csv", baselines)
    _write_csv(args.out / "recommendation_ablations.csv", ablations)
    payload = {
        "schema_version": "recommendation-search-analysis/v1",
        "run_count": len(pools),
        "seed_count": args.seed_count,
        "baseline_row_count": len(baselines),
        "ablation_row_count": len(ablations),
        "oracle_policy": "post_hoc_only",
        "replay_scope": "fixed_measured_candidate_pool",
        "unsupported_ablation_count": sum(str(row["status"]).startswith("unsupported") for row in ablations),
    }
    (args.out / "recommendation_search_analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
