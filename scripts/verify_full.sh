#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="$(mktemp -d /tmp/serve-optimize-dist.XXXXXX)"
trap 'rm -rf "$DIST_DIR"' EXIT

python -m compileall -q src tests
pytest -q
ruff check .
python -m build --no-isolation --outdir "$DIST_DIR" .
WHEEL_PATH="$(find "$DIST_DIR" -maxdepth 1 -name 'serve_optimize-*.whl' -print -quit)"
python -m zipfile -e "$WHEEL_PATH" "$DIST_DIR/site"
PYTHONPATH="$DIST_DIR/site" python -m serve_optimize --version
python -m json.tool feature_list.json >/tmp/serve_optimize_feature_list.json
serve-optimize --help >/tmp/serve_optimize_help.txt
serve-optimize managed-evaluate --help >/tmp/serve_optimize_managed_help.txt
serve-optimize validate-campaign --help >/tmp/serve_optimize_campaign_help.txt
serve-optimize campaign-plan --help >/tmp/serve_optimize_campaign_plan_help.txt
serve-optimize release-check --help >/tmp/serve_optimize_release_check_help.txt
serve-optimize research-package --help >/tmp/serve_optimize_research_package_help.txt
serve-optimize release-check --out results/release-check
