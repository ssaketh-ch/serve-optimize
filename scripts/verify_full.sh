#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src tests
pytest -q
ruff check .
python -m json.tool feature_list.json >/tmp/serve_optimize_feature_list.json
serve-optimize --help >/tmp/serve_optimize_help.txt
serve-optimize managed-evaluate --help >/tmp/serve_optimize_managed_help.txt
serve-optimize validate-campaign --help >/tmp/serve_optimize_campaign_help.txt
serve-optimize release-check --help >/tmp/serve_optimize_release_check_help.txt
serve-optimize research-package --help >/tmp/serve_optimize_research_package_help.txt
serve-optimize release-check --out results/release-check
