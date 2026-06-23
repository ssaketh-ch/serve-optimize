#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src tests
pytest -q
ruff check .
python -m json.tool feature_list.json >/tmp/serve_optimize_feature_list.json
serve-optimize --help >/tmp/serve_optimize_help.txt
serve-optimize optimize --help >/tmp/serve_optimize_optimize_help.txt
serve-optimize validate-campaign --help >/tmp/serve_optimize_campaign_help.txt
serve-optimize campaign-plan --help >/tmp/serve_optimize_campaign_plan_help.txt
