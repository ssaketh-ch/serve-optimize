#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PROFILE TARGET_VENV" >&2
  exit 2
fi

profile=$1
target=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
requirements_file="$repo_root/requirements/profiles/$profile.txt"

if [[ ! -f "$requirements_file" ]]; then
  echo "unknown installation profile: $profile" >&2
  exit 2
fi

if [[ -e "$target" ]]; then
  echo "target already exists: $target" >&2
  exit 2
fi

if [[ "$profile" == "vllm" || "$profile" == "sglang" ]]; then
  if ! "${PYTHON:-python3}" - <<'PY'
import pathlib
import os
import sysconfig
paths = [pathlib.Path(sysconfig.get_path("include"))]
paths.extend(pathlib.Path(item) for item in os.environ.get("C_INCLUDE_PATH", "").split(os.pathsep) if item)
raise SystemExit(0 if any((path / "Python.h").is_file() for path in paths) else 1)
PY
  then
    echo "Python development headers are required. On Ubuntu, install python3-dev and build-essential." >&2
    exit 2
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/." >&2
  exit 2
fi

uv venv --python "${PYTHON:-python3}" "$target"
cd "$repo_root"
if [[ "$profile" == "vllm" ]]; then
  uv pip install --python "$target/bin/python" --torch-backend=auto -r "$requirements_file"
elif [[ "$profile" == "sglang" ]]; then
  uv pip install --python "$target/bin/python" -r "$requirements_file"
else
  uv pip install --python "$target/bin/python" -r "$requirements_file"
fi

uv pip check --python "$target/bin/python"
"$target/bin/serve-optimize" doctor --profile "$profile"
