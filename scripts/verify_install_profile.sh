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

python -m venv "$target"
"$target/bin/python" -m pip install --upgrade "pip==26.1.2"
cd "$repo_root"
"$target/bin/python" -m pip install -r "$requirements_file"

if [[ "$profile" == "sglang" ]]; then
  source "$repo_root/scripts/env_base_runtime.sh"
fi

"$target/bin/python" -m pip check
"$target/bin/serve-optimize" doctor --profile "$profile"
