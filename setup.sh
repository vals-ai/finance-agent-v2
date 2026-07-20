#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

command -v fabv2 >/dev/null
command -v uv >/dev/null
test -x /app/.venv/bin/python

uv pip install --python /app/.venv/bin/python --no-deps --reinstall "$SCRIPT_DIR"
mkdir -p /app/results
