#!/usr/bin/env bash
# Build or update the development environment of TriboWorkFlow (see envs/README.md).
#
# Usage:  bash envs/dev/create_env.sh
# Optional variables: UV (uv executable), VENV, PYTHON (interpreter for a new venv).
#
# The venv is synchronised with the lock file: packages that are not in the
# lock are removed. TriboWorkFlow is then installed in editable mode.
set -euo pipefail

UV=${UV:-uv}
VENV=${VENV:-$HOME/.venvs/tribchem}
PYTHON=${PYTHON:-python3.12}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON" -m venv "$VENV"
fi
"$UV" pip sync --python "$VENV/bin/python" --require-hashes "$HERE/requirements.lock"
"$UV" pip install --python "$VENV/bin/python" --no-deps -e "$REPO"
"$UV" pip check --python "$VENV/bin/python"

echo "development environment ready: $VENV"
