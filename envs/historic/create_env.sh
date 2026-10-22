#!/usr/bin/env bash
# Build the historic TribChem environment (see envs/README.md).
#
# Usage:  bash envs/historic/create_env.sh <TribChem source root>
#   <TribChem source root> is the folder that contains setup.py.
# Optional variables: UV (uv executable), VENV, COPY_DIR.
#
# The TribChem sources are never written: they are copied to COPY_DIR, the
# copy is checked against tribchem-source.sha256 and installed from there.
set -euo pipefail

SRC=${1:?usage: create_env.sh <TribChem source root, the folder with setup.py>}
SRC=$(cd "$SRC" && pwd)
UV=${UV:-uv}
VENV=${VENV:-$HOME/.venvs/tribchem-storico}
COPY_DIR=${COPY_DIR:-$HOME/tribchem-storico/src/$(basename "$SRC")}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

manifest() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum); }

for path in "$VENV" "$COPY_DIR"; do
    if [ -e "$path" ]; then
        echo "error: $path already exists, remove it first" >&2
        exit 1
    fi
done

# 1. Copy of the sources, checked against the recorded version.
mkdir -p "$(dirname "$COPY_DIR")"
cp -a "$SRC/." "$COPY_DIR/"
if ! diff -q <(manifest "$COPY_DIR") "$HERE/tribchem-source.sha256" > /dev/null; then
    echo "error: the TribChem sources differ from tribchem-source.sha256" >&2
    exit 1
fi

# 2. Python 3.12.4, as pinned by TribChem's requirements.txt, and the locked packages.
"$UV" python install 3.12.4
"$UV" venv "$VENV" --managed-python --python 3.12.4 --prompt tribchem-storico
"$UV" pip sync --python "$VENV/bin/python" --require-hashes "$HERE/requirements.lock"

# 3. The atomate change of TribChem's Installation-Guide.md ("Fixing" some issues in atomate).
SITE=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
patch -p1 --forward --no-backup-if-mismatch -d "$SITE" < "$HERE/atomate-1.1.0-vdw-kernel.patch"

# 4. TribChem, installed as in the guide: pip install -e .
(cd "$COPY_DIR" && "$VENV/bin/python" -m pip install \
    --disable-pip-version-check --no-index --no-deps --no-build-isolation -e .)
"$VENV/bin/python" -m pip check

echo "historic environment ready: $VENV (TribChem from $COPY_DIR)"
