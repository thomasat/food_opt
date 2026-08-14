#!/bin/bash
# Regenerates desktop/requirements.lock.txt from the repo's requirements.txt
# (excluding pytest — a dev-only dependency). Run on an Apple Silicon Mac
# whenever requirements.txt changes, and commit the result.
set -euo pipefail

DESKTOP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$DESKTOP_DIR")"
UV_BIN="$("$DESKTOP_DIR/fetch_uv.sh")"

# requirements.txt carries Linux-deploy specifics that don't exist on macOS:
# the +cpu torch wheel variant (and its extra index) is Linux/Windows only —
# macOS arm64 uses the plain same-version wheel from PyPI, which is CPU-only
# anyway. pytest is a dev-only dependency.
sed -e '/^--extra-index-url/d' \
    -e 's/^torch==\([0-9.[:alnum:]]*\)+cpu$/torch==\1/' \
    -e '/^pytest/d' \
    "$REPO_DIR/requirements.txt" | "$UV_BIN" pip compile - \
  --python-version 3.13 \
  --generate-hashes \
  -o "$DESKTOP_DIR/requirements.lock.txt"

echo "Wrote $DESKTOP_DIR/requirements.lock.txt"
