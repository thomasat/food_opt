#!/bin/bash
# Regenerates desktop/requirements.lock.txt from the repo's requirements.txt
# (excluding pytest — a dev-only dependency). Run on an Apple Silicon Mac
# whenever requirements.txt changes, and commit the result.
set -euo pipefail

DESKTOP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$DESKTOP_DIR")"
UV_BIN="$("$DESKTOP_DIR/fetch_uv.sh")"

grep -v '^pytest' "$REPO_DIR/requirements.txt" | "$UV_BIN" pip compile - \
  --python-version 3.13 \
  -o "$DESKTOP_DIR/requirements.lock.txt"

echo "Wrote $DESKTOP_DIR/requirements.lock.txt"
