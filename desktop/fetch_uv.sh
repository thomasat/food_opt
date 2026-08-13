#!/bin/bash
# Downloads the pinned uv binary (macOS arm64) into desktop/.cache/, verifies
# its SHA256, and prints the binary's absolute path on stdout. Idempotent.
set -euo pipefail

UV_VERSION="0.12.3"
UV_SHA256="546f7f8a6c70ff13a3a9d2bc958db3427298cebf3e0cb756f9177133b7068843"   # sha256 of uv-aarch64-apple-darwin.tar.gz

DESKTOP_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE_DIR="$DESKTOP_DIR/.cache"
UV_BIN="$CACHE_DIR/uv-$UV_VERSION"

if [ ! -x "$UV_BIN" ]; then
  mkdir -p "$CACHE_DIR"
  TARBALL="$CACHE_DIR/uv-$UV_VERSION.tar.gz"
  URL="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-apple-darwin.tar.gz"
  echo "downloading $URL" >&2
  curl -fsSL -o "$TARBALL" "$URL"
  echo "$UV_SHA256  $TARBALL" | shasum -a 256 -c - >&2
  tar -xzf "$TARBALL" -C "$CACHE_DIR"
  mv "$CACHE_DIR/uv-aarch64-apple-darwin/uv" "$UV_BIN"
  rm -rf "$CACHE_DIR/uv-aarch64-apple-darwin" "$TARBALL"
  chmod 755 "$UV_BIN"
  # Record the extracted binary's own hash so later runs can re-verify the
  # cached copy without re-downloading (the pinned hash above is the tarball's).
  shasum -a 256 "$UV_BIN" | awk '{print $1}' > "$UV_BIN.sha256"
fi

# Re-verify the cached binary every run: a corrupted/tampered cache entry —
# or one missing its recorded hash — becomes a cache miss instead of being
# trusted forever.
if ! echo "$(cat "$UV_BIN.sha256" 2>/dev/null)  $UV_BIN" | shasum -a 256 -c - >/dev/null 2>&1; then
  echo "cached uv failed verification; removing it - re-run to re-download" >&2
  rm -f "$UV_BIN" "$UV_BIN.sha256"
  exit 1
fi

echo "$UV_BIN"
