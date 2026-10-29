#!/bin/bash
# Compiles both dependency locks from the repo's requirements.txt (pytest,
# a dev-only dependency, excluded):
#   desktop/requirements.lock.txt          macOS arm64, what launcher.sh installs
#   desktop/requirements-windows.lock.txt  Windows x64, constrained to the Mac lock
# Run on an Apple Silicon Mac whenever requirements.txt changes, and commit
# both. The Mac lock is replaced only when its package or hash lines change:
# launcher.sh hashes the whole file, so any byte change makes every Mac user
# redo the ~1 GB online setup.
# Test seams: FOODOPT_TEST_PLATFORM ("<uname -s> <uname -m>") and
# FOODOPT_TEST_UV (the uv binary to run).
set -euo pipefail

DESKTOP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$DESKTOP_DIR")"
MAC_LOCK="$DESKTOP_DIR/requirements.lock.txt"
WIN_LOCK="$DESKTOP_DIR/requirements-windows.lock.txt"
COMPILE_CMD="./desktop/update_lock.sh"

# The Mac compile resolves for the machine it runs on, so it must be the
# machine the app runs on.
PLATFORM="${FOODOPT_TEST_PLATFORM:-$(uname -s) $(uname -m)}"
if [ "$PLATFORM" != "Darwin arm64" ]; then
  echo "refusing: compile the locks on an Apple Silicon Mac (this is $PLATFORM)" >&2
  exit 2
fi

UV_BIN="${FOODOPT_TEST_UV:-$("$DESKTOP_DIR/fetch_uv.sh")}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP" "$MAC_LOCK.new" "$WIN_LOCK.new"' EXIT

# requirements.txt carries Linux-deploy specifics: the +cpu torch wheel
# variant (and its extra index) exists only on PyTorch's index. PyPI's macOS
# arm64 and Windows torch wheels of the same version are CPU-only already.
requirements() {
  sed -e '/^--extra-index-url/d' \
      -e 's/^torch==\([0-9.[:alnum:]]*\)+cpu$/torch==\1/' \
      -e '/^pytest/d' \
      "$REPO_DIR/requirements.txt"
}

# Each compile starts from a copy of the committed lock, so uv keeps today's
# pins as preferences and moves only what requirements.txt makes it move.
cp "$MAC_LOCK" "$TMP/mac.lock"
requirements | "$UV_BIN" pip compile - \
  --python-version 3.13 \
  --generate-hashes \
  --custom-compile-command "$COMPILE_CMD" \
  -o "$TMP/mac.lock"

# Windows: the Mac lock as it will be committed, plus what only Windows needs.
if [ -f "$WIN_LOCK" ]; then cp "$WIN_LOCK" "$TMP/windows.lock"; fi
requirements | "$UV_BIN" pip compile - \
  --python-platform x86_64-pc-windows-msvc \
  --python-version 3.13 \
  --generate-hashes \
  -c "$TMP/mac.lock" \
  --custom-compile-command "$COMPILE_CMD" \
  -o "$TMP/windows.lock"
# uv names the -c file in its "# via" lines: name the committed Mac lock,
# not this run's temporary copy of it.
sed -e "s|$TMP/mac.lock|desktop/requirements.lock.txt|g" \
  "$TMP/windows.lock" > "$TMP/windows.named.lock"

# Both compiles succeeded. Each changed lock is first staged beside its
# committed file as <lock>.new; only when both are staged is either moved into
# place, so a failed write leaves both committed locks as they were.
body() {  # the package and hash lines; comments and the header don't count
  grep -E '^[A-Za-z0-9]|^[[:space:]]+--hash=' "$1" || true
}
packages() {  # the package lines only
  grep -E '^[A-Za-z0-9]' "$1" || true
}
MAC_CHANGED=0
STAGED=()
MESSAGES=()
install_lock() {  # install_lock <compiled> <committed> <Mac|Windows>; stages only
  local name="desktop/${2##*/}"
  if [ ! -f "$2" ]; then
    cp "$1" "$2.new"
    STAGED+=("$2")
    MESSAGES+=("Wrote $name (new)")
  elif [ "$(body "$1")" = "$(body "$2")" ]; then
    MESSAGES+=("$3 lock unchanged (same packages and hashes): $name")
  else
    cp "$1" "$2.new"
    STAGED+=("$2")
    if [ "$(packages "$1")" = "$(packages "$2")" ]; then
      MESSAGES+=("Wrote $name (hashes changed)")
    else
      MESSAGES+=("Wrote $name (packages changed)")
    fi
    if [ "$3" = "Mac" ]; then MAC_CHANGED=1; fi
  fi
}
install_lock "$TMP/mac.lock" "$MAC_LOCK" Mac
install_lock "$TMP/windows.named.lock" "$WIN_LOCK" Windows
for lock in ${STAGED[@]+"${STAGED[@]}"}; do
  mv -f "$lock.new" "$lock"
done
printf '%s\n' "${MESSAGES[@]}"
if [ "$MAC_CHANGED" = "1" ]; then
  echo "Mac lock changed: every Mac user will redo the online setup; re-measure EXPECTED_SETUP_MB in launcher.sh"
fi
