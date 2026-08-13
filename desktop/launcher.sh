#!/bin/bash
# Food Optimizer — headless server manager.
# Run by the FoodOptimizer wrapper binary (Contents/MacOS/FoodOptimizer),
# which owns all user-facing UI. This script only manages the environment
# and the Streamlit server; it never shows dialogs or opens a browser.
# First run: provisions a per-user Python + dependencies using the bundled uv.
# Every run: backs up projects, then serves the app on localhost only.
#
# Exit codes the wrapper maps to friendly pages:
#   2 = unsupported machine   3 = setup failed (usually no internet)
set -u

PYTHON_VERSION="3.13.7"   # the single place the Python version is pinned

IDLE_TIMEOUT="${FOODOPT_IDLE_TIMEOUT_SECS:-900}"
ARCH="${FOODOPT_TEST_ARCH:-$(uname -m)}"

say() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

die() {  # $1: log message; $2: exit code (default 1)
  say "FATAL: $1"
  exit "${2:-1}"
}

# ---------- preflight: supported machine (before touching anything) ----------
OS_MAJOR="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)"
if [ "$ARCH" != "arm64" ] || [ "${OS_MAJOR:-0}" -lt 13 ] 2>/dev/null; then
  say "unsupported machine: arch=$ARCH macos=${OS_MAJOR:-unknown}"
  exit 2
fi

# ---------- paths ----------
MACOS_DIR="$(cd "$(dirname "$0")" && pwd)"     # .../Contents/MacOS
CONTENTS_DIR="$(dirname "$MACOS_DIR")"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
APP_BUNDLE="$(dirname "$CONTENTS_DIR")"
UV_BIN="$MACOS_DIR/uv"

SUPPORT_DIR="$HOME/Library/Application Support/FoodOptimizer"
# Home-folder root, NOT ~/Documents: macOS privacy protection (TCC) gates
# Documents/Desktop/Downloads behind a consent prompt and silently denies
# unsigned apps, killing the server at startup (getcwd -> EPERM).
DATA_DIR="$HOME/FoodOptimizer"
VENV_DIR="$SUPPORT_DIR/venv"
MARKER_FILE="$SUPPORT_DIR/setup_complete"
LOG_FILE="$SUPPORT_DIR/launcher.log"
PORT_FILE="$SUPPORT_DIR/server.port"
LOCK_FILE="$RESOURCES_DIR/requirements.lock.txt"

export UV_PYTHON_INSTALL_DIR="$SUPPORT_DIR/python"
export UV_CACHE_DIR="$SUPPORT_DIR/uv-cache"
export PYTHONDONTWRITEBYTECODE=1

# The wrapper offers to copy the app to /Applications in this case; noted
# here for the log (and for tests).
case "$APP_BUNDLE" in
  /Volumes/*|*/AppTranslocation/*) say "running from disk image at $APP_BUNDLE" ;;
esac

# ---------- log (rotate once) ----------
mkdir -p "$SUPPORT_DIR" "$DATA_DIR"
if [ -f "$LOG_FILE" ]; then mv -f "$LOG_FILE" "$LOG_FILE.1"; fi
exec > >(tee -a "$LOG_FILE") 2>&1
say "launcher started (bundle=$APP_BUNDLE)"

# ---------- single instance ----------
if [ -f "$PORT_FILE" ]; then
  read -r OLD_PORT OLD_PID < "$PORT_FILE" || true
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null \
     && curl -fsS --max-time 3 "http://127.0.0.1:${OLD_PORT}/_stcore/health" >/dev/null 2>&1; then
    say "already running on port $OLD_PORT - nothing to do"
    exit 0
  fi
  rm -f "$PORT_FILE"
fi

# ---------- one launch at a time ----------
LOCK_DIR="$SUPPORT_DIR/launch.lock"
OTHER_PID=""
acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo $$ > "$LOCK_DIR/pid"
    return 0
  fi
  OTHER_PID="$(cat "$LOCK_DIR/pid" 2>/dev/null)"
  if [ -n "${OTHER_PID:-}" ] && kill -0 "$OTHER_PID" 2>/dev/null; then
    return 1
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" 2>/dev/null && echo $$ > "$LOCK_DIR/pid"
}
if ! acquire_lock; then
  say "another launch is already in progress (pid ${OTHER_PID:-unknown}) - exiting"
  exit 0
fi
trap 'if [ "$(cat "$LOCK_DIR/pid" 2>/dev/null)" = "$$" ]; then rm -rf "$LOCK_DIR"; fi' EXIT

# ---------- first-run / upgrade setup ----------
LOCK_HASH="$(shasum -a 256 "$LOCK_FILE" | awk '{print $1}')"
NEED_SETUP=1
if [ -x "$VENV_DIR/bin/python" ] && [ -f "$MARKER_FILE" ] \
   && [ "$(head -n 1 "$MARKER_FILE" 2>/dev/null)" = "$LOCK_HASH" ]; then
  NEED_SETUP=0
fi

if [ "$NEED_SETUP" = "1" ]; then
  say "one-time setup starting (downloading software components)"
  rm -f "$MARKER_FILE"
  NET_MSG="setup failed - most likely no internet connection"
  say "installing Python $PYTHON_VERSION"
  "$UV_BIN" python install --no-bin "$PYTHON_VERSION" || die "$NET_MSG" 3
  say "creating environment"
  rm -rf "$VENV_DIR"
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR" || die "$NET_MSG" 3
  say "installing components"
  "$UV_BIN" pip sync --python "$VENV_DIR/bin/python" "$LOCK_FILE" || die "$NET_MSG" 3
  { echo "$LOCK_HASH"; echo "python=$PYTHON_VERSION"; } > "$MARKER_FILE"
  say "setup complete"
fi

# ---------- start the server ----------
PORT="${FOODOPT_PORT_BASE:-8501}"   # override lets tests avoid real-user ports
while lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; do PORT=$((PORT + 1)); done

cd "$DATA_DIR" || die "could not enter data dir $DATA_DIR"

# ---------- safety backups: snapshot all projects on every launch ----------
BACKUP_ROOT="$DATA_DIR/backups"
KEEP_BACKUPS=10
set -- ./*.pkl
if [ -e "$1" ]; then
  BACKUP_DIR="$BACKUP_ROOT/$(date '+%Y-%m-%d_%H%M%S')"
  mkdir -p "$BACKUP_DIR"
  cp ./*.pkl "$BACKUP_DIR/"
  say "projects backed up to $BACKUP_DIR"
  COUNT="$(find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  while [ "$COUNT" -gt "$KEEP_BACKUPS" ]; do
    OLDEST="$(find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)"
    rm -rf "$OLDEST"
    COUNT=$((COUNT - 1))
  done
fi

"$VENV_DIR/bin/python" -m streamlit run "$RESOURCES_DIR/app.py" \
  --server.headless=true \
  --server.address=127.0.0.1 \
  --server.port="$PORT" \
  --browser.gatherUsageStats=false \
  --client.toolbarMode=minimal &
SERVER_PID=$!
printf '%s %s\n' "$PORT" "$SERVER_PID" > "$PORT_FILE"

cleanup() {
  kill "$SERVER_PID" 2>/dev/null
  # Remove shared files only if this instance still owns them: a launcher
  # exiting late must never clobber a newer instance's port file or lock.
  if [ "$(cat "$PORT_FILE" 2>/dev/null)" = "$PORT $SERVER_PID" ]; then
    rm -f "$PORT_FILE"
  fi
  if [ "$(cat "$LOCK_DIR/pid" 2>/dev/null)" = "$$" ]; then
    rm -rf "$LOCK_DIR"
  fi
}
# cleanup FIRST: if our stdout pipe is already broken, say() dies on
# SIGPIPE and must not abort the cleanup.
trap 'cleanup; say "signal received - shut down"; exit 0' TERM INT
trap cleanup EXIT

# ---------- wait until healthy ----------
WAITED=0
until curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/_stcore/health" >/dev/null 2>&1; do
  kill -0 "$SERVER_PID" 2>/dev/null || die "server exited before becoming healthy"
  if [ "$WAITED" -ge 180 ]; then die "server did not become healthy within 180s"; fi
  sleep 2
  WAITED=$((WAITED + 2))
done
say "server healthy on port $PORT (pid $SERVER_PID)"
rm -rf "$LOCK_DIR"

# ---------- idle watchdog: exit after IDLE_TIMEOUT with no window connected ----------
# The wrapper normally quits us directly; this reaps the server if the
# wrapper ever dies without cleaning up.
IDLE=0
while kill -0 "$SERVER_PID" 2>/dev/null; do
  sleep 5
  if lsof -nP -iTCP:"$PORT" -sTCP:ESTABLISHED 2>/dev/null | grep -q ESTABLISHED; then
    IDLE=0
  else
    IDLE=$((IDLE + 5))
    if [ "$IDLE" -ge "$IDLE_TIMEOUT" ]; then
      say "no window connected for ${IDLE}s - shutting down"
      break
    fi
  fi
done
say "exiting"
