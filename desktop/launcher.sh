#!/bin/bash
# Food Optimizer — desktop launcher.
# Runs as "Food Optimizer.app/Contents/MacOS/launcher.sh".
# First run: provisions a per-user Python + dependencies using the bundled uv.
# Every run: starts the Streamlit app on localhost only and opens the browser.
set -u

PYTHON_VERSION="3.13.7"   # the single place the Python version is pinned
APP_NAME="Food Optimizer"

HEADLESS="${FOODOPT_HEADLESS:-0}"
IDLE_TIMEOUT="${FOODOPT_IDLE_TIMEOUT_SECS:-900}"
ARCH="${FOODOPT_TEST_ARCH:-$(uname -m)}"

say() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# ---------- preflight: supported machine (before touching anything) ----------
UNSUPPORTED_MSG="This app needs a Mac with an Apple chip (2020 or newer) running macOS 13 or later. Please contact us for help."
OS_MAJOR="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)"
if [ "$ARCH" != "arm64" ] || [ "${OS_MAJOR:-0}" -lt 13 ] 2>/dev/null; then
  say "unsupported machine: arch=$ARCH macos=${OS_MAJOR:-unknown}"
  if [ "$HEADLESS" != "1" ]; then
    osascript -e "display dialog \"$UNSUPPORTED_MSG\" with title \"$APP_NAME\" buttons {\"OK\"} default button 1 with icon caution" >/dev/null 2>&1
  fi
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

# ---------- dialog helpers (plain language only; no double quotes in messages) ----------
show_info() {  # non-blocking information dialog
  say "[dialog] $1"
  if [ "$HEADLESS" != "1" ]; then
    osascript -e "display dialog \"$1\" with title \"$APP_NAME\" buttons {\"OK\"} default button 1" >/dev/null 2>&1 &
  fi
}

ask_ok() {     # blocking OK / Not Now question; returns 0 on OK
  say "[ask] $1"
  if [ "$HEADLESS" = "1" ]; then return 1; fi
  osascript -e "display dialog \"$1\" with title \"$APP_NAME\" buttons {\"Not Now\", \"OK\"} default button \"OK\" cancel button \"Not Now\"" >/dev/null 2>&1
}

die() {        # fatal: plain-language dialog pointing at the log, then exit 1
  say "FATAL: $1"
  if [ "$HEADLESS" != "1" ]; then
    osascript -e "display dialog \"$1\n\nTechnical details were saved to:\n$LOG_FILE\" with title \"$APP_NAME\" buttons {\"OK\"} default button 1 with icon caution" >/dev/null 2>&1
  fi
  exit 1
}

open_ui() {    # $1: port. Chromium app-mode window (no address bar) when available,
               # so the app feels native; falls back to the default browser.
  UI_URL="http://localhost:$1"
  for CANDIDATE in "Google Chrome" "Microsoft Edge" "Brave Browser"; do
    if [ -d "/Applications/$CANDIDATE.app" ]; then
      say "opening app window via $CANDIDATE"
      if open -na "$CANDIDATE" --args --app="$UI_URL"; then return 0; fi
    fi
  done
  say "opening default browser"
  open "$UI_URL"
}

# ---------- preflight: running from the disk image? ----------
case "$APP_BUNDLE" in
  /Volumes/*|*/AppTranslocation/*)
    say "running from disk image at $APP_BUNDLE"
    if ask_ok "$APP_NAME should be copied to your Applications folder first - do that now?"; then
      rm -rf "/Applications/$APP_NAME.app"
      if ditto "$APP_BUNDLE" "/Applications/$APP_NAME.app"; then
        open "/Applications/$APP_NAME.app"
        exit 0
      fi
      show_info "The copy did not work. Please drag $APP_NAME onto the Applications folder, then open it from there."
    fi
    ;;
esac

# ---------- log (rotate once) ----------
mkdir -p "$SUPPORT_DIR" "$DATA_DIR"
if [ -f "$LOG_FILE" ]; then mv -f "$LOG_FILE" "$LOG_FILE.1"; fi
exec > >(tee -a "$LOG_FILE") 2>&1
say "launcher started (headless=$HEADLESS bundle=$APP_BUNDLE)"

# ---------- single instance ----------
if [ -f "$PORT_FILE" ]; then
  read -r OLD_PORT OLD_PID < "$PORT_FILE" || true
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null \
     && curl -fsS --max-time 3 "http://127.0.0.1:${OLD_PORT}/_stcore/health" >/dev/null 2>&1; then
    say "already running on port $OLD_PORT - reopening browser"
    if [ "$HEADLESS" != "1" ]; then open_ui "$OLD_PORT"; fi
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
  show_info "Setting up $APP_NAME (one time, usually 1 to 5 minutes depending on your internet speed). This downloads the app's software components; none of your data is sent anywhere. When setup finishes, the app opens on your screen. To open it again later, just open $APP_NAME from Applications, like any app."
  rm -f "$MARKER_FILE"
  NET_MSG="Setup needs an internet connection the first time you open $APP_NAME. Please connect to the internet and open the app again."
  say "installing Python $PYTHON_VERSION"
  "$UV_BIN" python install --no-bin "$PYTHON_VERSION" || die "$NET_MSG"
  say "creating environment"
  rm -rf "$VENV_DIR"
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR" || die "$NET_MSG"
  say "installing components"
  "$UV_BIN" pip sync --python "$VENV_DIR/bin/python" "$LOCK_FILE" || die "$NET_MSG"
  { echo "$LOCK_HASH"; echo "python=$PYTHON_VERSION"; } > "$MARKER_FILE"
  say "setup complete"
fi

# ---------- start the server ----------
PORT="${FOODOPT_PORT_BASE:-8501}"   # override lets tests avoid real-user ports
while lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; do PORT=$((PORT + 1)); done

cd "$DATA_DIR" || die "Could not open the FoodOptimizer folder inside your home folder."

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
trap 'say "signal received - shutting down"; cleanup; exit 0' TERM INT
trap cleanup EXIT

# ---------- wait until healthy ----------
START_MSG="The app could not start. Please open $APP_NAME again; if this keeps happening, contact us."
WAITED=0
until curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/_stcore/health" >/dev/null 2>&1; do
  kill -0 "$SERVER_PID" 2>/dev/null || die "$START_MSG"
  if [ "$WAITED" -ge 180 ]; then die "$START_MSG"; fi
  sleep 2
  WAITED=$((WAITED + 2))
done
say "server healthy on port $PORT (pid $SERVER_PID)"
rm -rf "$LOCK_DIR"

if [ "$HEADLESS" != "1" ]; then open_ui "$PORT"; fi

# ---------- idle watchdog: exit after IDLE_TIMEOUT with no browser connected ----------
IDLE=0
while kill -0 "$SERVER_PID" 2>/dev/null; do
  sleep 5
  if lsof -nP -iTCP:"$PORT" -sTCP:ESTABLISHED 2>/dev/null | grep -q ESTABLISHED; then
    IDLE=0
  else
    IDLE=$((IDLE + 5))
    if [ "$IDLE" -ge "$IDLE_TIMEOUT" ]; then
      say "no browser connected for ${IDLE}s - shutting down"
      break
    fi
  fi
done
say "exiting"
