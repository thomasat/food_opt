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
#   4 = not enough free disk space for setup
set -u

PYTHON_VERSION="3.13.7"   # the single place the Python version is pinned
# Installed size of the venv, used only to turn "MB on disk" into a percentage
# for the setup progress bar. Measured 855 MB on 2026-09-08 for the current
# lock, rounded up for headroom so the bar does not park at 99% for the last
# stretch; adjust when the lock changes (a wrong value only skews the bar).
EXPECTED_VENV_MB=1000

IDLE_TIMEOUT="${FOODOPT_IDLE_TIMEOUT_SECS:-900}"
ARCH="${FOODOPT_TEST_ARCH:-$(uname -m)}"

say() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

die() {  # $1: log message; $2: exit code (default 1)
  say "FATAL: $1"
  exit "${2:-1}"
}

# ---------- preflight: supported machine (before touching anything) ----------
OS_MAJOR="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)"
case "$OS_MAJOR" in
  ''|*[!0-9]*) OS_MAJOR=0 ;;   # non-numeric => fail safe as unsupported
esac
if [ "$ARCH" != "arm64" ] || [ "$OS_MAJOR" -lt 13 ]; then
  say "unsupported machine: arch=$ARCH macos=$OS_MAJOR"
  exit 2
fi

# ---------- paths ----------
# This script lives in Contents/Resources (scripts in Contents/MacOS break
# code signing: codesign demands a signature on everything in MacOS).
RESOURCES_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTENTS_DIR="$(dirname "$RESOURCES_DIR")"
APP_BUNDLE="$(dirname "$CONTENTS_DIR")"
UV_BIN="$CONTENTS_DIR/MacOS/uv"

SUPPORT_DIR="$HOME/Library/Application Support/FoodOptimizer"
# Home-folder root, NOT ~/Documents: macOS privacy protection (TCC) gates
# Documents/Desktop/Downloads behind a consent prompt and silently denies
# unsigned apps, killing the server at startup (getcwd -> EPERM).
DATA_DIR="$HOME/FoodOptimizer"
VENV_DIR="$SUPPORT_DIR/venv"
MARKER_FILE="$SUPPORT_DIR/setup_complete"
LOG_FILE="$SUPPORT_DIR/launcher.log"
PORT_FILE="$SUPPORT_DIR/server.port"
STATUS_FILE="$SUPPORT_DIR/status.txt"   # one line the wrapper shows while setting up
LOCK_DIR="$SUPPORT_DIR/launch.lock"   # held while a launch is in progress
LOCK_FILE="$RESOURCES_DIR/requirements.lock.txt"

export UV_PYTHON_INSTALL_DIR="$SUPPORT_DIR/python"
export UV_CACHE_DIR="$SUPPORT_DIR/uv-cache"
export PYTHONDONTWRITEBYTECODE=1

# Publish one progress line for the wrapper's status page. Format:
#   <human-readable text>|<percent>   (percent 0-100, or empty when unknown)
# Written atomically so the wrapper never reads a half-written line.
publish() {  # write the line only; the progress loop calls this every second
  printf '%s\n' "$1" > "$STATUS_FILE.tmp" && mv -f "$STATUS_FILE.tmp" "$STATUS_FILE"
}
status() {   # publish and log - for the handful of one-off step transitions
  publish "$1"
  say "${1%|*}"
}

# The wrapper offers to copy the app to /Applications in this case; noted
# here for the log (and for tests).
case "$APP_BUNDLE" in
  /Volumes/*|*/AppTranslocation/*) say "running from disk image at $APP_BUNDLE" ;;
esac

# ---------- log (rotate once) ----------
mkdir -p "$SUPPORT_DIR" "$DATA_DIR"
# Keep project data and the app environment private on shared Macs.
chmod 700 "$SUPPORT_DIR" "$DATA_DIR" 2>/dev/null || true
# Rotate only when no launch is in progress: rotating a live launch's log
# would shunt its output into launcher.log.1 mid-run (its tee keeps the old
# file handle) and leave a nearly-empty log for Help > Show Log File.
if [ -f "$LOG_FILE" ] && [ ! -d "$LOCK_DIR" ]; then mv -f "$LOG_FILE" "$LOG_FILE.1"; fi
exec > >(tee -a "$LOG_FILE") 2>&1
say "launcher started (bundle=$APP_BUNDLE)"

# ---------- replace any orphaned server ----------
# The wrapper spawns and owns exactly one launcher per launch. A server
# recorded here therefore belongs to a PREVIOUS app instance that didn't
# shut down cleanly (e.g. the wrapper was force-quit). Stop it and start
# fresh, so this launch owns a server it can manage — never reuse a process
# a live wrapper isn't tracking (that caused false "could not start" screens
# and stale-version reuse after upgrades).
if [ -f "$PORT_FILE" ]; then
  read -r _OLD_PORT OLD_PID < "$PORT_FILE" || true
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    say "stopping orphaned server (pid $OLD_PID) from a previous session"
    kill "$OLD_PID" 2>/dev/null
    _waited=0
    while kill -0 "$OLD_PID" 2>/dev/null && [ "$_waited" -lt 5 ]; do
      sleep 1; _waited=$((_waited + 1))
    done
    kill -9 "$OLD_PID" 2>/dev/null
  fi
  rm -f "$PORT_FILE"
fi

# ---------- one launch at a time ----------
OTHER_PID=""
acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo $$ > "$LOCK_DIR/pid"
    return 0
  fi
  # Lock dir exists. The owner may be mid-write (mkdir then pid write are not
  # one atomic step), so an empty pid is not proof of abandonment — retry
  # briefly before considering it stealable.
  OTHER_PID=""
  _tries=0
  while [ "$_tries" -lt 5 ]; do
    OTHER_PID="$(cat "$LOCK_DIR/pid" 2>/dev/null)"
    [ -n "${OTHER_PID:-}" ] && break
    sleep 1; _tries=$((_tries + 1))
  done
  if [ -n "${OTHER_PID:-}" ] && kill -0 "$OTHER_PID" 2>/dev/null; then
    return 1
  fi
  # Owner is dead, or never wrote a pid within the grace window — steal.
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" 2>/dev/null && echo $$ > "$LOCK_DIR/pid" && return 0
  return 1
}
if ! acquire_lock; then
  say "another launch is already in progress (pid ${OTHER_PID:-unknown}) - exiting"
  exit 0
fi
trap 'if [ "$(cat "$LOCK_DIR/pid" 2>/dev/null)" = "$$" ]; then rm -rf "$LOCK_DIR"; fi' EXIT
# Only now that this launch owns the lock is a leftover status file certainly
# ours to remove: a failed setup exits without reaching cleanup, and the
# wrapper would otherwise snap its bar to that stale percentage on Try again.
rm -f "$STATUS_FILE" "$STATUS_FILE.tmp"

# ---------- first-run / upgrade setup ----------
LOCK_HASH="$(shasum -a 256 "$LOCK_FILE" | awk '{print $1}')"
NEED_SETUP=1
if [ -x "$VENV_DIR/bin/python" ] && [ -f "$MARKER_FILE" ] \
   && [ "$(head -n 1 "$MARKER_FILE" 2>/dev/null)" = "$LOCK_HASH" ]; then
  NEED_SETUP=0
fi

if [ "$NEED_SETUP" = "1" ]; then
  # A missing/empty lock hash means the bundled resource is missing — that is
  # a broken install, not a network problem; don't send the user chasing Wi-Fi.
  if [ -z "$LOCK_HASH" ] || [ ! -f "$LOCK_FILE" ]; then
    die "setup resource missing (requirements.lock.txt) - the app may be damaged" 1
  fi
  # Setup needs ~6 GB; a full disk fails the same way as no internet, so check
  # first and report the real cause (exit 4 -> disk-space page).
  FREE_KB="$(df -k "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')"
  case "$FREE_KB" in ''|*[!0-9]*) FREE_KB=0 ;; esac
  if [ "$FREE_KB" -gt 0 ] && [ "$FREE_KB" -lt 6291456 ]; then
    die "not enough free disk space for setup (need about 6 GB)" 4
  fi
  say "one-time setup starting (downloading software components)"
  # The marker test must run BEFORE the rm below: its presence is what tells
  # an upgrade (components only) apart from a first install (Python too). An
  # upgrade skips the Python download, so it is a two-step job, not three -
  # numbering it 3, 2, 3 would look like the setup was going backwards.
  if [ -f "$MARKER_FILE" ]; then
    STEP_ENV="step 1 of 2"
    STEP_SYNC="step 2 of 2"
    status "Updating components…|"
  else
    STEP_ENV="step 2 of 3"
    STEP_SYNC="step 3 of 3"
    status "Downloading Python (step 1 of 3)|"
  fi
  rm -f "$MARKER_FILE"
  NET_MSG="setup failed - most likely no internet connection"
  say "installing Python $PYTHON_VERSION"
  "$UV_BIN" python install --no-bin "$PYTHON_VERSION" || die "$NET_MSG" 3
  status "Creating environment ($STEP_ENV)|"
  rm -rf "$VENV_DIR"
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR" || die "$NET_MSG" 3
  say "installing components"
  # Quitting mid-setup must not leave an orphaned uv behind: the post-server
  # trap below is not installed yet, so cover this window (143 = 128 + TERM).
  trap 'kill "${SYNC_PID:-}" 2>/dev/null; rm -f "$STATUS_FILE" "$STATUS_FILE.tmp"; exit 143' TERM INT
  # Run the long download in the background so we can report real progress:
  # the venv grows on disk as wheels are installed, which is the only
  # progress signal uv gives us without parsing its output.
  "$UV_BIN" pip sync --python "$VENV_DIR/bin/python" "$LOCK_FILE" &
  SYNC_PID=$!
  LAST_LOG_MB=-25       # so the first measurement is always logged
  LAST_LOG_AT=$SECONDS
  while kill -0 "$SYNC_PID" 2>/dev/null; do
    VENV_KB="$(du -sk "$VENV_DIR" 2>/dev/null | awk '{print $1}')"
    case "${VENV_KB:-}" in ''|*[!0-9]*) VENV_KB=0 ;; esac
    VENV_MB=$((VENV_KB / 1024))
    PCT=$((VENV_MB * 100 / EXPECTED_VENV_MB))
    [ "$PCT" -gt 99 ] && PCT=99   # never show 100% while work remains
    MSG="Installing components ($STEP_SYNC): $VENV_MB MB of about $EXPECTED_VENV_MB MB"
    publish "$MSG|$PCT"
    # The status file moves every second; the log gets a line only every 25 MB
    # or 30s, so Help > Show Log File stays readable.
    if [ $((VENV_MB - LAST_LOG_MB)) -ge 25 ] || [ $((SECONDS - LAST_LOG_AT)) -ge 30 ]; then
      say "$MSG"
      LAST_LOG_MB="$VENV_MB"
      LAST_LOG_AT="$SECONDS"
    fi
    sleep 1
  done
  wait "$SYNC_PID" || die "$NET_MSG" 3
  status "Finishing setup ($STEP_SYNC)|100"
  { echo "$LOCK_HASH"; echo "python=$PYTHON_VERSION"; } > "$MARKER_FILE"
  say "setup complete"
fi

# ---------- start the server ----------
PORT="${FOODOPT_PORT_BASE:-8501}"   # override lets tests avoid real-user ports
# lsof only sees this user's processes, so on a shared Mac another account's
# listener would be invisible and Streamlit would fail to bind on every
# launch. The connect probe sees everyone's listeners.
port_in_use() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1 && return 0
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}
while port_in_use "$PORT"; do PORT=$((PORT + 1)); done

cd "$DATA_DIR" || die "could not enter data dir $DATA_DIR"

# ---------- safety backups: snapshot all projects on every launch ----------
BACKUP_ROOT="$DATA_DIR/backups"
KEEP_BACKUPS=10
set -- ./*.pkl
if [ -e "$1" ]; then
  # Skip if nothing changed since the last backup, so repeatedly opening the
  # app in one day doesn't churn through the retained snapshots with dupes.
  CURR_SUM="$(cat ./*.pkl 2>/dev/null | shasum -a 256 | awk '{print $1}')"
  LAST_SUM="$(cat "$BACKUP_ROOT/.last_sum" 2>/dev/null || true)"
  if [ "$CURR_SUM" != "$LAST_SUM" ]; then
    BACKUP_DIR="$BACKUP_ROOT/$(date '+%Y-%m-%d_%H%M%S')"
    mkdir -p "$BACKUP_DIR"
    cp ./*.pkl "$BACKUP_DIR/"
    printf '%s' "$CURR_SUM" > "$BACKUP_ROOT/.last_sum"
    say "projects backed up to $BACKUP_DIR"
    COUNT="$(find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
    while [ "$COUNT" -gt "$KEEP_BACKUPS" ]; do
      OLDEST="$(find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)"
      rm -rf "$OLDEST"
      COUNT=$((COUNT - 1))
    done
  fi
fi

status "Starting the app|"
"$VENV_DIR/bin/python" -m streamlit run "$RESOURCES_DIR/app.py" \
  --server.headless=true \
  --server.address=127.0.0.1 \
  --server.port="$PORT" \
  --browser.gatherUsageStats=false \
  --client.toolbarMode=viewer &
SERVER_PID=$!
# Atomic write so the wrapper never reads a half-written port line.
printf '%s %s\n' "$PORT" "$SERVER_PID" > "$PORT_FILE.tmp" && mv -f "$PORT_FILE.tmp" "$PORT_FILE"

cleanup() {
  kill "$SERVER_PID" 2>/dev/null
  rm -f "$STATUS_FILE" "$STATUS_FILE.tmp"
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
# SIGPIPE and must not abort the cleanup. These also replace the setup-phase
# TERM/INT trap installed above.
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
rm -f "$STATUS_FILE"   # the wrapper is about to show the app itself
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
