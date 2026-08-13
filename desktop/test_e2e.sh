#!/bin/bash
# End-to-end verification of the desktop packaging.
#   ./desktop/test_e2e.sh                full run (level 1 + level 2)
#   ./desktop/test_e2e.sh --level1-only  fast static + build checks only
# Level 2 uses a throwaway temp HOME; your real environment is never touched.
set -u

LEVEL1_ONLY=0
[ "${1:-}" = "--level1-only" ] && LEVEL1_ONLY=1

DESKTOP_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC2034  # REPO_DIR is used by Level 2 (Task 8)
REPO_DIR="$(dirname "$DESKTOP_DIR")"
APP_NAME="Food Optimizer"
DIST_APP="$DESKTOP_DIR/dist/$APP_NAME.app"
TEST_VERSION="0.0.1"
DMG="$DESKTOP_DIR/dist/FoodOptimizer-$TEST_VERSION.dmg"
VOL="/Volumes/$APP_NAME"

PASS=0
FAIL=0
ok()   { PASS=$((PASS + 1)); echo "  ok: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }
assert() {  # assert "description" command [args...]
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then ok "$desc"; else fail "$desc"; fi
}
dir_nonempty() { [ -n "$(ls -A "$1" 2>/dev/null)" ]; }
not_exists()   { [ ! -e "$1" ]; }

echo "== Level 1: static checks =="
for s in launcher.sh build_dmg.sh test_e2e.sh fetch_uv.sh update_lock.sh; do
  assert "bash -n $s" bash -n "$DESKTOP_DIR/$s"
done
if command -v shellcheck >/dev/null 2>&1; then
  for s in launcher.sh build_dmg.sh test_e2e.sh fetch_uv.sh update_lock.sh; do
    assert "shellcheck $s" shellcheck "$DESKTOP_DIR/$s"
  done
else
  echo "  (shellcheck not installed - skipped)"
fi
assert "plutil -lint Info.plist" plutil -lint "$DESKTOP_DIR/Info.plist"

echo "== Level 1: lock file is a real compiled lock =="
LOCK="$DESKTOP_DIR/requirements.lock.txt"
for pkg in streamlit botorch gpytorch torch pandas numpy matplotlib tornado; do
  assert "lock pins $pkg" grep -qi "^$pkg==" "$LOCK"
done
if grep -qi '^pytest==' "$LOCK"; then fail "lock excludes pytest"; else ok "lock excludes pytest"; fi

echo "== Level 1: build =="
if "$DESKTOP_DIR/build_dmg.sh" "$TEST_VERSION"; then ok "build_dmg.sh runs"; else fail "build_dmg.sh runs"; fi

assert "launcher executable"  test -x "$DIST_APP/Contents/MacOS/launcher.sh"
assert "uv executable"        test -x "$DIST_APP/Contents/MacOS/uv"
for f in app.py food_bo.py theory.py requirements.lock.txt icon.icns; do
  assert "Resources/$f present" test -f "$DIST_APP/Contents/Resources/$f"
done
assert "Info.plist present"   test -f "$DIST_APP/Contents/Info.plist"

STRAY="$(find "$DIST_APP" \( -name '*.pkl' -o -name '__pycache__' -o -name 'data' -o -name 'results' -o -name 'plots' \) 2>/dev/null)"
if [ -z "$STRAY" ]; then ok "no stray files in bundle"; else fail "no stray files in bundle ($STRAY)"; fi

SIZE=$(stat -f%z "$DMG")
if [ "$SIZE" -lt 62914560 ]; then ok "dmg under 60MB ($SIZE bytes)"; else fail "dmg under 60MB ($SIZE bytes)"; fi

echo "== Level 1: dmg contents =="
hdiutil detach "$VOL" >/dev/null 2>&1 || true
assert "dmg mounts" hdiutil attach -nobrowse -readonly "$DMG"
assert "dmg has app"            test -d "$VOL/$APP_NAME.app"
assert "dmg has /Applications"  test -L "$VOL/Applications"
assert "dmg has Start Here.txt" test -f "$VOL/Start Here.txt"

if [ -n "${SIGN_IDENTITY:-}" ]; then
  assert "codesign verifies" codesign --verify --deep --strict "$DIST_APP"
  assert "spctl accepts app" spctl -a -vv "$VOL/$APP_NAME.app"
fi

if [ "$LEVEL1_ONLY" = "1" ]; then
  hdiutil detach "$VOL" >/dev/null 2>&1 || true
  echo "== DONE (level 1 only): $PASS passed, $FAIL failed =="
  [ "$FAIL" -eq 0 ] || exit 1
  exit 0
fi

echo "== Level 2: end-to-end (temp HOME) =="
E2E_HOME="$(mktemp -d)"
WORK="$(mktemp -d)"
SUPPORT="$E2E_HOME/Library/Application Support/FoodOptimizer"
DOCS="$E2E_HOME/Documents/FoodOptimizer"
PORT_FILE="$SUPPORT/server.port"
MARKER="$SUPPORT/setup_complete"
LAUNCHER_PID=""

teardown() {
  [ -n "$LAUNCHER_PID" ] && kill "$LAUNCHER_PID" 2>/dev/null
  sleep 2
  pkill -f "$E2E_HOME" 2>/dev/null
  hdiutil detach "$VOL" >/dev/null 2>&1 || true
  rm -rf "$E2E_HOME" "$WORK"
}
trap teardown EXIT

wait_for() {  # wait_for <timeout_secs> command [args...]
  local timeout="$1"; shift
  local t=0
  while ! "$@" >/dev/null 2>&1; do
    sleep 2; t=$((t + 2))
    [ "$t" -ge "$timeout" ] && return 1
  done
  return 0
}
server_port()    { awk '{print $1}' "$PORT_FILE" 2>/dev/null; }
server_healthy() { curl -fsS --max-time 2 "http://127.0.0.1:$(server_port)/_stcore/health" 2>/dev/null | grep -q ok; }
launch() {  # launch <idle_timeout> <logfile>  — starts launcher in background
  HOME="$E2E_HOME" FOODOPT_HEADLESS=1 FOODOPT_IDLE_TIMEOUT_SECS="$1" \
    "$WORK/$APP_NAME.app/Contents/MacOS/launcher.sh" > "$2" 2>&1 &
  LAUNCHER_PID=$!
}

# Copy the app out of the mounted dmg (dmg is still attached from level 1)
ditto "$VOL/$APP_NAME.app" "$WORK/$APP_NAME.app"

echo "-- test 0: unsupported-machine preflight touches nothing --"
HOME="$E2E_HOME" FOODOPT_HEADLESS=1 FOODOPT_TEST_ARCH=x86_64 \
  "$WORK/$APP_NAME.app/Contents/MacOS/launcher.sh" > "$WORK/arch.out" 2>&1
RC=$?
if [ "$RC" = "2" ]; then ok "arch preflight exit code 2"; else fail "arch preflight exit code 2 (got $RC)"; fi
assert "arch preflight message" grep -q "unsupported machine" "$WORK/arch.out"
assert "arch preflight created nothing" not_exists "$SUPPORT"

echo "-- test 1: fresh first launch (downloads ~2GB, be patient) --"
launch 30 "$WORK/run1.out"
if wait_for 900 server_healthy; then ok "server healthy after fresh setup"; else fail "server healthy after fresh setup"; fi
assert "venv created"            test -x "$SUPPORT/venv/bin/python"
assert "python under support"    dir_nonempty "$SUPPORT/python"
assert "uv cache under support"  dir_nonempty "$SUPPORT/uv-cache"
assert "marker written"          test -f "$MARKER"
assert "nothing in ~/.local"     not_exists "$E2E_HOME/.local"
assert "Documents dir created"   test -d "$DOCS"

echo "-- test 2: localhost-only binding --"
PORT="$(server_port)"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q '127.0.0.1'; then ok "bound to 127.0.0.1"; else fail "bound to 127.0.0.1"; fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -qE '\*:|0\.0\.0\.0'; then fail "not bound to all interfaces"; else ok "not bound to all interfaces"; fi

echo "-- test 3: single instance --"
HOME="$E2E_HOME" FOODOPT_HEADLESS=1 \
  "$WORK/$APP_NAME.app/Contents/MacOS/launcher.sh" > "$WORK/run2.out" 2>&1
assert "second launch reuses server" grep -q "already running on port $PORT" "$WORK/run2.out"
LISTENERS=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -c LISTEN)
if [ "$LISTENERS" = "1" ]; then ok "exactly one listener"; else fail "exactly one listener (got $LISTENERS)"; fi

echo "-- test 4: running-from-dmg detection --"
HOME="$E2E_HOME" FOODOPT_HEADLESS=1 \
  "$VOL/$APP_NAME.app/Contents/MacOS/launcher.sh" > "$WORK/run3.out" 2>&1
assert "dmg location detected" grep -q "running from disk image" "$WORK/run3.out"
hdiutil detach "$VOL" >/dev/null 2>&1 || true

echo "-- test 5: idle shutdown --"
if wait_for 120 not_exists "$PORT_FILE"; then ok "idle shutdown removed port file"; else fail "idle shutdown removed port file"; fi
assert "run1 logged idle shutdown" grep -q "no browser connected" "$WORK/run1.out"
LAUNCHER_PID=""

echo "-- test 6: core logic in the packaged environment --"
SMOKE_OUT="$(cd "$DOCS" && HOME="$E2E_HOME" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WORK/$APP_NAME.app/Contents/Resources" \
  "$SUPPORT/venv/bin/python" - <<'PY'
from food_bo import FoodOptimizer
opt = FoodOptimizer("E2E_Smoke")
opt.add_ingredient("flour", 10.0, 100.0)
opt.add_ingredient("sugar", 0.0, 50.0)
opt.add_objective("taste", 1.0, goal="max")
batch = opt.ask(n_suggestions=1)
opt.tell(batch[0], {"taste": 7.5})
print("SMOKE_OK")
PY
)"
if echo "$SMOKE_OUT" | grep -q SMOKE_OK; then ok "FoodOptimizer smoke test"; else fail "FoodOptimizer smoke test ($SMOKE_OUT)"; fi
assert "pkl saved to Documents" test -f "$DOCS/E2E_Smoke.pkl"

echo "-- test 7: upgrade path (stale marker hash) --"
sed -i '' '1s/.*/stale-hash-forces-resync/' "$MARKER"
launch 600 "$WORK/run4.out"
if wait_for 300 server_healthy; then ok "relaunch after stale marker"; else fail "relaunch after stale marker"; fi
FIRST_LINE="$(head -n 1 "$MARKER")"
if [ "$FIRST_LINE" != "stale-hash-forces-resync" ] && [ -n "$FIRST_LINE" ]; then ok "marker refreshed"; else fail "marker refreshed"; fi
assert "resync logged" grep -q "installing components" "$WORK/run4.out"
kill "$LAUNCHER_PID" 2>/dev/null
if wait_for 30 not_exists "$PORT_FILE"; then ok "TERM cleans up port file"; else fail "TERM cleans up port file"; fi
LAUNCHER_PID=""

echo "-- test 8: interrupted-setup recovery + warm relaunch under 30s --"
rm -f "$MARKER"
launch 600 "$WORK/run5.out"
if wait_for 300 server_healthy; then ok "recovery relaunch healthy"; else fail "recovery relaunch healthy"; fi
assert "marker recreated" test -f "$MARKER"
kill "$LAUNCHER_PID" 2>/dev/null
wait_for 30 not_exists "$PORT_FILE" || true
LAUNCHER_PID=""
START=$SECONDS
launch 600 "$WORK/run6.out"
if wait_for 60 server_healthy; then
  ELAPSED=$((SECONDS - START))
  if [ "$ELAPSED" -lt 30 ]; then ok "warm relaunch in ${ELAPSED}s"; else fail "warm relaunch in ${ELAPSED}s (>=30s)"; fi
else
  fail "warm relaunch healthy"
fi
if grep -q "installing components" "$WORK/run6.out"; then fail "warm relaunch skipped setup"; else ok "warm relaunch skipped setup"; fi
kill "$LAUNCHER_PID" 2>/dev/null
wait_for 30 not_exists "$PORT_FILE" || true
LAUNCHER_PID=""

echo "-- test 9: nothing was written into the bundle --"
if diff -r "$WORK/$APP_NAME.app" "$DIST_APP" >/dev/null 2>&1; then
  ok "bundle byte-identical after all runs"
else
  fail "bundle byte-identical after all runs"
fi

echo "== DONE: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ] || exit 1
