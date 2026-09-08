#!/bin/bash
# End-to-end verification of the desktop packaging.
#   ./desktop/test_e2e.sh                full run (level 1 + level 2)
#   ./desktop/test_e2e.sh --level1-only  fast static + build checks only
# Level 2 uses a throwaway temp HOME; your real environment is never touched.
set -u

LEVEL1_ONLY=0
[ "${1:-}" = "--level1-only" ] && LEVEL1_ONLY=1

DESKTOP_DIR="$(cd "$(dirname "$0")" && pwd)"
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
assert "launcher binds localhost only" grep -q -- '--server.address=127.0.0.1' "$DESKTOP_DIR/launcher.sh"
assert "launcher disables telemetry" grep -q -- '--browser.gatherUsageStats=false' "$DESKTOP_DIR/launcher.sh"
assert "launcher hides developer toolbar" grep -q -- '--client.toolbarMode=viewer' "$DESKTOP_DIR/launcher.sh"

echo "== Level 1: lock file is a real compiled lock =="
LOCK="$DESKTOP_DIR/requirements.lock.txt"
for pkg in streamlit botorch gpytorch torch pandas numpy tornado; do
  assert "lock pins $pkg" grep -qi "^$pkg==" "$LOCK"
done
# The macOS lock must never inherit the Linux-only +cpu wheel variant.
if grep -q '+cpu' "$LOCK"; then fail "lock is macOS-resolvable (no +cpu)"; else ok "lock is macOS-resolvable (no +cpu)"; fi
if grep -qi '^pytest==' "$LOCK"; then fail "lock excludes pytest"; else ok "lock excludes pytest"; fi

echo "== Level 1: build =="
if "$DESKTOP_DIR/build_dmg.sh" "$TEST_VERSION"; then ok "build_dmg.sh runs"; else fail "build_dmg.sh runs"; fi

assert "launcher executable"  test -x "$DIST_APP/Contents/Resources/launcher.sh"
# Scripts in Contents/MacOS break codesign (unsigned nested code); guard it.
if find "$DIST_APP/Contents/MacOS" -name '*.sh' | grep -q .; then
  fail "no scripts in Contents/MacOS"
else
  ok "no scripts in Contents/MacOS"
fi
assert "uv executable"        test -x "$DIST_APP/Contents/MacOS/uv"
assert "native wrapper executable" test -x "$DIST_APP/Contents/MacOS/FoodOptimizer"
if file "$DIST_APP/Contents/MacOS/FoodOptimizer" | grep -q "Mach-O 64-bit executable arm64"; then
  ok "native wrapper is arm64 Mach-O"
else
  fail "native wrapper is arm64 Mach-O"
fi
for f in app.py food_bo.py storage.py ui_helpers.py data/ingredients.csv requirements.lock.txt icon.icns; do
  assert "Resources/$f present" test -f "$DIST_APP/Contents/Resources/$f"
done
assert "Info.plist present"   test -f "$DIST_APP/Contents/Info.plist"

# The bundle deliberately ships Resources/data/ingredients.csv (sample project +
# template); anything else under a data dir is stray.
STRAY="$(find "$DIST_APP" \( -name '*.pkl' -o -name '__pycache__' -o -name 'results' -o -name 'plots' \) 2>/dev/null; find "$DIST_APP/Contents/Resources/data" -type f ! -name 'ingredients.csv' 2>/dev/null; find "$DIST_APP" -name data -not -path '*/Contents/Resources/data' 2>/dev/null)"
if [ -z "$STRAY" ]; then ok "no stray files in bundle"; else fail "no stray files in bundle ($STRAY)"; fi

SIZE=$(stat -f%z "$DMG")
if [ "$SIZE" -lt 62914560 ]; then ok "dmg under 60MB ($SIZE bytes)"; else fail "dmg under 60MB ($SIZE bytes)"; fi

echo "== Level 1: dmg contents =="
hdiutil detach "$VOL" >/dev/null 2>&1 || true
ATTACH_OUT="$(hdiutil attach -nobrowse -readonly "$DMG" 2>/dev/null)"
if [ -n "$ATTACH_OUT" ]; then ok "dmg mounts"; else fail "dmg mounts"; fi
# Use the ACTUAL mount point from hdiutil's output: if another Food Optimizer
# volume is already mounted (e.g. someone testing the shipped dmg), ours lands
# at "Food Optimizer 1" and assuming the name would silently run every later
# check against the wrong, stale bundle.
VOL="$(printf '%s\n' "$ATTACH_OUT" | grep -o '/Volumes/.*' | tail -1 | sed 's/[[:space:]]*$//')"
assert "dmg has app"            test -d "$VOL/$APP_NAME.app"
assert "dmg has /Applications"  test -L "$VOL/Applications"
assert "dmg has Start Here.txt" test -f "$VOL/Start Here.txt"
assert "dmg has example ingredients" test -f "$VOL/Example Data/example ingredients.csv"
assert "dmg has example experiments" test -f "$VOL/Example Data/example experiments.csv"

if [ -n "${SIGN_IDENTITY:-}" ]; then
  assert "codesign verifies" codesign --verify --deep --strict "$DIST_APP"
  # Gatekeeper acceptance additionally requires notarization, so only
  # assert it when this build was actually notarized.
  if [ -n "${NOTARY_PROFILE:-}" ]; then
    assert "spctl accepts app" spctl -a -vv "$VOL/$APP_NAME.app"
  fi
fi

if [ "$LEVEL1_ONLY" = "1" ]; then
  hdiutil detach "$VOL" >/dev/null 2>&1 || true
  echo "== DONE (level 1 only): $PASS passed, $FAIL failed =="
  [ "$FAIL" -eq 0 ] || exit 1
  rm -f "$DMG"   # test-version dmg; on success dist/ keeps only release dmgs
  exit 0
fi

echo "== Level 2: end-to-end (temp HOME) =="
# Use a high port base so servers on real-user ports (8501+) — or stale
# browser windows auto-reconnecting to them — can't interfere with the run.
export FOODOPT_PORT_BASE=18501
E2E_HOME="$(mktemp -d)"
WORK="$(mktemp -d)"
SUPPORT="$E2E_HOME/Library/Application Support/FoodOptimizer"
DATA="$E2E_HOME/FoodOptimizer"
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
pid_dead()       { ! kill -0 "$1" 2>/dev/null; }
backup_exists()  { ls "$DATA"/backups/*/E2E_Smoke.pkl >/dev/null 2>&1; }
server_healthy() { curl -fsS --max-time 2 "http://127.0.0.1:$(server_port)/_stcore/health" 2>/dev/null | grep -q ok; }
launch() {  # launch <idle_timeout> <logfile>  — starts launcher in background
  HOME="$E2E_HOME" FOODOPT_IDLE_TIMEOUT_SECS="$1" \
    "$WORK/$APP_NAME.app/Contents/Resources/launcher.sh" > "$2" 2>&1 &
  LAUNCHER_PID=$!
}

# Copy the app out of the mounted dmg (dmg is still attached from level 1)
ditto "$VOL/$APP_NAME.app" "$WORK/$APP_NAME.app"

echo "-- test 0: unsupported-machine preflight touches nothing --"
HOME="$E2E_HOME" FOODOPT_TEST_ARCH=x86_64 \
  "$WORK/$APP_NAME.app/Contents/Resources/launcher.sh" > "$WORK/arch.out" 2>&1
RC=$?
if [ "$RC" = "2" ]; then ok "arch preflight exit code 2"; else fail "arch preflight exit code 2 (got $RC)"; fi
assert "arch preflight message" grep -q "unsupported machine" "$WORK/arch.out"
assert "arch preflight created nothing" not_exists "$SUPPORT"

echo "-- test 1: fresh first launch (downloads ~1GB, be patient) --"
launch 600 "$WORK/run1.out"
# The wrapper's progress bar reads this file; it must appear while setup runs
# and be gone once the app is about to be shown.
for _ in $(seq 1 15); do [ -f "$SUPPORT/status.txt" ] && break; sleep 1; done
if [ -f "$SUPPORT/status.txt" ]; then ok "status.txt written during setup"; else fail "status.txt was not written during setup"; fi
if wait_for 900 server_healthy; then ok "server healthy after fresh setup"; else fail "server healthy after fresh setup"; fi
if wait_for 30 not_exists "$SUPPORT/status.txt"; then ok "status.txt removed once healthy"; else fail "status.txt still present after server healthy"; fi
assert "venv created"            test -x "$SUPPORT/venv/bin/python"
assert "python under support"    dir_nonempty "$SUPPORT/python"
assert "uv cache under support"  dir_nonempty "$SUPPORT/uv-cache"
assert "marker written"          test -f "$MARKER"
assert "nothing in ~/.local"     not_exists "$E2E_HOME/.local"
assert "data dir created in home" test -d "$DATA"

echo "-- test 2: localhost-only binding --"
PORT="$(server_port)"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q '127.0.0.1'; then ok "bound to 127.0.0.1"; else fail "bound to 127.0.0.1"; fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -qE '\*:|0\.0\.0\.0'; then fail "not bound to all interfaces"; else ok "not bound to all interfaces"; fi
assert "main page HTTP 200" curl -fsS -o /dev/null "http://127.0.0.1:$PORT/"

echo "-- test 3: a second launch replaces the running server (no duplicates) --"
OLD_PID="$(awk '{print $2}' "$PORT_FILE" 2>/dev/null)"
launch 30 "$WORK/run2.out"          # kills the orphan, owns a fresh server
if wait_for 30 pid_dead "$OLD_PID"; then ok "previous server stopped"; else fail "previous server stopped"; fi
if wait_for 60 server_healthy; then ok "replacement server healthy"; else fail "replacement server healthy"; fi
LISTENERS=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -c LISTEN)
if [ "$LISTENERS" = "1" ]; then ok "exactly one listener"; else fail "exactly one listener (got $LISTENERS)"; fi

echo "-- test 4: idle shutdown --"
if wait_for 120 not_exists "$PORT_FILE"; then ok "idle shutdown removed port file"; else fail "idle shutdown removed port file"; fi
assert "idle shutdown logged" grep -q "no window connected" "$WORK/run2.out"
LAUNCHER_PID=""

echo "-- test 5: core logic in the packaged environment --"
SMOKE_OUT="$(cd "$DATA" && HOME="$E2E_HOME" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WORK/$APP_NAME.app/Contents/Resources" \
  "$SUPPORT/venv/bin/python" - <<'PY'
from food_bo import FoodOptimizer
opt = FoodOptimizer("E2E_Smoke")
opt.add_ingredient("flour", 10.0, 100.0)
opt.add_ingredient("sugar", 0.0, 50.0)
opt.add_objective("taste", 1.0, goal="max")
batch = opt.ask(n_suggestions=1)
opt.tell(batch[0], {"taste": 7.5})
# Full process continues: a parameter added mid-run needs a baseline (clear
# error without one), history re-encodes, and everything survives a reload.
try:
    opt.add_process_parameter("oven_temp", 150.0, 220.0)
    raise SystemExit("expected ValueError without baseline")
except ValueError:
    pass
opt.add_process_parameter("oven_temp", 150.0, 220.0, baseline=180.0)
batch2 = opt.ask(n_suggestions=1)
assert 150.0 <= batch2[0]["oven_temp"] <= 220.0
opt.tell(batch2[0], {"taste": 8.0})
opt2 = FoodOptimizer("E2E_Smoke")
assert opt2.load_error is None and len(opt2.X_history) == 2
assert all(len(x) == 3 for x in opt2.X_history)
assert opt2.X_history[0][2] == 180.0  # first experiment encoded at baseline
print("SMOKE_OK")
PY
)"
if echo "$SMOKE_OUT" | grep -q SMOKE_OK; then ok "FoodOptimizer smoke test"; else fail "FoodOptimizer smoke test ($SMOKE_OUT)"; fi

echo "-- test 5b: bundled UI surfaces errors as messages, not tracebacks --"
UI_OUT="$(cd "$DATA" && HOME="$E2E_HOME" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WORK/$APP_NAME.app/Contents/Resources" \
  APP_RESOURCES="$WORK/$APP_NAME.app/Contents/Resources" \
  "$SUPPORT/venv/bin/python" - <<'PY'
import os
from streamlit.testing.v1 import AppTest
from food_bo import FoodOptimizer

opt = FoodOptimizer("UI_Check")
opt.add_ingredient("water", 0.0, 100.0)
opt.add_objective("taste", 1.0, goal="max")
opt.tell({"water": 50.0}, {"taste": 7.0})

at = AppTest.from_file(
    os.path.join(os.environ["APP_RESOURCES"], "app.py"), default_timeout=300)
at.session_state["_loaded_project"] = "UI_Check"
at.run()
assert not at.exception, at.exception
at.text_input(key="pp_name").set_value("oven_temp")
at.number_input(key="pp_min").set_value(150.0)
at.number_input(key="pp_max").set_value(220.0)
at.number_input(key="pp_base").set_value(100.0)  # outside [150, 220]
next(b for b in at.button if b.label == "Add Process Parameter").click()
at.run()
assert not at.exception, at.exception   # a traceback here is the bug
assert any("must be between" in str(e.value) for e in at.error)
print("UI_OK")
PY
)"
if echo "$UI_OUT" | grep -q UI_OK; then ok "bundled UI error handling"; else fail "bundled UI error handling ($UI_OUT)"; fi
assert "pkl saved to data dir" test -f "$DATA/E2E_Smoke.pkl"
# Project files must be JSON (safe to open), not executable pickle.
assert "project file is JSON" "$SUPPORT/venv/bin/python" -c "import json,sys; json.load(open(sys.argv[1]))" "$DATA/E2E_Smoke.pkl"

echo "-- test 6: upgrade path (stale marker hash) --"
sed -i '' '1s/.*/stale-hash-forces-resync/' "$MARKER"
launch 600 "$WORK/run4.out"
if wait_for 300 server_healthy; then ok "relaunch after stale marker"; else fail "relaunch after stale marker"; fi
FIRST_LINE="$(head -n 1 "$MARKER")"
if [ "$FIRST_LINE" != "stale-hash-forces-resync" ] && [ -n "$FIRST_LINE" ]; then ok "marker refreshed"; else fail "marker refreshed"; fi
assert "resync logged" grep -q "installing components" "$WORK/run4.out"
kill "$LAUNCHER_PID" 2>/dev/null
if wait_for 30 not_exists "$PORT_FILE"; then ok "TERM cleans up port file"; else fail "TERM cleans up port file"; fi
LAUNCHER_PID=""

echo "-- test 7: interrupted-setup recovery + warm relaunch under 30s --"
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
assert "launch backed up existing projects" backup_exists

echo "-- test 7b: late cleanup never clobbers another instance's port file --"
printf '%s\n' "1 1" > "$PORT_FILE"
kill "$LAUNCHER_PID" 2>/dev/null
sleep 3
if [ "$(cat "$PORT_FILE" 2>/dev/null)" = "1 1" ]; then ok "foreign port file preserved"; else fail "foreign port file preserved"; fi
rm -f "$PORT_FILE"
LAUNCHER_PID=""

echo "-- test 8: nothing was written into the bundle --"
if diff -r "$WORK/$APP_NAME.app" "$DIST_APP" >/dev/null 2>&1; then
  ok "bundle byte-identical after all runs"
else
  fail "bundle byte-identical after all runs"
fi

echo "-- test 9: running-from-dmg detection --"
HOME="$E2E_HOME" "$VOL/$APP_NAME.app/Contents/Resources/launcher.sh" > "$WORK/run_dmg.out" 2>&1 &
DMG_PID=$!
if wait_for 15 grep -q "running from disk image" "$WORK/run_dmg.out"; then ok "dmg location detected"; else fail "dmg location detected"; fi
kill "$DMG_PID" 2>/dev/null
DMG_SRV="$(awk '{print $2}' "$PORT_FILE" 2>/dev/null)"
[ -n "${DMG_SRV:-}" ] && kill "$DMG_SRV" 2>/dev/null
hdiutil detach "$VOL" >/dev/null 2>&1 || true

echo "== DONE: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ] || exit 1
rm -f "$DMG"   # test-version dmg; on success dist/ keeps only release dmgs
