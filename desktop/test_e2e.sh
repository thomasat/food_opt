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
# A test build is named apart from a release dmg, so a run never deletes one.
TEST_VERSION="$(cat "$DESKTOP_DIR/VERSION")"
DMG="$DESKTOP_DIR/dist/FoodOptimizer-$TEST_VERSION-test.dmg"
RECORD="${DMG%.dmg}.build.txt"
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
assert "launcher hides the Streamlit toolbar" grep -q -- '--client.toolbarMode=minimal' "$DESKTOP_DIR/launcher.sh"
assert "launcher shows no traceback in the window" grep -q -- '--client.showErrorDetails=none' "$DESKTOP_DIR/launcher.sh"
assert "starting line carries no percent" grep -qF -- 'status "Starting the app|"' "$DESKTOP_DIR/launcher.sh"
# The window shows the app on Streamlit's first healthy answer, which lands
# before app.py has imported torch and friends. The launcher imports them
# first, behind its own progress page, so that window is never blank.
assert "launcher warms the components before the server" \
  grep -qF 'import torch, botorch, gpytorch' "$DESKTOP_DIR/launcher.sh"
assert "the warm-up publishes a step line" \
  grep -qF 'WARM_MSG="Loading the app'"'"'s components' \
  "$DESKTOP_DIR/launcher.sh"
# Order is the whole point: warming the imports AFTER the server is spawned
# would leave the blank window exactly where it was.
WARM_AT="$(grep -n 'import torch, botorch, gpytorch' "$DESKTOP_DIR/launcher.sh" | head -n 1 | cut -d: -f1)"
RUN_AT="$(grep -n -- '-m streamlit run' "$DESKTOP_DIR/launcher.sh" | head -n 1 | cut -d: -f1)"
if [ -n "$WARM_AT" ] && [ -n "$RUN_AT" ] && [ "$WARM_AT" -lt "$RUN_AT" ]; then
  ok "the warm-up runs before the server is spawned"
else
  fail "the warm-up runs before the server is spawned"
fi
# A hung import must not strand the launch: it runs before the port file
# exists, so nothing else can give up on it.
assert "the warm-up wait is bounded" \
  grep -qF 'WARM_WAITED" -ge 120' "$DESKTOP_DIR/launcher.sh"
# ...and the window lists that step under the same name, or it would tick a
# step nobody is running.
assert "the window names the same step" \
  grep -qF 'let loadStepLabel = "Loading the app'"'"'s components"' \
  "$DESKTOP_DIR/FoodOptimizerApp.swift"
# The bar must exist from the first second of a first run, so the very first
# setup line the launcher publishes has to carry a percent, not an empty field.
assert "first setup line carries a percent" grep -qF '(step 1 of 3)|2' "$DESKTOP_DIR/launcher.sh"
assert "second setup line carries a percent" grep -q 'Creating environment (.*)|8' "$DESKTOP_DIR/launcher.sh"
assert "install step maps onto 10-99" grep -qF '10 + 89 * MB_DONE / EXPECTED_SETUP_MB' "$DESKTOP_DIR/launcher.sh"
# The download cache is most of the wait on a slow connection, so the install
# step's measure has to include it - and sample it apart from the venv, or the
# hard links between them dedupe away half the total.
assert "progress counts the download cache" grep -q 'du -sk .*UV_CACHE_DIR' "$DESKTOP_DIR/launcher.sh"
assert "progress counts the environment"    grep -q 'du -sk .*VENV_DIR' "$DESKTOP_DIR/launcher.sh"

echo "== Level 1: lock file is a real compiled lock =="
LOCK="$DESKTOP_DIR/requirements.lock.txt"
for pkg in streamlit botorch gpytorch torch pandas numpy tornado openpyxl; do
  assert "lock pins $pkg" grep -qi "^$pkg==" "$LOCK"
done
# The macOS lock must never inherit the Linux-only +cpu wheel variant.
if grep -q '+cpu' "$LOCK"; then fail "lock is macOS-resolvable (no +cpu)"; else ok "lock is macOS-resolvable (no +cpu)"; fi
if grep -qi '^pytest==' "$LOCK"; then fail "lock excludes pytest"; else ok "lock excludes pytest"; fi

echo "== Level 1: bundle list covers app.py's local imports =="
# app.py fails to import in the packaged app if a module it (or a ui_*.py it
# imports) imports at module level isn't in shared/bundle_manifest.txt, the
# list build_dmg.sh copies. Derive the required local modules straight from
# the source instead of hand-listing them, so this catches the next missed
# module too.
REPO_DIR="$(dirname "$DESKTOP_DIR")"
BUNDLE_FILES="$(cat "$DESKTOP_DIR/shared/bundle_manifest.txt")"
MISSING="$(REPO_DIR="$REPO_DIR" BUNDLE_FILES="$BUNDLE_FILES" python3 <<'PYEOF'
import ast, os

repo_dir = os.environ["REPO_DIR"]
bundle = set(os.environ["BUNDLE_FILES"].splitlines())  # names hold spaces


def top_level_local_imports(path):
    mods = set()
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:  # top-level only: skip imports nested in functions
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    return {m for m in mods if os.path.exists(os.path.join(repo_dir, m + ".py"))}


required = top_level_local_imports(os.path.join(repo_dir, "app.py"))
for mod in list(required):
    if mod.startswith("ui_"):
        required |= top_level_local_imports(os.path.join(repo_dir, mod + ".py"))

missing = sorted(mod + ".py" for mod in required if mod + ".py" not in bundle)
print("\n".join(missing))
PYEOF
)"
if [ -z "$MISSING" ]; then
  ok "bundle list covers app.py's local imports"
else
  fail "bundle list covers app.py's local imports (missing: $(printf '%s' "$MISSING" | tr '\n' ' '))"
fi

echo "== Level 1: build =="
if "$DESKTOP_DIR/build_dmg.sh" --test-build; then ok "build_dmg.sh runs"; else fail "build_dmg.sh runs"; fi
# The record is what a release is checked against on GitHub (dmg_sha256).
DMG_SHA="$(shasum -a 256 "$DMG" 2>/dev/null | awk '{print $1}')"
if [ -n "$DMG_SHA" ] && grep -qx "dmg_sha256=$DMG_SHA" "$RECORD" 2>/dev/null; then
  ok "build record names the dmg"
else
  fail "build record names the dmg"
fi
for lock in requirements.lock.txt requirements-windows.lock.txt; do
  case "$lock" in requirements.lock.txt) KEY=mac_lock_sha256 ;; *) KEY=windows_lock_sha256 ;; esac
  LOCK_SHA="$(shasum -a 256 "$DESKTOP_DIR/$lock" | awk '{print $1}')"
  assert "build record names $lock" grep -qx "$KEY=$LOCK_SHA" "$RECORD"
done

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
# The bundle is exactly the golden list: the manifest's files under
# Resources plus the Mac-only ones, every directory with a trailing /.
# _CodeSignature is the only part left out (signed builds only).
bundle_listing() {  # every entry under Contents/, dirs with a trailing /, _CodeSignature excluded
  (cd "$1/Contents" && find . -mindepth 1 -not -path './_CodeSignature' -not -path './_CodeSignature/*' |
    while IFS= read -r p; do p="${p#./}"; if [ -d "$p" ] && [ ! -L "$p" ]; then printf '%s/\n' "$p"; else printf '%s\n' "$p"; fi; done |
    LC_ALL=C sort)
}
if LISTING_DIFF="$(bundle_listing "$DIST_APP" | diff "$DESKTOP_DIR/mac_bundle_files.txt" - 2>&1)"; then
  ok "bundle holds exactly mac_bundle_files.txt"
else
  fail "bundle holds exactly mac_bundle_files.txt"
  printf '%s\n' "$LISTING_DIFF" | sed 's/^/    /'
fi

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
# The sample lives in the app (Try the sample project); a second copy on the
# disk image gave first-run users two routes and two names for one thing.
# The folder name below is the one 0.1.x shipped: it is asserted absent.
assert "dmg carries no bundled sample folder" test ! -e "$VOL/Example Data"
assert "dmg volume holds only app, Applications, Start Here" \
  test "$(find "$VOL" -mindepth 1 -maxdepth 1 -not -name '.*' -exec basename {} \; | sort | tr '\n' '|')" = "Applications|$APP_NAME.app|Start Here.txt|"

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
  rm -f "$DMG" "$RECORD"   # test build; on success dist/ keeps only release dmgs
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

# Tests 5-5g: the checks in shared/packaged_app_checks.py, run by the bundled
# venv against the copied bundle's Resources, in the throwaway projects folder.
run_check() { HOME="$E2E_HOME" PYTHONDONTWRITEBYTECODE=1 "$SUPPORT/venv/bin/python" "$DESKTOP_DIR/shared/packaged_app_checks.py" --resources "$WORK/$APP_NAME.app/Contents/Resources" --projects "$DATA" --only "$1" 2>&1; }

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
# Recorded so future sessions can read real first-run timings from launcher.log.
assert "setup duration logged"   grep -q "setup complete in" "$WORK/run1.out"
SETUP_SECS="$(sed -n 's/.*setup complete in \([0-9][0-9]*\) s.*/\1/p' "$WORK/run1.out" | tail -1)"
echo "  (measured: setup complete in ${SETUP_SECS:-?} s)"
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
SMOKE_OUT="$(run_check smoke)"
if echo "$SMOKE_OUT" | grep -q SMOKE_OK; then ok "FoodOptimizer smoke test"; else fail "FoodOptimizer smoke test ($SMOKE_OUT)"; fi

echo "-- test 5b: bundled UI surfaces errors as messages, not tracebacks --"
UI_OUT="$(run_check ui_errors)"
if echo "$UI_OUT" | grep -q UI_OK; then ok "bundled UI error handling"; else fail "bundled UI error handling ($UI_OUT)"; fi
assert "pkl saved to data dir" test -f "$DATA/E2E_Smoke.pkl"
# Project files must be JSON (safe to open), not executable pickle.
assert "project file is JSON" "$SUPPORT/venv/bin/python" -c "import json,sys; json.load(open(sys.argv[1]))" "$DATA/E2E_Smoke.pkl"

echo "-- test 5c: second wording wave controls are in the packaged app --"
WAVE_OUT="$(run_check wording_wave)"
if echo "$WAVE_OUT" | grep -q WAVE_OK; then ok "second wording wave controls in packaged app"; else fail "second wording wave controls in packaged app ($WAVE_OUT)"; fi

echo "-- test 5d: openpyxl importable in the bundled venv --"
OPENPYXL_OUT="$(run_check openpyxl)"
if echo "$OPENPYXL_OUT" | grep -q OPENPYXL_OK; then ok "import openpyxl"; else fail "import openpyxl ($OPENPYXL_OUT)"; fi

echo "-- test 5e: kitchen-trust controls (0.4.0) are in the packaged app --"
KT_OUT="$(run_check kitchen_trust)"
if echo "$KT_OUT" | grep -q KITCHEN_TRUST_OK; then ok "kitchen-trust controls in packaged app"; else fail "kitchen-trust controls in packaged app ($KT_OUT)"; fi

echo "-- test 5f: spreadsheet-feel controls (0.5.0) are in the packaged app --"
GRID_OUT="$(run_check spreadsheet_feel)"
if echo "$GRID_OUT" | grep -q GRID_OK; then ok "spreadsheet-feel controls in packaged app"; else fail "spreadsheet-feel controls in packaged app ($GRID_OUT)"; fi

echo "-- test 5g: rules, pre-mix, sample and draft controls are in the packaged app --"
RULES_OUT="$(run_check rules_premix_sample_draft)"
if echo "$RULES_OUT" | grep -q RULES_OK; then ok "rules controls in packaged app"; else fail "rules controls in packaged app ($RULES_OUT)"; fi

if echo "$RULES_OUT" | grep -q PREMIX_OK; then ok "pre-mix project and workbook in packaged app"; else fail "pre-mix project and workbook in packaged app ($RULES_OUT)"; fi

if echo "$RULES_OUT" | grep -q SAMPLE_OK; then ok "Try the sample project in packaged app"; else fail "Try the sample project in packaged app ($RULES_OUT)"; fi

if echo "$RULES_OUT" | grep -q DRAFT_OK; then ok "manual results persist in packaged app"; else fail "manual results persist in packaged app ($RULES_OUT)"; fi

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
rm -f "$DMG" "$RECORD"   # test build; on success dist/ keeps only release dmgs
