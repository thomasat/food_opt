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
assert "launcher hides the Streamlit toolbar" grep -q -- '--client.toolbarMode=minimal' "$DESKTOP_DIR/launcher.sh"
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
for f in app.py food_bo.py storage.py ui_helpers.py ui_setup.py ui_batch.py ui_results.py calculation_editor.py wording.py workbook_flow.py custom_records.py sample_projects.py data/sample_ingredients.csv requirements.lock.txt icon.icns; do
  assert "Resources/$f present" test -f "$DIST_APP/Contents/Resources/$f"
done
assert "Info.plist present"   test -f "$DIST_APP/Contents/Info.plist"

# The bundle deliberately ships Resources/data/sample_ingredients.csv (sample project +
# template); anything else under a data dir is stray.
STRAY="$(find "$DIST_APP" \( -name '*.pkl' -o -name '__pycache__' -o -name 'results' -o -name 'plots' \) 2>/dev/null; find "$DIST_APP/Contents/Resources/data" -type f ! -name 'sample_ingredients.csv' 2>/dev/null; find "$DIST_APP" -name data -not -path '*/Contents/Resources/data' 2>/dev/null)"
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
assert opt2.X_history[0][2] == 180.0  # first formulation encoded at baseline
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
import wording
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
# One grid holds both types: Type is a cell, and a setting added mid-run
# asks for the baseline the formulations already made were run at. Typing
# into a data editor is injecting its own record of what was typed, which is
# what the browser sends; it has to be injected before every run.
EDITS = {"edited_rows": {}, "deleted_rows": [], "added_rows": [{
    wording.NAME_LABEL: "oven_temp", wording.TYPE_LABEL: wording.KIND_SETTING,
    wording.LOWEST_LABEL: 150.0, wording.HIGHEST_LABEL: 220.0,
    wording.UNIT_LABEL: "C",
    wording.BASELINE_LABEL: 100.0}]}          # outside [150, 220]
at.session_state["ingredient_grid_0"] = dict(EDITS)
at.run()
next(b for b in at.button
     if b.key in ("save_ingredient_grid__save",
                  "save_ingredient_grid__btn")).click()
at.session_state["ingredient_grid_0"] = dict(EDITS)
at.run()
assert not at.exception, at.exception   # a traceback here is the bug
assert any("must be between" in str(e.value) for e in at.error), \
    [str(e.value) for e in at.error]
print("UI_OK")
PY
)"
if echo "$UI_OUT" | grep -q UI_OK; then ok "bundled UI error handling"; else fail "bundled UI error handling ($UI_OUT)"; fi
assert "pkl saved to data dir" test -f "$DATA/E2E_Smoke.pkl"
# Project files must be JSON (safe to open), not executable pickle.
assert "project file is JSON" "$SUPPORT/venv/bin/python" -c "import json,sys; json.load(open(sys.argv[1]))" "$DATA/E2E_Smoke.pkl"

echo "-- test 5c: second wording wave controls are in the packaged app --"
WAVE_OUT="$(cd "$DATA" && HOME="$E2E_HOME" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WORK/$APP_NAME.app/Contents/Resources" \
  APP_RESOURCES="$WORK/$APP_NAME.app/Contents/Resources" \
  "$SUPPORT/venv/bin/python" - <<'PY'
import os
import wording
from streamlit.testing.v1 import AppTest
from food_bo import FoodOptimizer

opt = FoodOptimizer("Wave_Check")
opt.add_ingredient("water", 0.0, 100.0)
opt.add_objective("taste", 1.0, goal="max")
opt.tell({"water": 50.0}, {"taste": 7.0})

at = AppTest.from_file(
    os.path.join(os.environ["APP_RESOURCES"], "app.py"), default_timeout=300)
at.session_state["_loaded_project"] = "Wave_Check"
at.run()
assert not at.exception, at.exception

# Tab 1 - Set up: the measurements grid carries Share of score as the
# column that is typed into, and the score line spells out the share each
# measurement gets. There is no Importance column anywhere (0.5.0).
grid = at.dataframe[1].value
assert [c for c in grid.columns if c != "_id"] == [
    "Measurement", "Goal", "Target", "Lowest measurable",
    "Highest measurable", "Unit", "Share of score (%)"], list(grid.columns)
assert list(grid["Share of score (%)"]) == [100.0]
# The score line is the shares and nothing else now: the weight behind a
# share left the screen with Importance, and the ceiling is always 100.
assert any(c.value.startswith("Overall score = 100 % \u00d7 taste closeness")
           and "scores 100." in c.value
           for c in at.caption), [c.value for c in at.caption]

# Tab 2 - Make a round: the retired Repeat checkbox never comes back, and
# a formulation of your own has an expander to land in instead.
at.session_state["main_tab"] = wording.TAB_BATCH
at.run()
assert not at.exception, at.exception
expander_labels = [e.label for e in at.expander]
assert wording.ADD_OWN_EXPANDER in expander_labels, expander_labels
assert not any(c.label and c.label.startswith("Repeat") for c in at.checkbox), \
    [c.label for c in at.checkbox]

# Tab 3 - Results: one "Edit past formulations" expander replaces the three
# retired controls; none of their old labels survive anywhere on the tab.
at.session_state["main_tab"] = wording.TAB_RESULTS
at.run()
assert not at.exception, at.exception
expander_labels = [e.label for e in at.expander]
assert wording.EDIT_PAST_FORMULATIONS_EXPANDER in expander_labels, expander_labels
retired = ("Correct a result", "Delete a batch or a formulation",
           "Delete the last batch", "Import past formulations from a CSV")
seen = (expander_labels + [b.label for b in at.button]
        + [c.label for c in at.checkbox if c.label])
assert not any(label in seen for label in retired), seen
print("WAVE_OK")
PY
)"
if echo "$WAVE_OUT" | grep -q WAVE_OK; then ok "second wording wave controls in packaged app"; else fail "second wording wave controls in packaged app ($WAVE_OUT)"; fi

echo "-- test 5d: openpyxl importable in the bundled venv --"
assert "import openpyxl" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WORK/$APP_NAME.app/Contents/Resources" \
  "$SUPPORT/venv/bin/python" -c "import openpyxl"

echo "-- test 5e: kitchen-trust controls (0.4.0) are in the packaged app --"
KT_OUT="$(cd "$DATA" && HOME="$E2E_HOME" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WORK/$APP_NAME.app/Contents/Resources" \
  APP_RESOURCES="$WORK/$APP_NAME.app/Contents/Resources" \
  "$SUPPORT/venv/bin/python" - <<'PY'
import os
import wording
from streamlit.testing.v1 import AppTest
from food_bo import FoodOptimizer

opt = FoodOptimizer("Kitchen_Check")
opt.add_ingredient("water", 0.0, 100.0)
opt.add_ingredient("flour", 0.0, 100.0)
opt.add_objective("taste", 1.0, goal="max")
opt.set_formulation_total(50.0)
# Five scored rows: the Compared-with column is drawn only once the cold
# start is over (the first five formulations are spread out, not compared).
for no, water in enumerate((20.0, 22.0, 24.0, 26.0, 28.0), start=1):
    opt.tell({"water": water, "flour": 50.0 - water}, {"taste": 5.0 + no * 0.4},
             formulation_no=no, batch_no=1)
opt.set_pending_batch([{"water": 25.0, "flour": 25.0}])


def _unknown(node, kind, label):
    """AppTest has no accessor for st.download_button or st.file_uploader:
    both arrive as UnknownElement carrying the raw proto."""
    def walk(n):
        children = getattr(n, "children", None) or {}
        if hasattr(children, "values"):
            children = children.values()
        for child in children:
            if (type(child).__name__ == "UnknownElement"
                    and getattr(child, "type", None) == kind
                    and getattr(child.proto, "label", None) == label):
                return child
            found = walk(child)
            if found is not None:
                return found
        return None
    return walk(node)


at = AppTest.from_file(
    os.path.join(os.environ["APP_RESOURCES"], "app.py"), default_timeout=300)
at.session_state["_loaded_project"] = "Kitchen_Check"
at.run()
assert not at.exception, at.exception

# Sidebar: Saved copies in plain words, not the retired backup language.
assert any(m.value == wording.SAVED_COPIES_HEADING for m in at.sidebar.markdown), \
    [m.value for m in at.sidebar.markdown]
assert _unknown(at.sidebar, "download_button", wording.SAVE_A_COPY) is not None
assert _unknown(at.sidebar, "file_uploader", wording.OPEN_A_SAVED_COPY) is not None

# Tab 1 - Set up: the default batch size box, inside More settings.
assert wording.formulation_total_label("g") in [n.label for n in at.number_input], \
    [n.label for n in at.number_input]

# Tab 2 - Make a round: one workbook download, and the last column of the
# round table says what each formulation is trying, against the best so far.
at.session_state["main_tab"] = wording.TAB_BATCH
at.run()
assert not at.exception, at.exception
assert _unknown(at.main, "download_button",
                wording.DOWNLOAD_BATCH_SHEETS) is not None
table = next(d.value for d in at.dataframe
             if any(str(c).startswith("Compared with") for c in d.value.columns))
assert any(str(c).startswith("Compared with") for c in table.columns), \
    list(table.columns)
print("KITCHEN_TRUST_OK")
PY
)"
if echo "$KT_OUT" | grep -q KITCHEN_TRUST_OK; then ok "kitchen-trust controls in packaged app"; else fail "kitchen-trust controls in packaged app ($KT_OUT)"; fi

echo "-- test 5f: spreadsheet-feel controls (0.5.0) are in the packaged app --"
GRID_OUT="$(cd "$DATA" && HOME="$E2E_HOME" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WORK/$APP_NAME.app/Contents/Resources" \
  APP_RESOURCES="$WORK/$APP_NAME.app/Contents/Resources" \
  "$SUPPORT/venv/bin/python" - <<'PY'
import io
import os
import wording
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest
from food_bo import FoodOptimizer

opt = FoodOptimizer("Grid_Check")
opt.add_ingredient("water", 0.0, 100.0)
opt.add_ingredient("flour", 0.0, 100.0)
opt.add_objective("taste", 1.0, goal="max")
opt.set_formulation_total(100.0)
opt.set_pending_batch([{"water": 30.0, "flour": 70.0}])

at = AppTest.from_file(
    os.path.join(os.environ["APP_RESOURCES"], "app.py"), default_timeout=300)
at.session_state["_loaded_project"] = "Grid_Check"
at.run()
assert not at.exception, at.exception

# Tab 1 - Set up: the ingredients list is one editable grid, with Vendor
# and SKU beside the range and Rule last (0.6.0). Baseline joins it only
# once results exist.
assert [c for c in at.dataframe[0].value.columns if c != "_id"] == [
    "Name", "Type", "Made as", "Lowest", "Highest", "Unit",
    "Rule"], list(at.dataframe[0].value.columns)

# Three tiers on the tab: the grids, then More settings, then Advanced.
labels = [e.label for e in at.expander]
assert wording.MORE_SETTINGS_EXPANDER in labels, labels
assert wording.ADVANCED_EXPANDER in labels, labels

# Nothing is written while typing: an edit in hand lights Save changes and
# puts Discard changes beside it. Typing into a data editor is injecting
# its own record of what was typed, which is what the browser sends.
at.session_state["ingredient_grid_0"] = {
    "edited_rows": {0: {wording.HIGHEST_LABEL: 80.0}},
    "deleted_rows": [], "added_rows": []}
at.run()
assert not at.exception, at.exception
# Either key: a save that would take the open round away is drawn by
# confirm_action (__btn) so it can ask first, and a save that would not is
# the plain button (__save). Both say Save changes and both are the one lit
# action on the tab. (No apostrophes in this block: the heredoc sits inside
# a double-quoted command substitution, where one would open a quote bash
# never sees closed.)
save = next(b for b in at.button
            if b.key in ("save_ingredient_grid__save",
                         "save_ingredient_grid__btn"))
assert save.label == wording.SAVE_CHANGES_BUTTON, save.label
assert save.proto.type == "primary", save.proto.type
assert any(b.key == "save_ingredient_grid__discard"
           and b.label == wording.DISCARD_CHANGES_BUTTON for b in at.button), \
    [(b.key, b.label) for b in at.button]

# Tab 2 - Make a round: the round screen's own Batch size box, and the
# round named by number.
assert wording.batch_size_label("g") in [n.label for n in at.number_input], \
    [n.label for n in at.number_input]
assert wording.make_these(1, 1) in [m.value for m in at.main.markdown], \
    [m.value for m in at.main.markdown]

# The workbook is protected, with a Lot cell per ingredient on the round's
# summary page and an Actual (g) column on each formulation page.
opt.set_records("lot", True)
opt.set_records("actual", True)
book = load_workbook(io.BytesIO(opt.workbook_bytes(opt.pending_batch, 100.0)))
summary, page = book[book.sheetnames[0]], book[book.sheetnames[1]]


def filled(sheet):
    return [str(c.value) for row in sheet.iter_rows()
            for c in row if c.value is not None]


assert summary.protection.sheet and page.protection.sheet
assert wording.LOT_COLUMN in filled(summary), filled(summary)
assert opt._actual_column_head() in filled(page), filled(page)
print("GRID_OK")
PY
)"
if echo "$GRID_OUT" | grep -q GRID_OK; then ok "spreadsheet-feel controls in packaged app"; else fail "spreadsheet-feel controls in packaged app ($GRID_OUT)"; fi

echo "-- test 5g: rules controls (0.6.0) are in the packaged app --"
RULES_OUT="$(cd "$DATA" && HOME="$E2E_HOME" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WORK/$APP_NAME.app/Contents/Resources" \
  APP_RESOURCES="$WORK/$APP_NAME.app/Contents/Resources" \
  "$SUPPORT/venv/bin/python" - <<'PY'
import os
import wording
from streamlit.testing.v1 import AppTest

# The sample as a reader meets it: Try the sample project in the sidebar.
# (No apostrophes in this block: the heredoc sits inside a double-quoted
# command substitution, where one would open a quote bash never sees
# closed.)
at = AppTest.from_file(
    os.path.join(os.environ["APP_RESOURCES"], "app.py"), default_timeout=300)
at.run()
assert not at.exception, at.exception
next(b for b in at.button if b.label == wording.TRY_SAMPLE_LABEL).click()
at.run()
assert not at.exception, at.exception

# Tab 1 - Set up: Rule is the last column on the ingredients grid.
grid = at.dataframe[0].value
columns = [c for c in grid.columns if c != "_id"]
assert columns[-1] == wording.FORMULA_LABEL, columns

# The sample gives Water the rest of the batch size, so its amount is not
# typed: Highest reads the one word the whole app uses for it, Lowest is
# blank beside it, and one caption under the grid says what the rule comes
# to in numbers.
water = grid[grid[wording.NAME_LABEL] == "Water"]
assert len(water) == 1, list(grid[wording.NAME_LABEL])
assert water.iloc[0][wording.FORMULA_LABEL] == "= " + wording.REST_TOKEN, \
    water.iloc[0][wording.FORMULA_LABEL]
assert water.iloc[0][wording.LOWEST_LABEL] == "", \
    water.iloc[0][wording.LOWEST_LABEL]
assert water.iloc[0][wording.HIGHEST_LABEL] == wording.WORKED_OUT, \
    water.iloc[0][wording.HIGHEST_LABEL]
captions = [c.value for c in at.caption]
assert any(c.startswith("Water is calculated to bring the total to") for c in captions), captions

# More settings - Limits: one Kind picker choosing the shape of the limit,
# and because the sample has a default batch size the "Write it as" choice
# offers % of default batch size too.
kind_select = next((s for s in at.selectbox if s.key == "qc_kind"), None)
assert kind_select is not None, [s.key for s in at.selectbox]
assert list(kind_select.options) == wording.LIMIT_KINDS, kind_select.options
unit_select = next((s for s in at.selectbox if s.key == "qc_unit"), None)
assert unit_select is not None, [s.key for s in at.selectbox]
assert unit_select.label == wording.LIMIT_WRITTEN_AS_LABEL, unit_select.label
assert wording.PERCENT_OF_BATCH_SIZE_UNIT in unit_select.options, \
    unit_select.options
print("RULES_OK")

# Both pre-mix modes and the preparation pages are in the shipped app.
import io
from openpyxl import load_workbook
opt = at.session_state["optimizer"]
assert list(grid[wording.NAME_LABEL]) == ["Textured pea protein", "Dry blend", "Wheat gluten", "Fats and oils", "Seasoning blend", "Water", "Mixing time after fat"]
assert wording.MADE_AS_LABEL in columns
assert opt.premixes["Fats and oils"]["mode"] == "weighed"
assert len(opt.premixes["Dry blend"]["parts"]) == 3
assert opt._by_name()["Seasoning blend"]["bounds"] == (2.2, 2.2)
opt.ask(3)
book = load_workbook(io.BytesIO(opt.workbook_bytes(opt.pending_batch, 100)))
assert book.sheetnames[:2] == ["Pre-mix · Dry blend", "Pre-mix · Seasoning blend"]
assert book.active["A1"].value.startswith("Dry blend · make ")
assert "this round needs" in book.active["A1"].value
assert book.active.protection.sheet and not book.active["D4"].protection.locked
summary = book[wording.batch_sheet_name(opt.pending_batch_no)]
assert wording.MAKE_FOR_ROUND_HEADING in [c.value for row in summary for c in row]
page = book[wording.formulation_sheet_name(opt.pending_batch[0]["formulation"])]
assert any(c.value == "Coconut oil" and c.alignment.indent == 1 for row in page for c in row)
compact = load_workbook(io.BytesIO(opt.workbook_bytes(opt.pending_batch, 100, print_pack=False)))
assert compact.sheetnames == ["Round overview", "Preparation", "Results"]
assert compact.active.title == "Round overview"
assert any(str(c.value).startswith("Cook loss (%)") for row in compact["Results"] for c in row)
print("PREMIX_OK")
# Committed manual fields survive a fresh application session.
at.run()
number = opt.pending_batch[0]["formulation"]
at.number_input(key=f"f{number}_Firmness").set_value(6).run()
fresh = AppTest.from_file(
    os.path.join(os.environ["APP_RESOURCES"], "app.py"), default_timeout=300)
fresh.session_state["_loaded_project"] = opt.project_name
fresh.run()
assert not fresh.exception, fresh.exception
assert fresh.number_input(key=f"f{number}_Firmness").value == 6
assert not next(b for b in fresh.button if b.label == wording.SAVE_RESULTS).disabled
print("DRAFT_OK")
PY
)"
if echo "$RULES_OUT" | grep -q RULES_OK; then ok "rules controls in packaged app"; else fail "rules controls in packaged app ($RULES_OUT)"; fi

if echo "$RULES_OUT" | grep -q PREMIX_OK; then ok "pre-mix sample and workbook in packaged app"; else fail "pre-mix sample and workbook in packaged app ($RULES_OUT)"; fi

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
rm -f "$DMG"   # test-version dmg; on success dist/ keeps only release dmgs
