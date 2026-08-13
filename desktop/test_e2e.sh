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

# LEVEL2_MARKER: Task 8 replaces everything below this line.
hdiutil detach "$VOL" >/dev/null 2>&1 || true
echo "== DONE: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ] || exit 1
