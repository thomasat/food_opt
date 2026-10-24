#!/bin/bash
# Builds "Food Optimizer.app" and packages it into a .dmg. The version is
# desktop/VERSION; there is no other.
#   ./desktop/build_dmg.sh [X.Y.Z]          release build -> dist/FoodOptimizer-X.Y.Z.dmg
#                                           (X.Y.Z, if given, must equal desktop/VERSION)
#   ./desktop/build_dmg.sh --check [X.Y.Z]  the release guards only; builds nothing
#   ./desktop/build_dmg.sh --test-build     internal build, guards off
#                                           -> dist/FoodOptimizer-X.Y.Z-test.dmg
# A release build refuses a tree with uncommitted changes and a HEAD that is
# not tag vX.Y.Z: build from a clean worktree at the tag. Every refusal exits 2.
# Distribution builds: set SIGN_IDENTITY ("Developer ID Application: ...")
# and NOTARY_PROFILE (a `xcrun notarytool store-credentials` profile name).
set -euo pipefail

USAGE="usage: build_dmg.sh [X.Y.Z] | --check [X.Y.Z] | --test-build"
usage() { echo "$USAGE" >&2; exit 2; }
refuse() { echo "refusing to build: $*" >&2; exit 2; }

# ---------- arguments ----------
MODE="release"
WANTED=""
case "${1:-}" in
  --check)
    MODE="check"
    [ $# -le 2 ] || usage
    WANTED="${2:-}"
    ;;
  --test-build)
    MODE="test"
    [ $# -eq 1 ] || usage
    ;;
  -*) usage ;;
  *)
    [ $# -le 1 ] || usage
    WANTED="${1:-}"
    ;;
esac
case "$WANTED" in -*) usage ;; esac

APP_NAME="Food Optimizer"
DESKTOP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$DESKTOP_DIR")"
DIST_DIR="$DESKTOP_DIR/dist"
APP_DIR="$DIST_DIR/$APP_NAME.app"

# ---------- version ----------
# One X.Y.Z line. Exactly one trailing newline is taken off; a carriage
# return (a file saved on Windows) or anything else fails.
VERSION_RE='^[0-9]+\.[0-9]+\.[0-9]+$'
VERSION=""
if [ -f "$DESKTOP_DIR/VERSION" ]; then
  VERSION="$(cat "$DESKTOP_DIR/VERSION"; printf x)"
  VERSION="${VERSION%x}"
  VERSION="${VERSION%$'\n'}"
fi
[[ "$VERSION" =~ $VERSION_RE ]] || refuse "desktop/VERSION must hold one X.Y.Z line"

# ---------- release guards (release builds and --check) ----------
if [ "$MODE" != "test" ]; then
  if [ -n "$WANTED" ] && [ "$WANTED" != "$VERSION" ]; then
    refuse "version $WANTED disagrees with desktop/VERSION ($VERSION)"
  fi
  # Ignored paths (dist/, .cache/) don't count; untracked ones do.
  STATUS="$(git -C "$REPO_DIR" status --porcelain --untracked-files=normal)" \
    || refuse "not a git checkout (build from a clean worktree at the tag)"
  [ -z "$STATUS" ] || refuse "uncommitted changes (build from a clean worktree at the tag)"
  HEAD_SHA="$(git -C "$REPO_DIR" rev-parse HEAD)"
  TAG_SHA="$(git -C "$REPO_DIR" rev-parse -q --verify "refs/tags/v$VERSION^{commit}")" || TAG_SHA=""
  if [ "$HEAD_SHA" != "$TAG_SHA" ]; then
    refuse "HEAD is not tag v$VERSION (tag first: git tag -a v$VERSION $(git -C "$REPO_DIR" rev-parse --short HEAD))"
  fi
  if [ "$MODE" = "check" ]; then
    echo "ready to build $VERSION from v$VERSION"
    exit 0
  fi
  DMG_NAME="FoodOptimizer-$VERSION"
else
  echo "=========================================================" >&2
  echo " TEST BUILD - not for release (no version or tree checks)." >&2
  echo "=========================================================" >&2
  DMG_NAME="FoodOptimizer-$VERSION-test"
fi
DMG_PATH="$DIST_DIR/$DMG_NAME.dmg"
RECORD="$DIST_DIR/$DMG_NAME.build.txt"

UV_BIN="$("$DESKTOP_DIR/fetch_uv.sh")"

# ---------- assemble the bundle ----------
rm -rf "$APP_DIR" "$DMG_PATH" "$RECORD"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

# launcher.sh lives in Resources, NOT MacOS: codesign treats everything in
# Contents/MacOS as code that must carry its own signature, which shell
# scripts can't do robustly — in Resources the bundle signature seals it.
cp "$DESKTOP_DIR/launcher.sh" "$APP_DIR/Contents/Resources/launcher.sh"
cp "$UV_BIN" "$APP_DIR/Contents/MacOS/uv"
chmod 755 "$APP_DIR/Contents/Resources/launcher.sh" "$APP_DIR/Contents/MacOS/uv"

# Native window wrapper (the bundle executable). Requires Xcode Command
# Line Tools on the build machine; recipients need nothing extra.
echo "compiling native window wrapper..."
xcrun swiftc -O -target arm64-apple-macos13.0 \
  "$DESKTOP_DIR/FoodOptimizerApp.swift" \
  -o "$APP_DIR/Contents/MacOS/FoodOptimizer"
chmod 755 "$APP_DIR/Contents/MacOS/FoodOptimizer"

# theory.py is deliberately not bundled: nothing in the app imports it, and
# it needs matplotlib, which left requirements.txt with the deploy cleanup.
for f in app.py food_bo.py storage.py ui_helpers.py ui_setup.py ui_batch.py ui_results.py insights.py ui_insights.py bo_views.py calculation_editor.py wording.py workbook_flow.py sample_projects.py custom_records.py; do
  cp "$REPO_DIR/$f" "$APP_DIR/Contents/Resources/$f"
done
mkdir -p "$APP_DIR/Contents/Resources/data/samples/yogurt_pilot"
cp "$REPO_DIR/data/ingredients_template.csv" "$APP_DIR/Contents/Resources/data/ingredients_template.csv"
for f in "Yogurt pilot - Round 6 to score.json" "Yogurt pilot - all 10 Rounds.json" "Round 6 results (pilot).xlsx"; do
  cp "$REPO_DIR/data/samples/yogurt_pilot/$f" "$APP_DIR/Contents/Resources/data/samples/yogurt_pilot/$f"
done
mkdir -p "$APP_DIR/Contents/Resources/data/samples/burger_setup"
cp "$REPO_DIR/data/samples/burger_setup/Example setup - plant-based burger.json" "$APP_DIR/Contents/Resources/data/samples/burger_setup/Example setup - plant-based burger.json"
cp "$DESKTOP_DIR/requirements.lock.txt" "$APP_DIR/Contents/Resources/requirements.lock.txt"
cp "$DESKTOP_DIR/icon.icns" "$APP_DIR/Contents/Resources/icon.icns"

cp "$DESKTOP_DIR/Info.plist" "$APP_DIR/Contents/Info.plist"
plutil -replace CFBundleShortVersionString -string "$VERSION" "$APP_DIR/Contents/Info.plist"
plutil -replace CFBundleVersion -string "$VERSION" "$APP_DIR/Contents/Info.plist"

# ---------- sign (distribution builds) ----------
if [ -n "${SIGN_IDENTITY:-}" ]; then
  echo "signing with: $SIGN_IDENTITY"
  codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_DIR/Contents/MacOS/uv"
  codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_DIR/Contents/MacOS/FoodOptimizer"
  codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_DIR"
  codesign --verify --strict "$APP_DIR"
else
  echo "=========================================================" >&2
  echo " WARNING: UNSIGNED build - internal testing only." >&2
  echo " Set SIGN_IDENTITY and NOTARY_PROFILE for distribution." >&2
  echo "=========================================================" >&2
fi

# ---------- package the dmg ----------
STAGING="$DIST_DIR/dmg-staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"
ditto "$APP_DIR" "$STAGING/$APP_NAME.app"
ln -s /Applications "$STAGING/Applications"
cp "$DESKTOP_DIR/start_here.txt" "$STAGING/Start Here.txt"
# No example-data folder: the app's "Try the sample project" button carries
# the same ingredients plus measurements, and the welcome panel offers the
# CSV template. One route to the sample keeps first-run instructions
# unambiguous.

if command -v create-dmg >/dev/null 2>&1; then
  create-dmg \
    --volname "$APP_NAME" \
    --window-size 640 400 \
    --icon-size 96 \
    --icon "$APP_NAME.app" 160 190 \
    --icon "Applications" 480 190 \
    --icon "Start Here.txt" 320 60 \
    --hide-extension "$APP_NAME.app" \
    "$DMG_PATH" "$STAGING"
else
  echo "create-dmg not found - building plain dmg (brew install create-dmg for the styled layout)" >&2
  hdiutil create -volname "$APP_NAME" -srcfolder "$STAGING" -ov -format UDZO "$DMG_PATH"
fi
rm -rf "$STAGING"

# ---------- notarize (distribution builds) ----------
if [ -n "${SIGN_IDENTITY:-}" ]; then
  codesign --force --timestamp --sign "$SIGN_IDENTITY" "$DMG_PATH"
  if [ -n "${NOTARY_PROFILE:-}" ]; then
    echo "submitting for notarization (this can take a few minutes)..."
    NOTARY_ARGS=(--keychain-profile "$NOTARY_PROFILE")
    if [ -n "${NOTARY_KEYCHAIN:-}" ]; then
      NOTARY_ARGS+=(--keychain "$NOTARY_KEYCHAIN")
    fi
    xcrun notarytool submit "$DMG_PATH" "${NOTARY_ARGS[@]}" --wait
    xcrun stapler staple "$DMG_PATH"
  else
    echo "WARNING: signed but NOT notarized (NOTARY_PROFILE unset) - Gatekeeper will warn." >&2
  fi
fi

# ---------- build record (written last, after stapling) ----------
yes_no() { if [ -n "$1" ]; then echo yes; else echo no; fi; }
NOTARIZED=""
if [ -n "${SIGN_IDENTITY:-}" ] && [ -n "${NOTARY_PROFILE:-}" ]; then NOTARIZED=1; fi
TREE_CLEAN=no
if NOW="$(git -C "$REPO_DIR" status --porcelain --untracked-files=normal 2>/dev/null)" \
    && [ -z "$NOW" ]; then
  TREE_CLEAN=yes
fi
{
  echo "version=$VERSION"
  echo "commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "tree_clean=$TREE_CLEAN"
  echo "mac_lock_sha256=$(shasum -a 256 "$DESKTOP_DIR/requirements.lock.txt" | awk '{print $1}')"
  echo "windows_lock_sha256=none"
  echo "dmg_sha256=$(shasum -a 256 "$DMG_PATH" | awk '{print $1}')"
  echo "signed=$(yes_no "${SIGN_IDENTITY:-}")"
  echo "notarized=$(yes_no "$NOTARIZED")"
} > "$RECORD"

echo "Built: $DMG_PATH"
echo "Build record: $RECORD"
