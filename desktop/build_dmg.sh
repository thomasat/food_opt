#!/bin/bash
# Builds "Food Optimizer.app" and packages it into a .dmg.
# Usage: ./desktop/build_dmg.sh [version]      (default 0.1.0)
# Distribution builds: set SIGN_IDENTITY ("Developer ID Application: ...")
# and NOTARY_PROFILE (a `xcrun notarytool store-credentials` profile name).
set -euo pipefail

VERSION="${1:-0.1.0}"
APP_NAME="Food Optimizer"

DESKTOP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$DESKTOP_DIR")"
DIST_DIR="$DESKTOP_DIR/dist"
APP_DIR="$DIST_DIR/$APP_NAME.app"
DMG_PATH="$DIST_DIR/FoodOptimizer-$VERSION.dmg"

UV_BIN="$("$DESKTOP_DIR/fetch_uv.sh")"

# ---------- assemble the bundle ----------
rm -rf "$APP_DIR" "$DMG_PATH"
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
for f in app.py food_bo.py storage.py ui_helpers.py ui_setup.py ui_batch.py ui_results.py wording.py; do
  cp "$REPO_DIR/$f" "$APP_DIR/Contents/Resources/$f"
done
mkdir -p "$APP_DIR/Contents/Resources/data"
cp "$REPO_DIR/data/sample_ingredients.csv" "$APP_DIR/Contents/Resources/data/sample_ingredients.csv"
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
    xcrun notarytool submit "$DMG_PATH" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$DMG_PATH"
  else
    echo "WARNING: signed but NOT notarized (NOTARY_PROFILE unset) - Gatekeeper will warn." >&2
  fi
fi

echo "Built: $DMG_PATH"
