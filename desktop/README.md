# Desktop packaging (macOS)

Everything needed to ship Food Optimizer as a double-clickable macOS app.
Design spec: `docs/superpowers/specs/2026-08-13-desktop-packaging-design.md`.

## Build

```bash
./desktop/build_dmg.sh 0.1.0        # version is the only argument
```

Output: `desktop/dist/FoodOptimizer-0.1.0.dmg`. Unsigned builds print a
loud warning and are for internal testing only — never send one to a
recipient.

## Test

```bash
./desktop/test_e2e.sh               # levels 1 + 2 (full E2E, ~5–20 min first time)
./desktop/test_e2e.sh --level1-only # fast static + build checks
```

Level 2 downloads ~2 GB of dependencies into a throwaway temp HOME
(deleted afterward); it never touches your real environment.

## Signing + notarization (required before distributing)

One-time setup:
1. Join the Apple Developer Program ($99/yr) with the org's Apple ID.
2. In Xcode (Settings → Accounts) or developer.apple.com, create a
   **Developer ID Application** certificate and install it in your keychain.
3. Create an app-specific password at appleid.apple.com, then store it:

```bash
xcrun notarytool store-credentials foodopt-notary \
  --apple-id YOUR_APPLE_ID --team-id YOUR_TEAM_ID \
  --password YOUR_APP_SPECIFIC_PASSWORD
```

Then every distribution build:

```bash
export SIGN_IDENTITY="Developer ID Application: Your Name (YOURTEAMID)"
export NOTARY_PROFILE=foodopt-notary
./desktop/build_dmg.sh 0.1.0
```

## Updating dependencies

When `requirements.txt` changes, regenerate the lock and commit it:

```bash
./desktop/update_lock.sh
```

The launcher detects the changed lock hash and re-runs setup on users'
machines automatically.

## Publishing a release (the download link you send people)

Releases are published as **GitHub Releases** — each version gets a
permanent download URL. Never commit dmg files to the repo.

The full ritual for version X.Y.Z:

```bash
# 0. Only if requirements.txt changed since the last release:
./desktop/update_lock.sh            # then commit the updated lock

# 1. Build (set SIGN_IDENTITY + NOTARY_PROFILE first for real distributions)
./desktop/build_dmg.sh X.Y.Z

# 2. Verify
./desktop/test_e2e.sh               # must end "0 failed"
# ...plus the manual checklist below before sending to real recipients.

# 3. Make sure the release is cut from main (merge + push first), then:
gh release create vX.Y.Z "desktop/dist/FoodOptimizer-X.Y.Z.dmg" \
  --title "Food Optimizer X.Y.Z" \
  --notes "See 'Start Here.txt' inside the download for install steps."
```

The link to send people (also shown on the release page):

```
https://github.com/thomasat/food_opt/releases/download/vX.Y.Z/FoodOptimizer-X.Y.Z.dmg
```

Notes:
- The repo is public, so the link works for anyone — no GitHub account needed.
- Unsigned builds are for internal testers only. Before a link goes to
  real recipients, complete the signing setup above and rebuild — an
  unsigned dmg triggers the exact security warnings this project exists
  to avoid.
- To replace a release's dmg (e.g. after signing):
  `gh release upload vX.Y.Z <dmg> --clobber`

## System requirements (copy-paste for emails / download page)

> Food Optimizer runs on Macs with an Apple chip (M1 or later — any Mac
> from late 2020 onward) on macOS 13 Ventura or newer, and needs about
> 5 GB of free disk space. The first launch needs an internet connection
> for a one-time 2–4 minute setup; after that it works fully offline.
> Your data never leaves your computer.

## Manual checklist before every distribution

1. `./desktop/test_e2e.sh` passes.
2. On a Mac that is NOT the build machine: download the dmg like a
   recipient would (browser/AirDrop), mount, drag to Applications,
   double-click. **It must open with no security warning of any kind.**
   Any warning = distribution blocked; fix signing/notarization.
3. Double-click the app *inside* the dmg window on purpose - using a
   downloaded, quarantined dmg, so Gatekeeper runs it from a
   translocated path: the "copy to Applications first?" offer appears
   and works.
4. Setup dialog appears, finishes; browser opens with the app.
5. UI walkthrough: create a project, upload `data/ingredients.csv` and
   `data/experiments_example.csv`, request a suggestion batch, log a
   result, confirm the `.pkl` appears in `~/Documents/FoodOptimizer/`.
6. Close the browser tab, wait past the idle timeout (15 min), confirm
   the server exited (Activity Monitor). Relaunch: fast, project loads.
7. Wi-Fi off on a set-up machine: works fully offline.
8. Wi-Fi off on a fresh machine: plain-language "needs internet once"
   dialog appears; succeeds after Wi-Fi is restored.
9. Read "Start Here.txt" and every dialog as a skeptical nontechnical
   user: accurate, understandable, and every error has a next step.
10. Upgrade path: with an older version already in /Applications,
    install the new dmg over it and open - no "damaged app" warning,
    and the new version runs.
