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

The build compiles the native window wrapper (`FoodOptimizerApp.swift`)
with `swiftc`, so the build machine needs the Xcode Command Line Tools
(`xcode-select --install`). Recipients need nothing extra.

## Test

```bash
./desktop/test_e2e.sh               # levels 1 + 2 (full E2E, ~5–20 min first time)
./desktop/test_e2e.sh --level1-only # fast static + build checks
```

Level 2 downloads ~1 GB of dependencies into a throwaway temp HOME
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
> 6 GB of free disk space. The first launch needs an internet connection
> for a one-time setup that downloads about 1 GB. On a good connection this
> takes under a minute; on a slow office network up to 15. After that it
> works fully offline.
> Your data never leaves your computer.

## Testing an unsigned build on another Mac

A browser download stamps the dmg with macOS's quarantine flag, and an
unsigned build fails that inspection: the app shows **"Food Optimizer is
damaged and can't be opened"**, and even `Start Here.txt` on the mounted
image can show a bogus **"you don't have permission"** error (sandboxed
apps refuse documents on quarantined volumes). The file is fine — this is
Gatekeeper rejecting the missing signature.

Internal testers only — remove the flag before mounting:

```bash
xattr -d com.apple.quarantine ~/Downloads/FoodOptimizer-X.Y.Z.dmg
```

(No output = success. If the app was already copied to /Applications from
a quarantined mount, delete it and drag it again from the cleaned image.)

Real recipients must never need this: signing + notarization removes the
warnings entirely, and the checklist below gates distribution on that.

## Manual checklist before every distribution

1. `./desktop/test_e2e.sh` passes.
2. On a Mac that is NOT the build machine: download the dmg like a
   recipient would (browser/AirDrop), mount, drag to Applications,
   double-click. **It must open with no security warning of any kind.**
   Any warning = distribution blocked; fix signing/notarization.
3. Double-click the app *inside* the dmg window on purpose - using a
   downloaded, quarantined dmg, so Gatekeeper runs it from a
   translocated path: the "copy to Applications first?" offer appears
   and works — accepting it must quit this copy and automatically
   reopen the app from /Applications (a detached helper does the
   relaunch; watch that the window actually comes back).
4. The app window opens immediately with the setup message, then loads
   the app when setup finishes. Dock shows the Food Optimizer icon and
   name (not a browser).
5. In-window plumbing: upload a CSV (file picker opens), download a
   backup (lands in ~/Downloads), Cmd-Q and window-close both stop the
   app completely (check Activity Monitor: no streamlit left).
6. UI walkthrough: click "Try the sample project" (eight ingredients,
   Juiciness and Firmness), go to Make a batch and generate one, record
   a result, confirm the `.pkl` appears in `~/FoodOptimizer/`.
7. Quit, relaunch: fast start, project still listed and loads.
8. Wi-Fi off on a set-up machine: works fully offline.
9. Wi-Fi off on a fresh machine: plain-language "needs internet once"
   message appears; succeeds after Wi-Fi is restored.
10. Read "Start Here.txt" and every message as a skeptical nontechnical
    user: accurate, understandable, and every error has a next step.
11. Upgrade path: with an older version already in /Applications,
    install the new dmg over it and open - no "damaged app" warning,
    and the new version runs.
