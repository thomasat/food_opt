# Desktop packaging (macOS)

Everything needed to ship Food Optimizer as a double-clickable macOS app.
Design spec: `docs/superpowers/specs/2026-08-13-desktop-packaging-design.md`.

## Build

The version is `desktop/VERSION` (one `X.Y.Z` line); nothing else sets it.

```bash
./desktop/build_dmg.sh --check      # can I build? guards only, builds nothing
./desktop/build_dmg.sh              # release build of desktop/VERSION
./desktop/build_dmg.sh --test-build # internal testing only
```

- A **release build** writes `desktop/dist/FoodOptimizer-X.Y.Z.dmg`. It
  refuses (exit 2) a tree with uncommitted changes, a HEAD that is not tag
  `vX.Y.Z`, and a version argument that disagrees with `desktop/VERSION`:
  build it from a clean worktree at the tag (see the ritual below).
  `--check` runs the same guards and prints `ready to build X.Y.Z from vX.Y.Z`.
- A **test build** skips the guards, prints a loud `TEST BUILD - not for
  release` warning and writes `desktop/dist/FoodOptimizer-X.Y.Z-test.dmg`.
  `test_e2e.sh` makes one. Never send it to a recipient.
- Every build writes a **build record** beside the dmg,
  `desktop/dist/FoodOptimizer-X.Y.Z.build.txt`: version, commit, whether the
  tree was clean, both lock hashes, the dmg's sha256, signed, notarized.
- Unsigned builds print a loud warning and are for internal testing only.

The build compiles the native window wrapper (`FoodOptimizerApp.swift`)
with `swiftc`, so the build machine needs the Xcode Command Line Tools
(`xcode-select --install`). Recipients need nothing extra.

## What ships

`desktop/shared/bundle_manifest.txt` lists the files both apps ship: one
repo-relative path per line, sorted, LF line endings, no comments. The Mac
build copies each to `Contents/Resources/<same path>`; `build_dmg.sh` names
no app file itself. `desktop/mac_bundle_files.txt` is the exact listing of
the built `Contents/`, and `test_e2e.sh` fails on any difference.

To ship a new module or data file:

1. add it to `desktop/shared/bundle_manifest.txt`;
2. add `Resources/<path>` (and any new folder) to `desktop/mac_bundle_files.txt`;
3. run `./desktop/test_e2e.sh`.

The golden list is edited by hand on purpose: a file reaching the bundle
is a decision, not a side effect.

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
./desktop/build_dmg.sh              # in a clean worktree at the tag
```

## Updating dependencies

When `requirements.txt` changes, run `./desktop/update_lock.sh` and commit
both locks it writes:

- **Run it on an Apple Silicon Mac.** It refuses anywhere else, because the
  Mac lock is resolved for the machine it runs on.
- **The Mac lock (`desktop/requirements.lock.txt`) is rewritten only when its
  packages or hashes change.** `launcher.sh` hashes the whole file, so any
  byte change, even a comment, makes every Mac user redo the ~1 GB online
  setup.
- **The Windows lock (`desktop/requirements-windows.lock.txt`) is derived
  from the Mac lock:** the same versions, plus the packages only Windows
  needs (`colorama`, `watchdog`). `tests/test_locks.py` checks they agree.
- **Commit both.** When the Mac lock changed, the script says so: re-measure
  `EXPECTED_SETUP_MB` in `launcher.sh` and update `MAC_LOCK_SHA256` in
  `tests/test_locks.py`.

## Publishing a release (the download link you send people)

Releases are published as **GitHub Releases** — each version gets a
permanent download URL. Never commit dmg files to the repo.

The ritual for version X.Y.Z (Mac half):

1. Only if `requirements.txt` changed since the last release: run
   `./desktop/update_lock.sh` and commit the result.
2. Release PR (true merge): bump `desktop/VERSION`, the README download
   links, and write `desktop/release_notes/X.Y.Z.md`, the text of the
   release page. The tests check all three agree.
3. Tag first; the build comes from the tag:

   ```bash
   git tag -a vX.Y.Z <sha>
   git push origin vX.Y.Z
   ```

4. Build and verify in a clean worktree at the tag, signing set up:

   ```bash
   git worktree add ../food_opt-vX.Y.Z vX.Y.Z
   cd ../food_opt-vX.Y.Z
   export SIGN_IDENTITY="Developer ID Application: Your Name (YOURTEAMID)"
   export NOTARY_PROFILE=foodopt-notary
   ./desktop/build_dmg.sh            # refuses a dirty tree or a HEAD that is not the tag
   ./desktop/test_e2e.sh             # must end "0 failed"
   ```

   ...plus the manual checklist below before sending to real recipients.
5. Draft the release, then compare GitHub's asset digest with `dmg_sha256`
   in `desktop/dist/FoodOptimizer-X.Y.Z.build.txt`:

   ```bash
   gh release create vX.Y.Z desktop/dist/FoodOptimizer-X.Y.Z.dmg --draft --verify-tag \
     --title "Food Optimizer X.Y.Z" --notes-file desktop/release_notes/X.Y.Z.md
   gh release view vX.Y.Z --json assets
   ```

6. Publish, then check the README link answers:

   ```bash
   gh release edit vX.Y.Z --draft=false --latest
   curl -I https://github.com/thomasat/food_opt/releases/download/vX.Y.Z/FoodOptimizer-X.Y.Z.dmg
   ```

   Keep the time between steps 2 and 6 short: the README link 404s in between.

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

> Food Optimizer runs on Macs with Apple silicon (M1 or newer)
> on macOS 13 Ventura or newer, and needs about
> 6 GB of free disk space. The first launch needs an internet connection
> for a one-time setup that downloads about 1 GB. Setup time depends on your connection and computer. After that it works fully offline.
> Your data never leaves your computer.

## Testing an unsigned build on another Mac

A browser download stamps the dmg with macOS's quarantine flag, and an
unsigned build fails that inspection: the app shows **"Food Optimizer is
damaged and can't be opened"**, and even `Start Here.txt` on the mounted
image can show a bogus **"you don't have permission"** error (sandboxed
apps refuse documents on quarantined volumes). The file is fine — this is
Gatekeeper rejecting the missing signature.

Internal testers only — clear the flag before mounting:

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
5. In-window plumbing: upload an ingredients file (file picker opens), save a
   copy (lands in ~/Downloads), Cmd-Q and window-close both stop the
   app completely (check Activity Monitor: no streamlit left).
6. Run the demo walkthrough in `shared/manual_walkthrough.md`.
7. Quit, relaunch: fast start, project still listed and loads.
8. Wi-Fi off on a set-up machine: works fully offline.
9. Wi-Fi off on a fresh machine: plain-language "needs internet once"
   message appears; succeeds after Wi-Fi is restored.
10. Read "Start Here.txt" and every message as a skeptical nontechnical
    user: accurate, understandable, and every error has a next step.
11. Upgrade path: with an older version already in /Applications,
    install the new dmg over it and open - no "damaged app" warning,
    and the new version runs.

Pre-mix workbook check, on a project of your own with a pre-mix and a
blend: each pre-mix's preparation page comes before the round summary, each headed "make N g (this round needs M g)"
and each ending in a "Pre-mix lot" line; every formulation page indents the
blend's parts under the blend they belong to. Enter a part lot number on
a preparation page and an actual amount on a formulation page, then
upload the completed workbook and verify both are retained. The summary
carries "Make for this round" and "Have on hand", and the project's Method
under the round's amounts.

For notarization credentials stored outside the login keychain, set
`NOTARY_KEYCHAIN` to that keychain's path alongside `NOTARY_PROFILE`.


Optional recording and returning the sheets

In Set up → Preparation and records, choose optional recording fields.
Vendor and SKU appear in the Ingredients table. Lot numbers are entered in
Make a round → Ingredient lot numbers or in the workbook. A separate option
records changes from planned amounts or settings. New projects start with
recording options off, as do the samples. Turning a field off keeps values already saved.
After filling in the workbook, upload it in 2 · Make a round → Save results
→ Or upload results from a file, check the preview, then save. Editing Excel
alone does not update the app. Preparation amounts and other filled-in boxes
are preserved as Bench records in the All formulations export; preparation
records do not change a pre-mix's percentages for future rounds. Formulation
Actual amounts are used as the amounts made. The Set-up and All formulations
sheets are records, rather than forms to fill in and send back.
Saved copies shows the three newest copies and puts the rest under Older
copies. Its confirmed cleanup keeps the newest three and the last seven days.
For a process or fermentation study, use numeric process settings. Record
separate measurements for different time points, such as pH at 6 h and pH at
24 h. Named categories such as strain or vessel type are not varied by this
version. The yogurt sample is a real, public yogurt-style pilot: its results were measured.
The example setup (a plant-based burger) is illustrative: no results.
