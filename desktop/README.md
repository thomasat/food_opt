# Desktop packaging (macOS)

Everything needed to ship Food Optimizer as a double-clickable macOS app.
Design spec: `docs/superpowers/specs/2026-08-13-desktop-packaging-design.md`.

## Build

```bash
./desktop/build_dmg.sh 0.6.0        # version is the only argument
```

Output: `desktop/dist/FoodOptimizer-0.6.0.dmg`. Unsigned builds print a
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
./desktop/build_dmg.sh 0.6.0
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

> Food Optimizer runs on Macs with Apple silicon (M1 or newer)
> on macOS 13 Ventura or newer, and needs about
> 6 GB of free disk space. The first launch needs an internet connection
> for a one-time setup that downloads about 1 GB. This usually takes under a minute; on a slow network, a few minutes. After that it works fully offline.
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
6. UI walkthrough: click "Try the sample project" (six rows and ten
   parts, Firmness, Juiciness and Cook loss, formulations of 100 g, one
   process setting and one fat limit). Dry blend is portioned; Fat phase
   holds two oils weighed into each formulation; Seasoning blend stays at
   2.2 g. Water's Rule cell says "= rest", so its
   Highest reads "worked out" and one line under the grid gives
   what it comes to in numbers. On "1 · Set up", type
   into the ingredients grid and check that "Save changes" lights up and
   "Discard changes" puts the grid back; open "More settings" and
   "Advanced" and check nothing else sits on the tab. In "More settings",
   check that an ingredient limit offers a "Limit is" choice of "At
   least", "At most", "Between" or "Exactly" - one box, or two for
   "Between" -
   and that "Write it as" offers "% of default batch size" (it is
   there only while a default batch size is set), and that picking an
   ingredient does not close "More settings" under you. Go to "2 · Make a
   round", generate one, check every row of the round table totals the
   batch size and that Water's column is headed "Water · worked out (g)",
   change "Batch size (g)" to 250 and watch every amount
   in the round table follow it, download the round sheets (Excel) and
   open it: the sheets
   are protected, only the cells the instruction line names take a value,
   and a printed greyscale copy makes clear which they are — every one of
   them is boxed, and boxed and shaded are the one treatment: there is no
   second, shaded-but-locked kind of cell. Each formulation page has an
   "Actual (g)" column beside its amounts, with a "Lot" cell per
   ingredient on the "Round 1" page. Water is printed as "Water · worked
   out" on the formulation pages and "Water · worked out (g)" on the
   round sheet — the mark on the name, the unit behind it, one shape
   everywhere, with the line "Water is worked out: = rest
   (batch size − every other ingredient). Weigh the amount printed."
   under the amounts. Record a result (or mark one Not
   scored and score it later from Results), and confirm the `.pkl` appears
   in `~/FoodOptimizer/`.
6a. On "3 · Results", click "Download all formulations (Excel)" and open
   it: the "Set up" sheet in THAT file is the one with the Rule column,
   and it spells Water out as "= rest (batch size − every other
   ingredient)" with "worked out" in its Status cell. (The round sheets
   in step 6 are a different download and carry no Set-up sheet.)
7. Quit, relaunch: fast start, project still listed and loads.
8. Wi-Fi off on a set-up machine: works fully offline.
9. Wi-Fi off on a fresh machine: plain-language "needs internet once"
   message appears; succeeds after Wi-Fi is restored.
10. Read "Start Here.txt" and every message as a skeptical nontechnical
    user: accurate, understandable, and every error has a next step.
11. Upgrade path: with an older version already in /Applications,
    install the new dmg over it and open - no "damaged app" warning,
    and the new version runs.

Pre-mix workbook check: preparation pages for Dry blend and Seasoning blend
come before the round summary, each headed "make N g (this round needs M g)"
and each ending in a "Pre-mix lot" line; every formulation page indents the
Fat phase parts under the pre-mix they belong to. Enter a part lot number on
a preparation page and an actual oil amount on a formulation page, then
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
recording options off; the sample enables lot numbers and actual amounts. Turning a field off keeps values already saved.
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
version. The sample targets and method are illustrative, not measured results.
