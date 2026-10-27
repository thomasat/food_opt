# Food Optimizer for Windows: design note

One page. The full spec is `docs/superpowers/specs/2026-10-23-windows-port-design.md`
(not committed); this is the committed summary. Build and IT facts are in
`README.md`.

## Why a thin shell and a shared launcher

- The Mac and Windows apps run the same Python: one shared launcher
  (`desktop/shared/launcher`) starts the Streamlit server, reports status
  and cleans up; one set of pages (`desktop/shared/pages`) is what both
  windows show. A bug fixed once is fixed on both.
- The Windows shell is a thin C# program (`FoodOptimizer.exe`): a WebView2
  window plus the code that starts the launcher and shows its status. It
  holds no product logic, so it stays small enough to review.
- Python is bundled (CPython, pinned in `pins.json`), started with `-I -X utf8
  -B`: no user site, no environment surprises, no bytecode written into the
  install.

## Install: per-user by default, all-users for IT

- Default: per-user, no administrator, into the user's profile. Updates
  replace the same copy.
- IT: an all-users install into `C:\Program Files\Food Optimizer` with
  `/ALLUSERS`. Only administrators can write there, which is what makes a
  path rule safe.
- The all-users installer never deletes anything in a per-user location.
  At a user's next launch of the IT copy, that user's own per-user copy runs
  its own uninstaller; their projects and state stay.
- The app, not the installer, pins the projects folder to Quick access, at
  its first launch. The installer then needs nothing from a user's profile.
- A running check per scope: per-user copies are seen in their own session,
  the IT copy by every user.

## No PowerShell at run time

Install and run use only the installer, the C# shell and Python. Nothing
starts PowerShell or a script host at run time, so constrained-language
mode and script block rules do not touch the app. PowerShell appears only
in build and test scripts.

## Local only

- The server listens on 127.0.0.1, on a random port, under an unguessable
  base path, and refuses any request whose Host header is not its own.
- WebView2 is set to block every request except to that loopback address,
  to keep crash dumps local, and to skip the SmartScreen reputation check
  for its pages. Links the user clicks open in the default browser.
- The bundled Python ignores proxy settings. Level 2 audits a full session
  with the Windows Filtering Platform: our processes connect to loopback
  only.

## One Job Object per launch

Each launch puts the launcher in its own Windows Job Object, so closing the
window, a crash or a kill ends the whole Python tree (server and warm-up
child). The launcher starts with `Process.Start` and is assigned to its job
right after, not created suspended: python's own start-up takes tens of
milliseconds before it can start a child, and a failed assign kills the
tree. `CREATE_SUSPENDED` would add a hand-written `CreateProcess` with its
own stdin pipe for no measurable gain.

## Deterministic build

The same tag builds the same files: pinned Python and runtime DLLs, locked
dependencies and a sorted file list. The build runs twice and compares. The
per-file hash list (`FoodOptimizer-<ver>-Windows.files.txt`) ships with each release, so
IT can verify an install or write hash rules.

## What changed in review

- The all-users installer no longer deletes per-user copies or their
  uninstall keys (see Install).
- The shell, not the installer, pins Quick access (see Install).
- no SBOM: it was dropped from the release; the per-file list stays.
- The running check is per scope (see Install).
- Allow-listing advice is scoped to the product: publisher + product,
  path and hash rules, never the vendors' signers machine-wide.
- `FOODOPT_TEST_ARCH` works only with `FOODOPT_TEST_HOOKS=1`.
