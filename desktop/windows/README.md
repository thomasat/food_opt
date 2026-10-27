# Food Optimizer for Windows

## Status

Windows build: unsigned, internal testing only.

The design is `docs/superpowers/specs/2026-10-23-windows-port-design.md`.
Releasing: "Release (Windows half)" below, inside the combined ritual in
`desktop/README.md`.

## System requirements (copy-paste for emails / download page)

Needs 64-bit Windows 10 (22H2) or Windows 11 and 2 GB free. Works offline.

## Build

On the build VM (VM_BUILD), with these installed (build time only; nothing
in the installed app runs PowerShell, uv or the network):

- Git, uv, Windows PowerShell 5.1;
- the .NET SDK 10.0.402 (`dotnet publish` of the shell);
- Inno Setup 6.3 or later (`ISCC.exe`; found through `FOODOPT_ISCC`, then
  `PATH`, then `%LOCALAPPDATA%\Programs\Inno Setup 6`, then the two
  Program Files folders).

```
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\build_installer.ps1 -Check
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\build_installer.ps1 -TestBuild
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\build_installer.ps1 [-Version X.Y.Z]
```

- `-Check` runs the release checks only: a clean tree, HEAD at tag
  `v<desktop/VERSION>`, a valid `bundle_manifest.txt`. It builds nothing.
- `-TestBuild` skips the release checks, prints `TEST BUILD - not for
  release`, and names the installer `...-Windows-test.exe`.
- A release build (no switch) refuses unless the release checks pass, and
  stops if `size_gate.json` is not set.
- `-OutDir DIR` changes the image folder (default `desktop\windows\build`);
  the installer goes to `DIR`'s sibling `dist\`. `<OutDir>\FoodOptimizer\`
  must be at most 80 characters, and the build refuses a folder there that
  holds anything but an earlier build.
- Every build prints `UNSIGNED build - internal testing only.`

The steps: the bundled Python and the Windows lock, Microsoft's C++
runtime, the app, the shared launcher and Start Here (phase 3b), then
`dotnet publish` of the shell (`FoodOptimizer.exe`, copied to the image
root), the image gate (the shell's files listed apart from
`image_files.txt`), the unsigned list, the per-file manifest, the image
record, `ISCC setup.iss`, the size gate and the release record. A VM build
takes about 6 minutes (ISCC about 2).

`python desktop/windows/build_env.py listing --write` rewrites `image_files.txt`
from the repo; review the diff. Without `--write` it prints the diff and exits 1 when stale.

Output:

- `build\FoodOptimizer\`: the image the installer copies.
- `build\FoodOptimizer.build.txt`: the image record (commit, both lock
  hashes, pinned download hashes, image hash, .NET SDK, shell file count,
  `signed=no`).
- `build\FoodOptimizer.files.txt`: the sha256 of every file in the image.
- `build\FoodOptimizer.unsigned.txt`: the binaries not signed by their
  vendor (phase 6 signs them; it includes `FoodOptimizer.exe`, the shell's
  own DLLs, and the two WebView2 wrappers that ReadyToRun recompiles).
- `dist\FoodOptimizer-<ver>-Windows[-test].exe`: the installer.
- `dist\FoodOptimizer-<ver>-Windows[-test].files.txt`: a byte copy of the
  per-file list, published as a release asset.
- `dist\FoodOptimizer-<ver>-Windows[-test].build.txt`: the release record:
  the image record, then `inno_setup`, `installer`, `installer_bytes`,
  `installer_sha256`, `disk_free_gb`, `size_gate`, `files_list_sha256`
  and `signed=no`.

**Size gate.** `size_gate.json` holds the ceilings for the installer and
the image (the first measurement plus 10%, rounded up to 10 MiB) and the
disk figure the installer enforces and the copy states:
`disk_free_gb = ceil((image bytes + 512 MiB) / GiB)`. A build over a
ceiling stops with `build failed: size gate: ...`. Raise a ceiling only on
purpose (a new dependency), with the new measurement and commit in
`measured`, and change the disk figure in Start Here and here with it.

Exit codes: `0` built (or `-Check` passed); `2` refused before anything was
fetched or created; `1` the build failed (`build failed: <step>: <reason>`,
and no record is written).

Downloads are pinned in `pins.json` and cached in `desktop\windows\.cache\`;
a cached file is re-hashed on every run and deleted if its hash is wrong.
Change a pin only by editing `pins.json` after checking, in the VM, that the
new file's `Get-AuthenticodeSignature` is `Valid` for the pinned signer and
that its version is what the pin says.

Microsoft's C++ runtime (`msvcp140*.dll`, `vcruntime140*.dll`) comes from
Microsoft's own redistributable (`VC_redist.x64.exe`, pinned). WiX 3.14's
`dark.exe -x` unpacks it (the redistributable's own `/layout` no longer
writes its MSIs), and `msiexec /a` extracts the seven DLLs without
installing anything. Each must be `Valid` for Microsoft's runtime signer
(`dll_signer` in `pins.json`), at the pinned version, and no older than a
copy inside the wheels. `dark.exe` is not signed, so the WiX zip's sha256 is
its only pin: when changing it, hash a second download from another machine.

## Install switches and exit codes

Per-user (no admin; to `%LOCALAPPDATA%\Programs\Food Optimizer`):

```
"FoodOptimizer-<ver>-Windows.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS
```

All users (IT, Intune; run as SYSTEM; to `C:\Program Files\Food Optimizer`):

```
"FoodOptimizer-<ver>-Windows.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS /ALLUSERS [/NODESKTOPSHORTCUT]
```

Uninstall:

```
"<install folder>\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
```

- Detection: the key
  `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{DE876ED9-927A-45AD-AD9B-D6E4B4C9A4A3}_is1`
  (all users; `HKCU\...` for a per-user copy), value `DisplayVersion`.
- `/LOG=<file>` writes the installer's log; every refusal is logged as
  `refused: <reason>` and the pre-warm as `prewarm rc=<n> secs=<n>`.
- An all-users install with `/DIR` outside Program Files makes that
  folder Administrators' and gives it Program Files' permissions before
  copying: users can read and run it, not change it (`acl: locked
  <folder>` in the log). The folder must be empty or this copy's own.
- A running per-user copy is seen only in its own user's session, while
  an IT upgrade waits for every user's running all-users copy.
- An all-users install never changes a per-user copy. When that user
  next opens Food Optimizer, it runs their copy's own uninstaller; projects
  are kept. A per-user install over an all-users copy is refused.
- Uninstalling a per-user copy deletes the app and the user's
  `%LOCALAPPDATA%\FoodOptimizer` and keeps the projects in
  `%USERPROFILE%\FoodOptimizer`; it keeps that state folder while an
  all-users copy is installed (both use it). An all-users uninstall
  deletes the app only: every user's state folder and projects stay.
- No reboot. Upgrade by running the new installer with the same command;
  it replaces the bundled Python and app folders whole.

| Code | Meaning | Intune type |
|---|---|---|
| `0` | Installed | Success |
| `1` | Food Optimizer is running, so Setup stopped before changing anything (also Inno's "Setup failed to initialize") | Retry |
| `2` | Cancelled before installing | Failed |
| `3` | A fatal error while getting ready to install; nothing changed | Failed |
| `4` | A fatal error while installing; run the same command again, it repairs | Failed |
| `5` | Stopped while installing; run the same command again, it repairs | Failed |
| `6` | Setup was ended from outside (for example by Task Manager); run it again | Failed |
| `7` | Refused: SYSTEM without `/ALLUSERS`; installed by IT (all-users copy present); install folder over 80 characters; install folder holds other files; too little free space | Failed |
| `8` | Needs a restart before it can install (not used by Food Optimizer) | Failed |
| `9` | Installed, but the pre-warm import failed (security software blocked a file); the app will not open until IT allows it | Failed |

Code 1 under `/SUPPRESSMSGBOXES` means the app is open: Setup does not
wait, it exits at once. Map `1` to Retry in Intune's return codes (as for
Intune's own `1618`): the install changed nothing, and succeeds once the
user closes Food Optimizer.

Uninstall exit codes (`unins000.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART`):

| Code | Meaning | Intune type |
|---|---|---|
| `0` | Uninstalled | Success |
| `1` | Food Optimizer is running (or the uninstaller could not start); nothing changed | Retry |

## For IT teams (reference)

A repo reference, never sent: the release page has five lines for IT.
The switches, exit codes and detection key are above.

### Requirements

- Windows 10 22H2 (build 19045) or Windows 11, 64-bit. Arm PCs run it
  under x64 emulation.
- Microsoft Edge WebView2 Runtime (Evergreen), which Windows 10 and 11
  receive through Windows Update. Without it the app shows "Food Optimizer
  can't open" with "Ask IT to install Microsoft Edge WebView2 Runtime."
- No internet needed, at install or at run time.

### What runs

```
FoodOptimizer.exe                        the window (.NET, WebView2)
  python.exe -I -X utf8 -B -m launcher   the bundled Python, from <install folder>\python
    python.exe                           the local server (Streamlit)
    python.exe                           a warm-up child
  msedgewebview2.exe (several)           from the Microsoft Edge WebView2 Runtime folder
```

- `FOODOPT_TEST_HOOKS=1` turns on test-only switches such as
  `FOODOPT_TEST_ARCH`; without it they are ignored. Never set it outside
  tests.
- Every launch runs in its own Windows Job Object; closing the window ends
  the Python tree.
- The server listens on 127.0.0.1 only, on a random port, under an
  unguessable base path, and refuses any request whose Host header is not
  its own. No inbound firewall rule is needed.
- At install, the pre-warm imports the main libraries once, so on-access
  scanning happens then, not at the first open (exit code 9 if blocked).

### Network

- FoodOptimizer.exe and the bundled python.exe connect only to loopback:
  Level 2's `test_our_processes_connect_only_to_loopback` audits a full
  session with the Windows Filtering Platform.
- The WebView2 Runtime's own processes may contact Microsoft for updates
  and diagnostic data under the PC's policies. Food Optimizer turns on
  custom crash reporting (no crash dumps go to Microsoft), turns off the
  SmartScreen reputation check for its pages, and blocks every page
  request except to its own loopback address.
- The bundled Python ignores proxy settings. Links the user clicks open in
  the default browser.

### Data at rest

- State, per user: `%LOCALAPPDATA%\FoodOptimizer\`, holding `server.port`,
  `server.base`, `status.txt`, `status.txt.tmp`, `launch.lock`,
  `launcher.log`, `launcher.log.1`, `settings.json`, `shell.log`,
  `shell.log.1` and the `WebView2` folder (the page's browser profile).
- Projects, per user: `%USERPROFILE%\FoodOptimizer\`, one `<name>.pkl`
  file of JSON each. At each launch, when a project changed, a dated copy
  goes into a subfolder there; the newest 10 launch copies are kept.
- Project files are read as JSON only and never unpickled: a file that is
  not JSON is reported as damaged.
- Moving them: `FOODOPT_STATE_DIR` and `FOODOPT_PROJECTS_DIR` replace the
  two folders above (the tests use them; IT may set them in the user's
  environment before the first launch). Uninstall deletes a moved state
  folder only when it is named `FoodOptimizer`.
- Nothing is encrypted by the app (it relies on BitLocker). The logs list
  file paths, which include the Windows user name. `launcher.log` moves to
  `launcher.log.1` at each launch; `shell.log` moves to `shell.log.1` at
  1,000,000 bytes.

### Files, registry and services

- Shortcuts: two in Start (Food Optimizer, and Food Optimizer - Start
  Here) and one on the desktop (unless `/NODESKTOPSHORTCUT`).
- Food Optimizer pins the projects folder to Quick access the first time
  each user opens it. Uninstalling a per-user copy unpins it only if the
  app pinned it.
- The only registry entry is the uninstall key, which Inno Setup writes
  (HKCU per user, HKLM for all users). The app only reads its own HKCU
  uninstall key (to find this user's per-user copy) and writes none.
- The all-users installer never deletes anything in a user's profile: at
  a user's next launch of the IT copy, that user's own per-user copy runs
  its own uninstaller, silently.
- No services, scheduled tasks, drivers or shell extensions.

### Allow-listing

Scope the rules to the product, not to the signers. The files carry their
vendors' signatures; only those their vendors left unsigned
(`build\FoodOptimizer.unsigned.txt`) are signed by Food System Innovations.
What the files carry:

- Food System Innovations: the installer, the uninstaller,
  `FoodOptimizer.exe` and the files it signs;
- Python Software Foundation: `python.exe`, `pythonw.exe`, `python313.dll`
  and CPython's own modules;
- Microsoft Windows Software Compatibility Publisher: the C++ runtime DLLs
  beside `python.exe`;
- Microsoft Corporation: the .NET runtime files.

The rules:

- A publisher + product rule (AppLocker, WDAC or Defender for Endpoint) for
  Food System Innovations' files, Food Optimizer as the product.
- A path rule for `%PROGRAMFILES%\Food Optimizer\*` (the all-users
  install, where only administrators can write).
- Hash rules for the vendor-signed files: check the installed files against
  the release's `FoodOptimizer-<ver>-Windows.files.txt` (flat SHA-256 values,
  not the PE hashes AppLocker and WDAC use), then generate the rules from
  those files with `Get-AppLockerFileInformation -Directory "<install folder>"
  -Recurse | New-AppLockerPolicy -RuleType Hash ...` or
  `New-CIPolicy -Level Hash`.
- Third-party DLLs and `.pyd` files that Food System Innovations re-signed
  keep their own product name, so under WDAC they need the path rule or
  hash rules; a publisher + product rule covers only Food Optimizer's own
  files.
- Do not allow the Python Software Foundation or Microsoft signers
  machine-wide: the path or hash rule already covers their files here.
- AppLocker's default rules allow non-admins only `%PROGRAMFILES%` and
  `%WINDIR%`, so they block a per-user install (it runs from the user's
  profile). Use the
  all-users install on such PCs.

## Test

**Level 1** runs on the Mac (`python -m pytest tests/test_windows_*.py`)
and in the VM (`desktop\windows\tests`, with a test environment on top of
the bundled Python that writes nothing into the image):

```
uv venv C:\work\venv --python desktop\windows\build\FoodOptimizer\python\python.exe --system-site-packages
uv pip install --python C:\work\venv\Scripts\python.exe -c desktop\requirements-windows.lock.txt "pytest>=9,<10" matplotlib
uv pip uninstall --python C:\work\venv\Scripts\python.exe numpy packaging pillow python-dateutil six tzdata colorama
set FOODOPT_SLOW_TESTS=1& set PYTHONDONTWRITEBYTECODE=1& C:\work\venv\Scripts\python.exe -m pytest desktop\windows\tests -k "not l2" -q -p no:cacheprovider
```

These packages are already in the image's Python; removing the venv's
copies makes the tests import the image's own.

The VM's quirks:

- Clone with `git -c core.autocrlf=false clone ...` and then
  `git config core.autocrlf false`, so files stay LF.
- Long or UI work runs as a scheduled task:
  `schtasks /Create /TN <name> /TR "<cmd>" /SC ONCE /SD 01/01/2020 /ST 00:00 /RU <user> /IT /F`
  never fires on its own schedule; start it with `schtasks /Run` and delete
  it afterwards with `schtasks /Delete /F /TN <name>`.
- UI tests need the owner signed in at the console (`quser` shows
  `console ... Active`) and nobody touching the VM window.
- Turn sleep off on the VM, or a long run stops half way.

**Level 2** installs the built installer for real, as the limited user,
as SYSTEM and as admin, on VM_BUILD with the owner signed in at the
console. Build the two test installers, then run everything:

```
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\tests\make_test_installer.ps1 -Variant old
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\tests\make_test_installer.ps1 -Variant broken
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\tests\run_level2.ps1 -Installer desktop\windows\dist\FoodOptimizer-<ver>-Windows-test.exe -OldInstaller C:\work\l2\installers\FoodOptimizer-0.0.1-Windows-old.exe -BrokenInstaller C:\work\l2\installers\FoodOptimizer-<ver>-Windows-broken.exe -Cleanup
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\FoodOptimizer\uitests\run_ui_tests.ps1 -Stage C:\work\stage -Installer desktop\windows\dist\FoodOptimizer-<ver>-Windows-test.exe -Filter "--filter-trait Category=Installer --filter-trait Category=Installed"
```

`run_level2.ps1` ends with `EXIT=0` and prints `C:\work\l2\report.json`
(install, pre-warm, first-open and warm-relaunch seconds; the zero-traffic
event counts). Its installs use user folders under `C:\FOL2\` (a user
named `José Müller`) and never your own projects. `-Cleanup` lets it
uninstall a Food Optimizer it did not install.

## Testing an unsigned build (internal only)

An unsigned build is never sent to a recipient; signing is phase 6.
Opening one shows SmartScreen's "Windows protected your PC" with
"Unknown publisher": click More info, then Run anyway.

## Manual checklist (every release)

Run in a VM-local clone at the tag, from a fresh-Windows snapshot. Each
item is "Do → Expect"; `[S]` marks the patch smoke list.

- **W1** [S] Download the installer through Edge (Mark of the Web) → write
  down Edge's download warning and the steps to Keep it, and SmartScreen's
  exact words and the publisher it names.
- **W2** [S] Install per-user by double-click → three pages (Welcome,
  Installing, Finish), no admin prompt, "Add a desktop shortcut" ticked,
  Finish offers "Open Food Optimizer" and "Open Start Here". Untick Open
  Food Optimizer.
- **W3** Open it offline from the start: block the internet first, then
  type Food Optimizer in Start and open it → "Setting up Food Optimizer",
  then the app. Time it.
- **W4** [S] The taskbar → Food Optimizer's icon and name, not Edge or
  Python; Task Manager groups the processes under Food Optimizer.
- **W5** Display scaling 150% → the window fits and stays sharp.
- **W6** [S] Menus File, View and Help; Ctrl+= / Ctrl+- / Ctrl+0 and
  Ctrl+mouse wheel; Start Here, Get Help (the address), Show Log File, About
  (the version) → each does what it says.
- **W7** Right-click → only Cut, Copy, Paste and Select all in a text box;
  nothing on the page.
- **W8** [S] Upload via Quick access → FoodOptimizer → "Round 6 results
  (pilot).xlsx" → the file loads.
- **W9** [S] Download a round sheet → the Save dialog opens in Downloads,
  File Explorer shows the file, and a sound plays. `Get-Item <file> -Stream *`
  lists no `Zone.Identifier`. Excel opens it editable; with no Excel on
  the machine, write "no Excel".
- **W10** [S] The demo walkthrough (`shared/manual_walkthrough.md`) → as
  written.
- **W11** [S] Close the window → within 5 s Task Manager shows no
  Food Optimizer and no python.exe from the install folder.
- **W12** Sleep (Start → Power → Sleep, or Fusion Suspend for 5 minutes)
  → the app responds, or "Food Optimizer stopped unexpectedly", whose Try
  again loads it.
- **W13** A Windows user named "José Müller": install, open, open the
  yogurt pilot, close → all work.
- **W14** Upgrade over the older test installer → projects kept; About
  shows the new version.
- **W15** [S] Uninstall from Settings → Apps → our confirm words; the last
  words name the FoodOptimizer folder; projects stay; the Quick access and
  Start entries are gone.
- **W16** Zero traffic during W10 → the agent's WFP capture shows no
  non-loopback connection from our processes.
- **W17** Read Start Here and every installer message as a skeptical
  nontechnical reader → accurate, plain, and every error has a next step.

Results go in `docs/superpowers/notes/windows-checklist-<date>-<machine>.md`
(local only; screenshots beside it, never committed):

| Item | Pass/Fail | Exact words seen | Note |
|---|---|---|---|
| W1 | | | |

## Locked-down profile

On a Windows 11 Enterprise evaluation VM: a standard user (`fostd`), no
network, Defender real-time on with the rare-file rule
(`01443614-cd74-433a-b99e-2ecdc07bfc25`) enabled, AppLocker's default
rules enforced for Exe, Msi, Script and Appx.
Smart App Control is off on the evaluation image (it blocks unsigned
builds outright; on a signed build, note its state in L1). AppLocker's
default rules allow only `%PROGRAMFILES%` and `%WINDIR%` for non-admins,
which is why L2 blocks the per-user installer and L3 uses the IT install.

- **L1** The machine state: the effective AppLocker policy, the ASR rule
  and the firewall.
- **L2** As `fostd`, double-click the installer in Downloads → AppLocker
  blocks it. Record the exact words.
- **L3** The IT install: SYSTEM runs the all-users command → record the
  exit code and the time.
- **L4** As `fostd`: open from Start, open the yogurt pilot, download a
  round sheet, close → record any prompt or block.
- **L5** Defender events 1121/1122 and AppLocker events 8003/8004 since L3
  → list each blocked file. On an unsigned build, rare-file blocks are
  expected; they feed phase 6.
- **L6** SYSTEM uninstall → exit 0; `fostd`'s projects and state folder
  stay.

VM setup: download the Windows 11 Enterprise evaluation ISO for ARM64,
make a 4-core, 8 GB, 80 GB Fusion VM with a local admin, add the SSH key
and the `FoodOpt-SSH-In` firewall rule, take snapshot "Locked fresh", then
the agent configures the profile over SSH. Not yet run: results go here
("Seen on <date>, unsigned build: ...").

## Release (Windows half)

Steps 6 and 7 of the combined ritual ("Publishing a release" in
`desktop/README.md`), after the tag `vX.Y.Z` is pushed. On VM_BUILD:

1. A VM-local clone at the tag (never a shared folder):

   ```
   git clone --branch vX.Y.Z https://github.com/thomasat/food_opt "C:\work\release X.Y.Z"
   ```

2. The release build (no switch: it refuses unless the tree is clean and
   HEAD is the tag) from that clone, then signing:

   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\build_installer.ps1
   ```

   PENDING(signing step)
3. pytest under the image's Python at the tag, as in "Test" above: 0
   failed, apart from a list stated in the release PR.
4. Level 2 with the three installers and `-Cleanup` (`run_level2.ps1`, as
   in "Test"), then `run_ui_tests.ps1` with both filters (Installer and
   Installed, then the default set). Its
   `test_our_processes_connect_only_to_loopback` backs the release page's
   "sends nothing".
5. The manual checklist W1 to W17 and the locked-down profile L1 to L6.
   A patch release that changes only shared Python may run the `[S]` items
   only.
6. Hand back to the Mac at step 7 of the ritual: it copies
   `dist\FoodOptimizer-X.Y.Z-Windows.exe` and its `.build.txt` and `.files.txt`.
