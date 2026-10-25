# Food Optimizer for Windows

## Status

Windows build: unsigned, internal testing only.

The design is `docs/superpowers/specs/2026-10-23-windows-port-design.md`.
Releasing: "Release (Windows half)" below, inside the combined ritual in
`desktop/README.md`.

## System requirements (copy-paste for emails / download page)

Needs Windows 10 or 11 (64-bit) and 2 GB free. Works offline.

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
record, `ISCC setup.iss`, the size gate, the SBOM and the release record. A VM build
takes about 6 minutes (ISCC about 2).

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
- `dist\FoodOptimizer-<ver>-Windows[-test].cdx.json`: the software bill of
  materials (CycloneDX 1.6), written by `sbom.py` under the image's Python
  from the image's own package metadata, the shell's `deps.json` with its
  NuGet `.nuspec` files, and `sbom_declared.json`. It fails the build on a
  package whose licence it cannot read, on a lock mismatch, and on an
  unsigned file no component owns.
- `dist\FoodOptimizer-<ver>-Windows[-test].build.txt`: the release record:
  the image record, then `inno_setup`, `installer`, `installer_bytes`,
  `installer_sha256`, `disk_free_gb`, `size_gate`, `files_list_sha256`,
  `sbom_sha256`, `sbom_components`, `sbom_gpl_licensed`,
  `sbom_gpl_mentions` and `signed=no`.

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
- A per-user and an all-users copy never coexist. An all-users install
  deletes every per-user copy in the default folder, its Start menu
  entries, and (for users signed in at the time) its Apps entry and
  desktop shortcut. A per-user install over an all-users copy is refused.
- Uninstalling deletes the app and the user's `%LOCALAPPDATA%\FoodOptimizer`
  and keeps the projects in `%USERPROFILE%\FoodOptimizer`. A SYSTEM
  uninstall cannot reach users' `%LOCALAPPDATA%`, so their state folders
  stay.
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

## Test

**Level 1** runs on the Mac (`python -m pytest tests/test_windows_*.py`)
and in the VM (`desktop\windows\tests`, with a test environment on top of
the bundled Python that writes nothing into the image):

```
uv venv C:\work\p5-venv --python desktop\windows\build\FoodOptimizer\python\python.exe --system-site-packages
uv pip install --python C:\work\p5-venv\Scripts\python.exe -c desktop\requirements-windows.lock.txt "pytest>=9,<10" matplotlib
uv pip uninstall --python C:\work\p5-venv\Scripts\python.exe numpy packaging pillow python-dateutil six tzdata colorama
set FOODOPT_SLOW_TESTS=1& set PYTHONDONTWRITEBYTECODE=1& C:\work\p5-venv\Scripts\python.exe -m pytest desktop\windows\tests -k "not l2" -q -p no:cacheprovider
```

The uninstall step keeps the image's own copies of those packages in use.

**Level 2** installs the built installer for real, as the limited user,
as SYSTEM and as admin, on VM_BUILD with the owner signed in at the
console. Build the two test installers, then run everything:

```
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\tests\make_test_installer.ps1 -Variant old
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\tests\make_test_installer.ps1 -Variant broken
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\tests\run_level2.ps1 -Installer desktop\windows\dist\FoodOptimizer-<ver>-Windows-test.exe -OldInstaller C:\work\l2\installers\FoodOptimizer-0.0.1-Windows-old.exe -BrokenInstaller C:\work\l2\installers\FoodOptimizer-<ver>-Windows-broken.exe -Cleanup
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\FoodOptimizer\uitests\run_ui_tests.ps1 -Stage C:\work\p5-stage -Installer desktop\windows\dist\FoodOptimizer-<ver>-Windows-test.exe -Filter "--filter-trait Category=Installer --filter-trait Category=Installed"
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

- **L1** The machine state: the effective AppLocker policy, the ASR rule
  and the firewall.
- **L2** As `fostd`, double-click the installer in Downloads → AppLocker
  blocks it. Record the exact words ("Blocked? Forward this email to IT.").
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
   Installed, then the default set).
5. The manual checklist W1 to W17 and the locked-down profile L1 to L6.
   A patch release that changes only shared Python may run the `[S]` items
   only.
6. Hand back to the Mac at step 7 of the ritual: it copies
   `dist\FoodOptimizer-X.Y.Z-Windows.exe`, its `.build.txt`, `.files.txt`
   and `.cdx.json`, and `C:\work\l2\report.json`.
