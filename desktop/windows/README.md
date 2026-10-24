# Food Optimizer for Windows

Status: Windows build in progress; environment only; unsigned, internal.

The design is `docs/superpowers/specs/2026-10-23-windows-port-design.md`.
The shell, the installer and signing come in later phases; this folder
builds the bundled Python environment they will ship.

## Build the environment

Needs Git, uv and Windows PowerShell 5.1 (build time only; nothing in the
built folder runs PowerShell, uv or the network).

```
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\build_installer.ps1 -Check
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\build_installer.ps1 -TestBuild
powershell -NoProfile -ExecutionPolicy Bypass -File desktop\windows\build_installer.ps1 [-Version X.Y.Z]
```

- `-Check` runs the release checks only: a clean tree, HEAD at tag
  `v<desktop/VERSION>`, a valid `bundle_manifest.txt`. It builds nothing.
- `-TestBuild` skips the release checks and prints `TEST BUILD - not for release`.
- A release build (no switch) refuses unless the release checks pass.
- `-OutDir DIR` changes the output folder (default `desktop\windows\build`).
  The image folder `<OutDir>\FoodOptimizer\` must be at most 80 characters.
  Each build replaces it, so the build refuses a folder there that holds
  anything but an earlier build.
- Every build prints `UNSIGNED build - internal testing only.`

Exit codes: `0` built (or `-Check` passed); `2` refused before anything was
fetched or created; `1` the build failed (`build failed: <step>: <reason>`,
and no build record is written).

Output, in `<OutDir>`:

- `FoodOptimizer\`: the image. `python\` (python.org 3.13.7 x64 from NuGet,
  with the Windows lock installed, stripped and precompiled), `app\` (every
  `bundle_manifest.txt` entry), `shared\` (the launcher, the status page,
  `messages.json`) and `Start Here.txt`.
- `FoodOptimizer.build.txt`: the build record (commit, both lock hashes,
  pinned download hashes, image hash, `signed=no`).
- `FoodOptimizer.files.txt`: the sha256 of every file in the image.
- `FoodOptimizer.unsigned.txt`: the binaries not signed by their vendor,
  which the signing phase signs.

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

## Test it

The tests in `desktop\windows\tests` run in the Windows VM against the
built image. Make a test environment on top of the bundled Python (it
writes nothing into the image):

```
uv venv C:\work\p3b-venv --python desktop\windows\build\FoodOptimizer\python\python.exe --system-site-packages
uv pip install --python C:\work\p3b-venv\Scripts\python.exe -c desktop\requirements-windows.lock.txt "pytest>=9,<10" matplotlib
uv pip uninstall --python C:\work\p3b-venv\Scripts\python.exe numpy packaging pillow python-dateutil six colorama
set FOODOPT_SLOW_TESTS=1& set PYTHONDONTWRITEBYTECODE=1& C:\work\p3b-venv\Scripts\python.exe -m pytest desktop\windows\tests -q -p no:cacheprovider
```

The uninstall step keeps the image's own copies of those packages in use.
On the Mac the same folder's tests skip; `tests/test_windows_*.py` check the
build scripts and `build_env.py` there.
