# Windows port (planned — not built yet)

Mirror of the macOS approach. The design, launcher logic, UX flows, copy,
and test structure all transfer; only the window wrapper is rewritten
(Swift/AppKit does not exist on Windows). Estimated effort: under a week,
mostly mechanical — the thinking is done on the macOS side.

1. **Window wrapper:** a small C# app hosting **WebView2** (preinstalled
   on Windows 10/11) — the direct equivalent of
   `../FoodOptimizerApp.swift`, which doubles as its spec (~250 lines):
   instant window with setup status page + spinner, poll `server.port`
   then health endpoint, load the app, native menus/shortcuts, save
   dialog + reveal-in-Explorer for downloads, file-picker uploads,
   friendly pages for exit code 2 (unsupported) / 3 (needs internet) /
   4 (not enough disk space) / server death, quit terminates the launcher and awaits
   its cleanup. Cannot be Python: it must run before the venv exists.
2. **Launcher:** `launcher.ps1` — line-by-line port of `../launcher.sh`
   (same exit codes, port file, marker/lock/backup logic, idle watchdog).
   Paths: environment under `%LOCALAPPDATA%\FoodOptimizer`, projects in
   `%USERPROFILE%\FoodOptimizer` (home-folder root for parity with macOS,
   where ~/Documents is privacy-gated; Windows has no TCC, so those
   permission traps do not exist there).
3. **uv:** bundle `uv-x86_64-pc-windows-msvc.zip`'s `uv.exe`, pinned +
   SHA256-verified like `../fetch_uv.sh`.
4. **Lock:** Windows needs its own compiled lock
   (`requirements-windows.lock.txt`), generated with
   `uv pip compile --generate-hashes --python-platform x86_64-pc-windows-msvc`.
5. **Installer:** Inno Setup script (`setup.iss`) producing
   `FoodOptimizer-Setup.exe`: installs the app folder + Start Menu
   shortcut launching the wrapper exe. Icon from `../logo/` (convert to
   .ico).
6. **No-warnings requirement (budget this):** unsigned installers
   trigger Microsoft SmartScreen. An Authenticode code-signing
   certificate — or Azure Trusted Signing (~$10/mo) — is the Windows
   equivalent of the Apple Developer ID. Required before distributing
   to recipients.
7. **Build both installers with one action:** a GitHub Actions workflow
   with two jobs — `macos-14` (Apple Silicon) builds + tests the dmg,
   `windows-latest` builds + tests the exe — both attached to the same
   GitHub release on every version tag. Neither OS can build the other's
   installer, so CI is the one-command path to shipping both.
8. **Test suite:** port `../test_e2e.sh` to PowerShell with the same
   structure (static checks, build checks, throwaway-profile end-to-end
   against the launcher).
