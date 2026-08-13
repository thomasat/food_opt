# Windows port (planned — not built yet)

Mirror of the macOS approach, so this stays a mechanical add:

1. **Launcher:** `launcher.ps1` (PowerShell) with identical logic to
   `../launcher.sh`: preflight → first-run `uv` setup → localhost-only
   Streamlit → idle shutdown. Paths: environment under
   `%LOCALAPPDATA%\FoodOptimizer`, projects in
   `%USERPROFILE%\Documents\FoodOptimizer`.
2. **uv:** bundle `uv-x86_64-pc-windows-msvc.zip`'s `uv.exe`, pinned +
   SHA256-verified like `../fetch_uv.sh`.
3. **Lock:** Windows needs its own compiled lock
   (`requirements-windows.lock.txt`), generated on Windows or with
   `uv pip compile --python-platform x86_64-pc-windows-msvc`.
4. **Installer:** Inno Setup script (`setup.iss`) producing
   `FoodOptimizer-Setup.exe`: installs the app folder + Start Menu
   shortcut that runs the launcher via `powershell -WindowStyle Hidden`.
5. **No-warnings requirement (budget this):** unsigned installers
   trigger Microsoft SmartScreen. An Authenticode code-signing
   certificate — or Azure Trusted Signing (~$10/mo) — is the Windows
   equivalent of the Apple Developer ID. Required before distributing
   to recipients.
6. **Build:** any Windows 10/11 x64 machine, or a `windows-latest`
   GitHub Actions runner (matrix-build alongside the macOS job).
