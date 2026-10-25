# Food Optimizer for Windows: information for IT teams

This file is the source. Each release's PDF, `FoodOptimizer-<version>-IT-sheet.pdf`, fills in the values shown in braces. A value in braces or a "PENDING" note here means the release has not filled it yet; a release PDF is never made while one is left. Every statement below is checked by the test named in the last section.

## Release values

| Item | Value |
|---|---|
| Version | {{version}} |
| Installer | `{{exe_name}}` ({{exe_url}}) |
| Size | {{exe_mb}} MB |
| SHA-256 | `{{exe_sha256}}` |
| Source commit | `{{commit}}` |
| Signed | {{signed}} |
| Software bill of materials | `{{sbom_name}}` (CycloneDX 1.6, a release asset) |
| Per-file hash list | `{{files_list_name}}` (a release asset) |

## Summary

Food Optimizer is a desktop app for food scientists. It plans and records formulation studies on the user's own PC. It is one installer (Inno Setup), installs per user without admin rights by default, and installs for all users with `/ALLUSERS` (Intune, run as SYSTEM). It bundles its own Python and libraries in its install folder, runs only on the PC, and needs no internet at install or at run time. It writes no services, scheduled tasks or drivers.

## Publisher and contact

- Publisher: Food System Innovations PENDING(publisher validation by Microsoft).
- Legal address: PENDING(legal address).
- Website: PENDING(website).
- Support and security contact: Sohum Patnaik, sohum.patnaik@fsi.org.
- Report security issues to the same address.
- The source code is public: https://github.com/thomasat/food_opt
- The app is licensed under the MIT licence (`LICENSE` in the repository).

## System requirements

- Windows 10 22H2 (build 19045) or Windows 11, 64-bit. Arm PCs run it under x64 emulation.
- Microsoft Edge WebView2 Runtime (Evergreen), which Windows 10 and 11 receive through Windows Update. Without it the app shows "Food Optimizer can't open" with "Ask IT to install Microsoft Edge WebView2 Runtime."
- {{disk_free_gb}} GB free, enforced by the installer (it refuses with exit code 7 when there is less).
- No internet connection, at install or at run time.
- Tested on: Windows 11 on Arm (VM, x64 emulation). PENDING(Windows 10 22H2 and Windows 11 x64 results)

## Install and uninstall

Per user (the default; no admin rights; to `%LOCALAPPDATA%\Programs\Food Optimizer`):

```
"FoodOptimizer-<ver>-Windows.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS
```

All users (IT, Intune; run as SYSTEM or an administrator; to `C:\Program Files\Food Optimizer`):

```
"FoodOptimizer-<ver>-Windows.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS /ALLUSERS [/NODESKTOPSHORTCUT]
```

Uninstall (from the install folder):

```
"<install folder>\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
```

- `/LOG=<file>` writes the installer's log. Every refusal is logged as `refused: <reason>`, and the pre-warm as `prewarm rc=<n> secs=<n>`.
- `/NODESKTOPSHORTCUT` skips the desktop shortcut.
- A per-user and an all-users copy never coexist. An all-users install deletes every per-user copy in the default folder, with its Start menu entries, and (for users signed in at the time) its Apps entry and desktop shortcut. A per-user install over an all-users copy is refused (exit code 7).
- Running as SYSTEM without `/ALLUSERS` is refused (exit code 7), as is an install folder longer than 80 characters or one that holds other files.
- Uninstalling deletes the app, its shortcuts and the user's state folder `%LOCALAPPDATA%\FoodOptimizer`, and keeps the user's projects in `%USERPROFILE%\FoodOptimizer`.
- A SYSTEM uninstall cannot reach users' `%LOCALAPPDATA%`, so each user's `%LOCALAPPDATA%\FoodOptimizer` stays.

## Intune

Package the installer as a Win32 app (`.intunewin`), with these settings.

- Install command (Install behavior: System):

  ```
  "FoodOptimizer-<ver>-Windows.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS /ALLUSERS
  ```

- Uninstall command:

  ```
  "C:\Program Files\Food Optimizer\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
  ```

- Detection rule: Registry.
  - Key path:

    ```
    HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{DE876ED9-927A-45AD-AD9B-D6E4B4C9A4A3}_is1
    ```

  - Value name: `DisplayVersion`. Detection method: Version comparison, greater than or equal to `{{version}}`.
  - Associated with a 32-bit app on 64-bit clients: No.
- Requirements: Windows 10 22H2 or later; architecture x64 (Arm64 devices run it under x64 emulation); {{disk_free_gb}} GB free.
- Device restart behavior: no specific action. The installer never needs a reboot.
- Return codes: `0` Success; `1` Retry (Food Optimizer is open, nothing changed); `2` to `9` Failed. See "Exit codes".
- Upgrade: deploy the new version with the same install command. It replaces the bundled Python and app folders whole and keeps every user's projects. For supersedence, supersede the old app without "Uninstall previous version".
- PENDING(all-users upgrade test)

## Exit codes

Installer (`FoodOptimizer-<ver>-Windows.exe`):

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

Code 1 under `/SUPPRESSMSGBOXES` means the app is open: Setup does not wait, it exits at once. Map `1` to Retry, as for Intune's own `1618`: the install changed nothing, and succeeds once the user closes Food Optimizer.

Uninstaller (`unins000.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART`):

| Code | Meaning | Intune type |
|---|---|---|
| `0` | Uninstalled | Success |
| `1` | Food Optimizer is running (or the uninstaller could not start); nothing changed | Retry |

## Signing and hashes

- Certificate: PENDING(certificate subject and CA chain). The installer, the uninstaller and every binary in the install folder that its vendor did not sign are signed by Food System Innovations through Microsoft Artifact Signing.
- Identity EKU: PENDING(identity EKU OID).
- Artifact Signing certificates are short-lived and rotate about every 3 days, so allow-list by publisher or by the identity EKU, never by certificate thumbprint.
- Verify a file in PowerShell:

  ```
  Get-AuthenticodeSignature -FilePath ".\FoodOptimizer-<ver>-Windows.exe" | Format-List Status, SignerCertificate
  ```

- The installer's SHA-256 (`Get-FileHash`'s default): `{{exe_sha256}}`.

  ```
  Get-FileHash ".\FoodOptimizer-<ver>-Windows.exe"
  ```

- `{{files_list_name}}` lists one `<sha256>  <path>` line for every file the installer copies, path relative to the install folder (the `sha256sum` format, sorted by path). The install folder also holds `unins000.exe` and `unins000.dat`, which Inno Setup writes at install time.

## What runs

```
FoodOptimizer.exe                        the window (.NET, WebView2)
  python.exe -I -X utf8 -B -m launcher   the bundled Python, from <install folder>\python
    python.exe                           the local server (Streamlit)
    python.exe                           a warm-up child
  msedgewebview2.exe (several)           from the Microsoft Edge WebView2 Runtime folder
```

- Every launch runs in its own Windows Job Object. Closing the window ends the whole tree, and no Food Optimizer process outlives it.
- The server listens on 127.0.0.1 only, on a random port, under an unguessable base path, and refuses any request whose Host header is not its own. No inbound firewall rule is needed.
- Pre-warm at install: the installer copies {{image_files}} files ({{image_gb}} GB), then imports the main libraries once with the bundled Python, so on-access scanning of the DLLs happens during the install, not at the first open. If security software blocks a file then, the installer exits with code 9.
- The bundled PyTorch (oneDNN) generates machine code at run time, so forcing Arbitrary Code Guard on `python.exe` would stop the app. PENDING(Arbitrary Code Guard check)

## Network

- Food Optimizer's own processes (FoodOptimizer.exe and the bundled python.exe) make no internet connections.
- Measured: {{wfp_events}} connection events from {{wfp_processes}}, all loopback, by a Windows Filtering Platform audit during a full session.
- The Microsoft Edge WebView2 Runtime's own processes may contact Microsoft for updates and diagnostic data under the PC's policies. Food Optimizer turns on custom crash reporting, so no crash dumps go to Microsoft, turns off the SmartScreen reputation check for its pages, and blocks every page request except to its own loopback address.
- Links the user clicks open in their default browser.
- No proxy and no firewall rules are needed. The bundled Python ignores proxy settings.

```
 user's PC
 +--------------------------------------------------------------+
 |  FoodOptimizer.exe --- WebView2 --- http://127.0.0.1:<port>  |
 |        |                               |                     |
 |        +-- python.exe (launcher) --- python.exe (server)     |
 |                                         |                    |
 |        %USERPROFILE%\FoodOptimizer  (projects, read/write)   |
 +--------------------------------------------------------------+
   no connection leaves the PC from Food Optimizer's processes
   (msedgewebview2.exe: Microsoft's own updates and diagnostics only)
```

## Data at rest

- State, per user: `%LOCALAPPDATA%\FoodOptimizer\`, holding `server.port`, `server.base`, `status.txt`, `status.txt.tmp`, `launch.lock`, `launcher.log`, `launcher.log.1`, `settings.json`, `shell.log`, `shell.log.1` and the `WebView2` folder (the page's browser profile).
- Projects, per user: `%USERPROFILE%\FoodOptimizer\`. Each project is a `<name>.pkl` file that holds JSON. At each launch, when a project changed, a dated copy of the project files goes into a subfolder there; the newest 10 launch copies are kept.
- The app does not encrypt these files; it relies on the disk's encryption (BitLocker).
- The projects folder is not in OneDrive, so it is not backed up unless IT backs it up.
- Project files are read as JSON only and never unpickled: a file that is not JSON is reported as damaged.
- The logs contain file paths, which include the Windows user name. `launcher.log` moves to `launcher.log.1` at each launch; `shell.log` moves to `shell.log.1` at 1,000,000 bytes. Both are deleted at a per-user uninstall, with the rest of the state folder.

## Files, registry and services

- The install folder (see "Install and uninstall").
- Shortcuts: two in Start (Food Optimizer, and Food Optimizer - Start Here) and one on the desktop (unless `/NODESKTOPSHORTCUT`).
- Per-user installs pin the projects folder to File Explorer's Quick access; uninstalling unpins it.
- The uninstall key, which Inno Setup writes: in HKCU for a per-user install, in HKLM for all users. An all-users install also deletes each replaced per-user copy's uninstall key.
- No services, scheduled tasks, drivers, shell extensions or other registry writes.
- After uninstall: the projects folder `%USERPROFILE%\FoodOptimizer`, and after a SYSTEM uninstall, each user's `%LOCALAPPDATA%\FoodOptimizer`. Nothing else.

## Allow-listing

- AppLocker: a publisher rule for Food System Innovations covers the installer and every binary it installs. For all-users installs, a path rule `%PROGRAMFILES%\Food Optimizer\*` also works. AppLocker's default rules block running an installer from Downloads, so a user's own install needs the publisher rule.
- App Control for Business: a signer rule on the identity EKU of Food System Innovations' certificate.
- Microsoft Defender for Endpoint: a certificate indicator (Allow) for the publisher, or file indicators for the hashes in `{{files_list_name}}`.
- The attack surface reduction rule "Block executable files from running unless they meet a prevalence, age, or trusted list criterion" (`01443614-cd74-433a-b99e-2ecdc07bfc25`) may block a new release's files until their reputation builds; the certificate indicator above allows them.
- EDR: the expected process tree is in "What runs".
- PENDING(allow-listing rules tested on a signed build)
- PENDING(locked-down profile results)

## Software bill of materials and licences

- `{{sbom_name}}` (CycloneDX 1.6 JSON) lists {{sbom_components}} components, generated at build time from the bundled packages' own metadata and checked against the release's lock file.
- Licence families: {{licence_families}}.
- {{gpl_statement}}
- {{resigned_files}} upstream binaries that their vendors left unsigned are re-signed by Food System Innovations; each is counted on its component in the SBOM (`foodopt:signing`, `foodopt:resigned_files`).
- Bundled runtimes: Python {{python_version}} (PSF licence), the Microsoft C++ runtime {{vc_runtime_version}} (beside `python.exe`), and the installer's own Inno Setup {{inno_setup}}.

## Vulnerabilities and updates

- There is no auto-update. Each version is a new installer that IT deploys.
- Load the SBOM into your own scanner to track CVEs in Python, PyTorch, Streamlit, Tornado and the other components.
- Report a vulnerability to sohum.patnaik@fsi.org.
- PENDING(vulnerability check before each release (owner))

## How this sheet is tested

| Section | Statement | Test |
|---|---|---|
| Summary | A silent per-user install needs no admin rights and registers Food System Innovations as publisher | `desktop/windows/tests/test_l2_10_install.py::test_silent_per_user_install_needs_no_admin` |
| Summary | No service, scheduled task or registry write besides the uninstall key | `tests/test_windows_it_sheet.py::test_nothing_writes_registry_services_or_tasks` |
| System requirements | The build number, the disk figure and the WebView2 message match the installer and the shell | `tests/test_windows_it_sheet.py::test_requirements_match_the_installer` |
| System requirements | Too little free space is refused with exit code 7 | `desktop/windows/tests/test_l2_70_disk.py::test_too_little_free_space_is_refused` |
| Install and uninstall | The commands are the Windows README's, and the switches are the installer's | `tests/test_windows_it_sheet.py::test_switches_match_the_installer` |
| Install and uninstall | The installed files are the image, plus unins000.exe and unins000.dat | `desktop/windows/tests/test_l2_10_install.py::test_installed_files_are_the_image` |
| Install and uninstall | An all-users install as admin goes to Program Files | `desktop/windows/tests/test_l2_50_scopes.py::test_all_users_install_as_admin` |
| Install and uninstall | A per-user install over an all-users copy is refused | `desktop/windows/tests/test_l2_50_scopes.py::test_per_user_install_refuses_when_it_installed_it` |
| Install and uninstall | An all-users install replaces a per-user copy | `desktop/windows/tests/test_l2_50_scopes.py::test_all_users_install_replaces_a_per_user_copy` |
| Install and uninstall | An install folder over 80 characters is refused | `desktop/windows/tests/test_l2_50_scopes.py::test_a_long_install_folder_is_refused` |
| Install and uninstall | A silent uninstall deletes the app and keeps the projects | `desktop/windows/tests/test_l2_45_uninstall.py::test_silent_uninstall_keeps_projects` |
| Intune | Install and uninstall as SYSTEM with the commands above | `desktop/windows/tests/test_l2_50_scopes.py::test_intune_install_and_uninstall_as_system` |
| Intune | SYSTEM without /ALLUSERS is refused | `desktop/windows/tests/test_l2_50_scopes.py::test_system_without_allusers_is_refused` |
| Intune | An install while the app runs exits 1 and changes nothing | `desktop/windows/tests/test_l2_50_scopes.py::test_silent_install_while_the_app_runs` |
| Intune | An upgrade replaces the bundled folders and leaves nothing stale | `desktop/windows/tests/test_l2_60_upgrade.py::test_upgrade_after_a_lock_change_leaves_nothing_stale` |
| Exit codes | The tables match the installer's exit codes and Intune types | `tests/test_windows_installer.py::test_exit_code_table_matches_l2_support` |
| Exit codes | The tables here are the Windows README's, row for row | `tests/test_windows_it_sheet.py::test_exit_code_tables_match_the_readme` |
| Exit codes | A blocked pre-warm says so and exits 9 | `desktop/windows/tests/test_l2_60_upgrade.py::test_a_failed_prewarm_says_so_and_exits_9` |
| Signing and hashes | The per-file list is sorted, LF-only and stable | `tests/test_windows_build_env.py::test_file_manifest_is_sorted_lf_and_stable` |
| Signing and hashes | Every unsigned binary is attributed to exactly one component | `tests/test_windows_sbom.py::test_every_unsigned_file_is_attributed_once` |
| Signing and hashes | A release PDF is refused for an unsigned build or a file whose hash differs from its record | `tests/test_release_docs.py::test_release_render_refuses` |
| What runs | The launch command is the shell's | `tests/test_windows_it_sheet.py::test_what_runs_matches_the_launch` |
| What runs | No console window; closing the window leaves no python.exe | `desktop/windows/tests/test_l2_30_shell.py::test_shell_smoke_no_console_health_close_no_python` |
| What runs | Try again on the installed app leaves one server | `desktop/windows/FoodOptimizer/FoodOptimizer.UiTests/Installer/InstalledAppTests.cs::Try_again_on_the_installed_app_leaves_one_server` |
| What runs | The server answers only on its base path and its own Host | `tests/desktop_launcher/test_launcher_streamlit.py::test_real_streamlit_answers_only_on_its_base_path_and_host` |
| What runs | Any other Host header is refused | `tests/desktop_launcher/test_launcher_guard.py::test_host_check_refuses_any_other_host` |
| What runs | The pre-warm runs at install | `desktop/windows/tests/test_l2_10_install.py::test_prewarm_ran_at_install` |
| Network | The audit sees the bundled Python's connections | `desktop/windows/tests/test_l2_40_traffic.py::test_the_capture_sees_our_python` |
| Network | Food Optimizer's processes connect only to loopback | `desktop/windows/tests/test_l2_40_traffic.py::test_our_processes_connect_only_to_loopback` |
| Network | Proxies are turned off for the bundled Python | `tests/desktop_launcher/test_launcher_guard.py::test_guard_turns_proxies_off` |
| Network | The server's own address lookups are stubbed out | `tests/desktop_launcher/test_launcher_guard.py::test_guard_stubs_streamlit_ip_lookups` |
| Network | The page fetches nothing from the internet | `tests/test_windows_ready.py::test_ui_fetches_nothing_from_the_internet` |
| Network | Crash reporting, the reputation check and the request filter are pinned | `tests/test_windows_it_sheet.py::test_webview2_privacy_settings_are_pinned` |
| Data at rest | The state and project paths and file names are the code's | `tests/test_windows_it_sheet.py::test_paths_match_the_code` |
| Data at rest | A project file is JSON, and a pickle is never run | `tests/test_windows_it_sheet.py::test_a_saved_project_is_plain_json_and_never_unpickled` |
| Data at rest | The projects folder is made, and pinned to Quick access | `desktop/windows/tests/test_l2_10_install.py::test_projects_folder_is_made_and_pinned` |
| Files, registry and services | The Start and desktop shortcuts point at the app and Start Here | `desktop/windows/tests/test_l2_10_install.py::test_shortcuts_point_at_the_app` |
| Files, registry and services | No registry, service or scheduled-task writes in the shell or the installer | `tests/test_windows_it_sheet.py::test_nothing_writes_registry_services_or_tasks` |
| Software bill of materials and licences | GPL-family notices in bundled licence texts are found and listed | `tests/test_windows_sbom.py::test_gpl_mentions_are_found_in_bundled_notices` |
| Software bill of materials and licences | An unknown licence fails the build | `tests/test_windows_sbom.py::test_unknown_licence_fails_the_build` |
| Software bill of materials and licences | Re-signed binaries are marked on their components | `tests/test_windows_sbom.py::test_every_unsigned_file_is_attributed_once` |
