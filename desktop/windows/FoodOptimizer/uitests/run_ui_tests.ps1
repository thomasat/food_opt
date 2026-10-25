# Runs FoodOptimizer.UiTests on the interactive desktop (a scheduled task
# with /IT), against a folder staged by stage_layout.ps1, and waits for it.
# -Installer <exe> is what the Installer and Installed tests install
# (phase 5), -BrokenInstaller <exe> make_test_installer.ps1's blocked
# pre-warm variant; they ignore the stage. Their filter:
#   -Filter "--filter-trait Category=Installer --filter-trait Category=Installed"
# ASCII only: Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI.
param(
    [Parameter(Mandatory = $true)][string]$Stage,
    [string]$Filter = "--filter-not-trait Category=RealApp --filter-not-trait Category=Installer --filter-not-trait Category=Installed",
    [string]$Results = "",
    [string]$Installer = "",
    [string]$BrokenInstaller = ""
)
if (-not $Results) { $Results = "$Stage-results" }
$sln = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$user = $env:USERNAME

$sessions = (query user 2>$null) | Out-String
$active = $sessions -split "`n" | Where-Object { $_ -match [regex]::Escape($user) -and $_ -match "Active" }
if (-not $active) { Write-Output "no interactive session for $user"; exit 2 }

$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"
$env:DOTNET_NOLOGO = "1"
Push-Location $sln
dotnet build FoodOptimizer.UiTests -c Release -v q -clp:NoSummary | Out-Host
$built = $LASTEXITCODE
Pop-Location
if ($built -ne 0) { Write-Output "ui tests: build failed ($built)"; exit $built }

New-Item -ItemType Directory -Force -Path $Results | Out-Null
$out = Join-Path $Results "out.txt"
$rc = Join-Path $Results "rc.txt"
Remove-Item -LiteralPath $out, $rc -ErrorAction SilentlyContinue
$cmd = Join-Path $Results "run.cmd"
$lines = @(
    "@echo off",
    "set FOODOPT_UI_STAGE=$Stage",
    "set FOODOPT_UI_RESULTS=$Results",
    "set FOODOPT_UI_INSTALLER=$Installer",
    "set FOODOPT_UI_BROKEN_INSTALLER=$BrokenInstaller",
    "set DOTNET_CLI_TELEMETRY_OPTOUT=1",
    "set DOTNET_NOLOGO=1",
    "cd /d `"$sln`"",
    "dotnet test --project FoodOptimizer.UiTests --no-build -c Release $Filter > `"$out`" 2>&1",
    ">> `"$rc`" echo %ERRORLEVEL%"
)
[IO.File]::WriteAllLines($cmd, $lines, [Text.Encoding]::ASCII)

$task = "FoodOptUiTests"
schtasks /Create /TN $task /TR "`"$cmd`"" /SC ONCE /ST 23:59 /RU $user /IT /F | Out-Null
schtasks /Run /TN $task | Out-Null
$deadline = (Get-Date).AddMinutes(40)
while (-not (Test-Path -LiteralPath $rc) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
schtasks /Delete /TN $task /F | Out-Null

if (Test-Path -LiteralPath $out) { Get-Content -LiteralPath $out }
if (-not (Test-Path -LiteralPath $rc)) { Write-Output "ui tests: rc=timeout"; exit 3 }
$code = [int]((Get-Content -LiteralPath $rc -Raw).Trim())
Write-Output "ui tests: rc=$code"
exit $code
