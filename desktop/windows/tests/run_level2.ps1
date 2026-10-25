# Runs the Level 2 packaged-app tests (desktop\windows\tests\test_l2_*.py)
# against a built installer, on this PC's interactive desktop.
#
#   run_level2.ps1 -Installer <exe> [-OldInstaller <exe>] [-BrokenInstaller <exe>]
#                  [-K <pytest -k expression>] [-Cleanup]
#
# The tests run from an elevated runner (a scheduled task on the signed-in
# user's desktop, /RL HIGHEST) under an independent uv-managed Python in
# C:\work\l2\venv; each install or app start drops to the limited user or
# SYSTEM. The user named by USERNAME must be signed in at the console.
# Exit code: pytest's (0 = all passed), or 2 if the run could not start or did not
# finish in -TimeoutMinutes (its pytest and tasks are then ended and deleted).
# Windows PowerShell 5.1 rules: ASCII only, no stderr redirection of native
# commands, $LASTEXITCODE checked after each one.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][string]$Installer,
    [string]$OldInstaller = '',
    [string]$BrokenInstaller = '',
    [string]$K = '',
    [switch]$Cleanup,
    [int]$TimeoutMinutes = 240
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$L2 = 'C:\work\l2'
$Tests = $PSScriptRoot
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Utf8 = New-Object Text.UTF8Encoding $false
$User = $env:USERNAME

function Say([string]$line) { [Console]::Out.WriteLine($line) }
function Stop-Run([string]$message) { [Console]::Error.WriteLine($message); exit 2 }

function Invoke-Native([string]$what, [string]$exe, [string[]]$arguments) {
    $out = & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw ($what + ' exited ' + $LASTEXITCODE) }
    return $out
}

function Write-Lf([string]$path, [string[]]$lines) {
    $text = ''
    foreach ($line in $lines) { $text += $line + "`n" }
    [IO.File]::WriteAllText($path, $text, $Utf8)
}

# 1. Someone must be signed in at the console: installs and the shell run
#    on that desktop.
$sessions = (& query user | Out-String)
$active = $false
foreach ($line in ($sessions -split "`n")) {
    if ($line -match ('^\s*>?' + [regex]::Escape($User) + '\s') -and $line -match '\sActive\s') { $active = $true }
}
if (-not $active) { Stop-Run "no interactive session for $User" }

foreach ($exe in @($Installer, $OldInstaller, $BrokenInstaller)) {
    if ($exe -ne '' -and -not (Test-Path -LiteralPath $exe -PathType Leaf)) { Stop-Run "no installer at $exe" }
}
$Installer = (Resolve-Path -LiteralPath $Installer).Path
if ($OldInstaller -ne '') { $OldInstaller = (Resolve-Path -LiteralPath $OldInstaller).Path }
if ($BrokenInstaller -ne '') { $BrokenInstaller = (Resolve-Path -LiteralPath $BrokenInstaller).Path }

# 2. The runner's own Python, independent of the install (reused, so an
#    offline machine can rerun once it exists).
New-Item -ItemType Directory -Force -Path $L2, (Join-Path $L2 'jobs'), (Join-Path $L2 'logs') | Out-Null
$venv = Join-Path $L2 'venv'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $uv = Get-Command uv.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($null -ne $uv) { $uv = @($uv)[0].Source } else { $uv = Join-Path $L2 'uv.exe' }
    if (-not (Test-Path -LiteralPath $uv)) { Stop-Run 'uv not found (on PATH or C:\work\l2\uv.exe)' }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $L2 'python'
    Invoke-Native 'uv venv' $uv @('venv', $venv, '--python', '3.13') | ForEach-Object { Say $_ }
    Invoke-Native 'uv pip install' $uv @('pip', 'install', '--python', $python, 'pytest>=9,<10') | ForEach-Object { Say $_ }
}

# 3. The job runner scheduled tasks start.
Copy-Item -LiteralPath (Join-Path $Tests 'l2_job.py') -Destination (Join-Path $L2 'l2_job.py') -Force

# 4. The run, as a .cmd the elevated task executes.
$files = @(Get-ChildItem -LiteralPath $Tests -File -Filter 'test_l2_*.py' | Sort-Object Name | ForEach-Object { '"' + $_.FullName + '"' })
$log = Join-Path $L2 'run.log'
$report = Join-Path $L2 'report.json'
$junit = Join-Path $L2 'junit.xml'
foreach ($old in @($log, $report, $junit)) { if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Force } }
$kArg = ''
if ($K -ne '') { $kArg = '-k "' + $K + '" ' }
$cleanupValue = '0'
if ($Cleanup) { $cleanupValue = '1' }
$cmd = Join-Path 'C:\work\run' 'l2.cmd'
$basetemp = Join-Path $L2 ('bt-' + (Get-Date -Format 'HHmmss'))
New-Item -ItemType Directory -Force -Path 'C:\work\run' | Out-Null
$lines = @(
    '@echo off',
    ('cd /d "' + $Repo + '"'),
    ('set "FOODOPT_L2_INSTALLER=' + $Installer + '"'),
    ('set "FOODOPT_L2_OLD_INSTALLER=' + $OldInstaller + '"'),
    ('set "FOODOPT_L2_BROKEN_INSTALLER=' + $BrokenInstaller + '"'),
    ('set "FOODOPT_L2_REPORT=' + $report + '"'),
    ('set "FOODOPT_L2_CLEANUP=' + $cleanupValue + '"'),
    ('set "FOODOPT_L2_ROOT=' + $L2 + '"'),
    'set "PYTHONUTF8=1"',
    ('"' + $python + '" -m pytest ' + ($files -join ' ') + ' ' + $kArg + '-rA -p no:cacheprovider --basetemp "' +
        $basetemp + '" --junitxml "' + $junit + '" > "' + $log + '" 2>&1'),
    ('>> "' + $log + '" echo EXIT=%ERRORLEVEL%')
)
[IO.File]::WriteAllText($cmd, (($lines -join "`r`n") + "`r`n"), (New-Object Text.ASCIIEncoding))

# 5. Run it elevated on the user's desktop, and wait for its EXIT= line.
$task = 'FoodOptL2-run'
$exit = $null
$started = Get-Date
try {
    # A start boundary in the past: the task never fires on schedule, only on /Run.
    Invoke-Native 'schtasks /Create' 'schtasks.exe' @('/Create', '/F', '/TN', $task, '/TR', $cmd, '/SC', 'ONCE',
        '/SD', '01/01/2020', '/ST', '00:00', '/RU', $User, '/IT', '/RL', 'HIGHEST') | Out-Null
    Invoke-Native 'schtasks /Run' 'schtasks.exe' @('/Run', '/TN', $task) | Out-Null
    Say "running Level 2 (log: $log)"
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while ($null -eq $exit) {
        Start-Sleep -Seconds 5
        if (Test-Path -LiteralPath $log) {
            $tail = @(Get-Content -LiteralPath $log -Tail 3 -ErrorAction SilentlyContinue)
            foreach ($line in $tail) { if ($line -match '^EXIT=(-?\d+)') { $exit = [int]$Matches[1] } }
        }
        if ($null -eq $exit -and (Get-Date) -gt $deadline) {
            Stop-Run "Level 2 did not finish in $TimeoutMinutes minutes"
        }
    }
}
finally {
    if ($null -eq $exit) {
        # No EXIT= line (deadline passed, or a failure). /End stops only the
        # task's cmd.exe, so end this run's pytest too (its --basetemp names
        # it; started after this script), and the job tasks it can no longer
        # delete itself.
        & schtasks.exe /End /TN $task | Out-Null
        $mark = '*' + [Management.Automation.WildcardPattern]::Escape($basetemp) + '*'
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
            $_.CommandLine -like $mark -and $_.CreationDate -ge $started
        } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Get-ScheduledTask -TaskName 'FoodOptL2-*' -ErrorAction SilentlyContinue |
            Unregister-ScheduledTask -Confirm:$false -ErrorAction SilentlyContinue
    }
    & schtasks.exe /Delete /F /TN $task | Out-Null
}
Get-Content -LiteralPath $log -Tail 40 | ForEach-Object { Say $_ }
if (Test-Path -LiteralPath $report) {
    Say '--- report.json ---'
    Get-Content -LiteralPath $report | ForEach-Object { Say $_ }
}
exit $exit
