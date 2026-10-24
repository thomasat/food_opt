# Builds the Windows environment: <OutDir>\FoodOptimizer\ holding python.org
# Python with the Windows lock installed, Microsoft's C++ runtime, the app,
# the shared launcher and Start Here. Build time only (the VM now, CI later);
# nothing in the image calls PowerShell, uv or the network at run time.
# The version is desktop/VERSION; there is no other.
#
#   build_installer.ps1 [-Version X.Y.Z]   release build (clean tree at tag vX.Y.Z)
#   build_installer.ps1 -Check [-Version X.Y.Z]   the release guards only; builds nothing
#   build_installer.ps1 -TestBuild         internal build, release guards off
#   -OutDir DIR                            default desktop\windows\build
#
# Exit codes: 0 built (or -Check passed); 2 refused before anything was
# fetched or created; 1 the build failed ("build failed: <reason>", no record).
# Windows PowerShell 5.1 rules: ASCII only, no stderr redirection of native
# commands (uv writes progress there), $LASTEXITCODE checked after each one.
param(
    [string]$Version = '',
    [switch]$Check,
    [switch]$TestBuild,
    [string]$OutDir = '',
    [Parameter(ValueFromRemainingArguments = $true)]
    $Rest
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$USAGE = 'usage: build_installer.ps1 [-Version X.Y.Z] [-Check] [-TestBuild] [-OutDir DIR]'
$ROOT_BUDGET = 80
$WindowsDir = $PSScriptRoot
$DesktopDir = Split-Path -Parent $WindowsDir
$RepoDir = Split-Path -Parent $DesktopDir

function Say([string]$line) { [Console]::Out.WriteLine($line) }

function Refuse([string]$message) {
    [Console]::Error.WriteLine($message)
    exit 2
}

function Fail([string]$message) {
    [Console]::Error.WriteLine('build failed: ' + $message)
    exit 1
}

# Runs a native command and throws if it exits non-zero. Its stderr is never
# redirected: under 5.1 with 'Stop', a redirected stderr line would throw.
function Invoke-Native([string]$what, [string]$exe, [string[]]$arguments) {
    $out = & $exe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw ($what + ' exited ' + $LASTEXITCODE)
    }
    return $out
}

# ---------- 1. arguments ----------
if ($Rest -ne $null -and @($Rest).Count -gt 0) { Refuse $USAGE }
if ($Check -and $TestBuild) { Refuse $USAGE }
if ($TestBuild -and $Version -ne '') { Refuse $USAGE }

# ---------- 2. PowerShell ----------
$psv = $PSVersionTable.PSVersion
if ($psv.Major -lt 5 -or ($psv.Major -eq 5 -and $psv.Minor -lt 1)) {
    Refuse 'refusing to build: needs Windows PowerShell 5.1 or later'
}

# ---------- 3. version ----------
# One X.Y.Z line. Exactly one trailing LF is taken off; a CR (a file saved
# with Windows line endings) or anything else fails.
$ver = ''
$versionFile = Join-Path $DesktopDir 'VERSION'
if (Test-Path -LiteralPath $versionFile -PathType Leaf) {
    $ver = [IO.File]::ReadAllText($versionFile, (New-Object Text.UTF8Encoding $false))
    if ($ver.EndsWith("`n")) { $ver = $ver.Substring(0, $ver.Length - 1) }
}
if ($ver -cnotmatch '\A[0-9]+\.[0-9]+\.[0-9]+\z') {
    Refuse 'refusing to build: desktop/VERSION must hold one X.Y.Z line'
}

# ---------- 4. bundle manifest ----------
# The files both apps ship, one repo-relative path per line (names may hold
# spaces). A last line without a newline still counts.
$manifestFile = Join-Path $DesktopDir 'shared\bundle_manifest.txt'
if (-not (Test-Path -LiteralPath $manifestFile -PathType Leaf)) {
    Refuse 'refusing to build: bundle_manifest.txt is missing'
}
$manifestText = [IO.File]::ReadAllText($manifestFile, (New-Object Text.UTF8Encoding $false))
if ($manifestText.Contains("`r")) {
    Refuse 'refusing to build: bundle_manifest.txt has Windows line endings'
}
$entries = @()
if ($manifestText -ne '') {
    $lines = $manifestText.Split("`n")
    if ($manifestText.EndsWith("`n")) { $lines = $lines[0..($lines.Length - 2)] }
    foreach ($entry in $lines) {
        if ($entry -eq '' -or $entry.StartsWith('/') -or $entry.StartsWith('\') -or
                $entry.Contains(':') -or $entry.Contains('\') -or $entry -eq '..' -or
                $entry.StartsWith('../') -or $entry.EndsWith('/..') -or $entry.Contains('/../')) {
            Refuse "refusing to build: bundle_manifest.txt entry must be a repo-relative path: $entry"
        }
        if (-not (Test-Path -LiteralPath (Join-Path $RepoDir $entry) -PathType Leaf)) {
            Refuse "refusing to build: listed in bundle_manifest.txt but missing from the repo: $entry"
        }
        $entries += $entry
    }
}
if ($entries.Count -eq 0) {
    Refuse 'refusing to build: bundle_manifest.txt is empty'
}

# ---------- 5. tools ----------
$tools = @('git')
if (-not $Check) { $tools += 'uv' }
foreach ($tool in $tools) {
    if ($null -eq (Get-Command $tool -CommandType Application -ErrorAction SilentlyContinue)) {
        Refuse "refusing to build: $tool not found on PATH"
    }
}

# ---------- 6. release guards (release builds and -Check) ----------
if (-not $TestBuild) {
    if ($Version -ne '' -and $Version -cne $ver) {
        Refuse "refusing to build: version $Version disagrees with desktop/VERSION ($ver)"
    }
    # Ignored paths (build\, .cache\) don't count; untracked ones do.
    $status = & git -C $RepoDir status --porcelain --untracked-files=normal
    if ($LASTEXITCODE -ne 0) {
        Refuse 'refusing to build: not a git checkout (build from a clean worktree at the tag)'
    }
    if (@($status | Where-Object { $_ -ne '' }).Count -gt 0) {
        Refuse 'refusing to build: uncommitted changes (build from a clean worktree at the tag)'
    }
    $head = (& git -C $RepoDir rev-parse HEAD | Out-String).Trim()
    $tagSha = (& git -C $RepoDir rev-parse -q --verify "refs/tags/v$ver^{commit}" | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { $tagSha = '' }
    if ($head -ne $tagSha) {
        $short = (& git -C $RepoDir rev-parse --short HEAD | Out-String).Trim()
        Refuse "refusing to build: HEAD is not tag v$ver (tag first: git tag -a v$ver $short)"
    }
}

# ---------- 7. the image root ----------
if ($OutDir -eq '') { $OutDir = Join-Path $WindowsDir 'build' }
$OutDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutDir).TrimEnd('\')
$Image = Join-Path $OutDir 'FoodOptimizer'
$n = ($Image + '\').Length
if ($n -gt $ROOT_BUDGET) {
    Refuse "refusing to build: image folder is $n characters; at most 80 (clone to a shorter path)"
}

# ---------- 8. -Check ----------
if ($Check) {
    Say "ready to build $ver from v$ver"
    Say ('manifest: ' + $entries.Count + ' files')
    exit 0
}

# ---------- 9. the build ----------
if ($TestBuild) { Say 'TEST BUILD - not for release' }
Say '========================================================='
Say ' UNSIGNED build - internal testing only.'
Say '========================================================='

Fail 'build not implemented yet'
