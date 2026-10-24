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
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter()][string]$Version = '',
    [Parameter()][switch]$Check,
    [Parameter()][switch]$TestBuild,
    [Parameter()][string]$OutDir = '',
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

$Utf8 = New-Object Text.UTF8Encoding $false
$Record = Join-Path $OutDir 'FoodOptimizer.build.txt'
$FilesList = Join-Path $OutDir 'FoodOptimizer.files.txt'
$UnsignedList = Join-Path $OutDir 'FoodOptimizer.unsigned.txt'
$CacheDir = Join-Path $WindowsDir '.cache'
$BuildEnv = Join-Path $WindowsDir 'build_env.py'
$Py = Join-Path $Image 'python\python.exe'
$Pins = Get-Content -LiteralPath (Join-Path $WindowsDir 'pins.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$started = Get-Date

function Get-Sha256([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Lf([string]$path, [string[]]$lines) {
    $text = ''
    foreach ($line in $lines) { $text += $line + "`n" }
    [IO.File]::WriteAllText($path, $text, $Utf8)
}

function Remove-Tree([string]$path) {
    if (Test-Path -LiteralPath $path) { [IO.Directory]::Delete($path, $true) }
}

# Every "key=value" line build_env.py printed, as a hashtable.
function Read-Values($lines) {
    $values = @{}
    foreach ($line in @($lines)) {
        if ($line -match '^([a-z0-9_]+)=(.*)$') { $values[$Matches[1]] = $Matches[2] }
    }
    return $values
}

# Runs build_env.py with the bundled Python, echoes its output, and returns it.
function Invoke-BuildEnv([string[]]$arguments) {
    $out = & $Py -I -B $BuildEnv @arguments
    $code = $LASTEXITCODE
    foreach ($line in @($out)) { Say $line }
    if ($code -ne 0) { throw ('build_env.py ' + $arguments[0] + ' exited ' + $code) }
    return $out
}

function Assert-Signed([string]$path, [string]$signer) {
    $sig = Get-AuthenticodeSignature -LiteralPath $path
    $subject = ''
    if ($sig.SignerCertificate -ne $null) { $subject = $sig.SignerCertificate.Subject }
    if ($sig.Status -ne 'Valid' -or $subject -notmatch ('^CN=' + [regex]::Escape($signer) + ',')) {
        throw ($path + ' is not signed by ' + $signer + ' (' + $sig.Status + ', ' + $subject + ')')
    }
}

$step = 'start'
try {
    # Recorded before anything is written, so the record says what was built.
    $step = 'git'
    $commit = (Invoke-Native 'git rev-parse' 'git' @('-C', $RepoDir, 'rev-parse', 'HEAD') | Out-String).Trim()
    $porcelain = Invoke-Native 'git status' 'git' @('-C', $RepoDir, 'status', '--porcelain', '--untracked-files=normal')
    $treeClean = 'yes'
    if (@($porcelain | Where-Object { $_ -ne '' }).Count -gt 0) { $treeClean = 'no' }
    $uvVersion = ((Invoke-Native 'uv --version' 'uv' @('--version') | Out-String).Trim() -split ' ')[1]

    # 1. No record from an older build may outlive this one, nor any file of
    #    the old image (a package an older lock had, a removed app module).
    $step = 'clean'
    foreach ($side in @($Record, $FilesList, $UnsignedList)) {
        if (Test-Path -LiteralPath $side) { Remove-Item -LiteralPath $side -Force }
    }
    try {
        Remove-Tree $Image
    } catch {
        throw ('could not remove ' + $Image + ' (is its Python running?)')
    }
    New-Item -ItemType Directory -Path $Image -Force | Out-Null

    # 2-3. python.org's NuGet package; its tools\ folder is the Python.
    $step = 'python'
    $nupkg = & (Join-Path $WindowsDir 'fetch_pinned.ps1') -Name python -CacheDir $CacheDir
    if ($LASTEXITCODE -ne 0) { throw 'fetch_pinned.ps1 -Name python failed' }
    $nupkg = (@($nupkg) | Select-Object -Last 1).Trim()
    $stage = Join-Path $CacheDir 'stage-python'
    Remove-Tree $stage
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($nupkg, $stage)
    Move-Item -LiteralPath (Join-Path $stage 'tools') -Destination (Join-Path $Image 'python')
    Remove-Tree $stage

    # 4-5. The PSF's signature, and the exact version and platform. Never
    #      platform.machine(): under emulation it reports the host's ARM64.
    $step = 'python signature'
    foreach ($name in @('python.exe', 'python313.dll')) {
        Assert-Signed (Join-Path $Image "python\$name") $Pins.python.signer
    }
    $step = 'python version'
    $probe = (Invoke-Native 'python.exe' $Py @('-I', '-c', 'import sys,sysconfig;print(sys.version.split()[0], sysconfig.get_platform())') | Out-String).Trim()
    $wantedProbe = $Pins.python.version + ' win-amd64'
    if ($probe -ne $wantedProbe) { throw ('python.exe reports ' + $probe + ', expected ' + $wantedProbe) }

    # 6. The Windows lock and nothing else: sync removes what the lock lacks
    #    (NuGet's pip); copy, so the image shares no hard links with uv's cache.
    $step = 'uv'
    $lock = Join-Path $DesktopDir 'requirements-windows.lock.txt'
    Invoke-Native 'uv pip sync' 'uv' @('pip', 'sync', '--python', $Py, '--require-hashes',
        '--only-binary', ':all:', '--link-mode', 'copy', '--no-config',
        '--default-index', 'https://pypi.org/simple', $lock) | ForEach-Object { Say $_ }

    # 7. Microsoft's C++ runtime beside python.exe: torch needs msvcp140.dll,
    #    python.org doesn't ship it, and sklearn ships its own copy under the
    #    same name (spike B4). One Microsoft-signed set of one version, from
    #    Microsoft's own redistributable: /layout unpacks the bundle's
    #    embedded payloads (no network), msiexec /a extracts the files
    #    (nothing is installed, no admin needed).
    $step = 'vc runtime'
    $vcPin = $Pins.vc_runtime
    $redist = & (Join-Path $WindowsDir 'fetch_pinned.ps1') -Name vc_runtime -CacheDir $CacheDir
    if ($LASTEXITCODE -ne 0) { throw 'fetch_pinned.ps1 -Name vc_runtime failed' }
    $redist = (@($redist) | Select-Object -Last 1).Trim()
    Assert-Signed $redist $vcPin.signer
    $vcStage = Join-Path $CacheDir 'stage-vc'
    Remove-Tree $vcStage
    $layout = Join-Path $vcStage 'layout'
    $admin = Join-Path $vcStage 'admin'
    $proc = Start-Process -FilePath $redist -ArgumentList @('/layout', ('"' + $layout + '"'), '/quiet', '/norestart') -Wait -PassThru
    $msi = @()
    if (Test-Path -LiteralPath $layout) {
        $msi = @(Get-ChildItem -LiteralPath $layout -Recurse -File -Filter 'vc_runtimeMinimum_x64.msi')
    }
    if ($proc.ExitCode -ne 0 -or $msi.Count -ne 1) {
        throw ('could not extract the VC runtime from ' + $redist + ' (/layout exit ' + $proc.ExitCode + ', ' + $msi.Count + ' vc_runtimeMinimum_x64.msi)')
    }
    $proc = Start-Process -FilePath 'msiexec.exe' -Wait -PassThru -ArgumentList @('/a', ('"' + $msi[0].FullName + '"'), '/qn',
        ('TARGETDIR="' + $admin + '"'), '/L*v', ('"' + (Join-Path $vcStage 'msiexec.log') + '"'))
    if ($proc.ExitCode -ne 0) {
        throw ('could not extract the VC runtime from ' + $msi[0].FullName + ' (msiexec /a exit ' + $proc.ExitCode + ')')
    }
    $vcFiles = @()
    foreach ($name in $vcPin.files) {
        $found = @(Get-ChildItem -LiteralPath $admin -Recurse -File | Where-Object { $_.Name -eq $name })
        if ($found.Count -ne 1) { throw ('expected one ' + $name + ' in the VC runtime, found ' + $found.Count) }
        Assert-Signed $found[0].FullName $vcPin.signer
        $v = $found[0].VersionInfo
        $fileVersion = '{0}.{1}.{2}.{3}' -f $v.FileMajorPart, $v.FileMinorPart, $v.FileBuildPart, $v.FilePrivatePart
        if ($fileVersion -ne $vcPin.version) {
            throw ($name + ' is ' + $fileVersion + '; pins.json says ' + $vcPin.version)
        }
        $vcFiles += $found[0].FullName
    }
    # Never older than a copy a wheel ships (today sklearn's msvcp140.dll).
    $pinned = [version]$vcPin.version
    $sitePackages = Join-Path $Image 'python\Lib\site-packages'
    Get-ChildItem -LiteralPath $sitePackages -Recurse -File |
        Where-Object { $vcPin.files -contains $_.Name.ToLowerInvariant() } | ForEach-Object {
            $v = $_.VersionInfo
            $wheel = [version]('{0}.{1}.{2}.{3}' -f $v.FileMajorPart, $v.FileMinorPart, $v.FileBuildPart, $v.FilePrivatePart)
            if ($wheel -gt $pinned) {
                $rel = $_.FullName.Substring($Image.Length + 1).Replace('\', '/')
                throw ('bundled ' + $vcPin.version + ' is older than ' + $rel + ' ' + $wheel + '; update pins.json')
            }
        }
    # Microsoft's app-local rule: one set, from one version, so NuGet's own
    # vcruntime140*.dll are replaced.
    foreach ($file in $vcFiles) {
        Copy-Item -LiteralPath $file -Destination (Join-Path $Image 'python') -Force
    }
    Remove-Tree $vcStage
    $vcVersion = $vcPin.version
    $vcSha = Get-Sha256 $redist

    # 8. Layout, strip, precompile, gate: build_env.py, with the bundled Python.
    $step = 'assemble'
    Invoke-BuildEnv @('assemble', $RepoDir, $Image) | Out-Null
    $step = 'strip'
    Invoke-BuildEnv @('strip', $Image) | Out-Null
    $step = 'compile'
    Invoke-BuildEnv @('compile', $Image) | Out-Null
    $step = 'gate'
    $gate = Read-Values (Invoke-BuildEnv @('gate', $RepoDir, $Image))

    # 9. What phase 6 signs: every binary not validly signed by its vendor.
    $step = 'unsigned list'
    $unsigned = New-Object System.Collections.Generic.List[string]
    $prefix = $Image.Length + 1
    # Filtered by extension here: 5.1 ignores an include filter on a literal path.
    Get-ChildItem -LiteralPath $Image -Recurse -File |
            Where-Object { @('.exe', '.dll', '.pyd') -contains $_.Extension.ToLowerInvariant() } | ForEach-Object {
        if ((Get-AuthenticodeSignature -LiteralPath $_.FullName).Status -ne 'Valid') {
            $unsigned.Add($_.FullName.Substring($prefix).Replace('\', '/'))
        }
    }
    $sorted = $unsigned.ToArray()
    [Array]::Sort($sorted, [StringComparer]::Ordinal)
    Write-Lf $UnsignedList $sorted

    # 10. The per-file manifest.
    $step = 'files'
    $files = Read-Values (Invoke-BuildEnv @('files', $Image, $FilesList))

    # 11. The record, written last: a failed build leaves none.
    $step = 'record'
    $testBuildValue = 'no'
    if ($TestBuild) { $testBuildValue = 'yes' }
    Write-Lf $Record @(
        "version=$ver",
        "commit=$commit",
        "tree_clean=$treeClean",
        "test_build=$testBuildValue",
        ('mac_lock_sha256=' + (Get-Sha256 (Join-Path $DesktopDir 'requirements.lock.txt'))),
        ('windows_lock_sha256=' + (Get-Sha256 $lock)),
        ('python_version=' + $Pins.python.version),
        ('python_nupkg_sha256=' + (Get-Sha256 $nupkg)),
        "vc_runtime_version=$vcVersion",
        "vc_redist_sha256=$vcSha",
        "uv_version=$uvVersion",
        ('image_files=' + $files['image_files']),
        ('image_bytes=' + $files['image_bytes']),
        ('image_sha256=' + $files['image_sha256']),
        ('longest_path_chars=' + $gate['longest_path_chars']),
        ('unsigned_binaries=' + $sorted.Length),
        'signed=no'
    )
} catch {
    Fail ($step + ': ' + $_.Exception.Message)
}

$mb = [math]::Round([double]$files['image_bytes'] / 1MB)
$seconds = [math]::Round(((Get-Date) - $started).TotalSeconds)
Say ('Built: ' + $Image + ' (' + $files['image_files'] + ' files, ' + $mb + ' MB)')
Say ('build time: ' + $seconds + ' s')
Say 'UNSIGNED build - internal testing only.'
exit 0
