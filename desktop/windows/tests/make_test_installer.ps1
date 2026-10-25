# Builds one of Level 2's two test installers from a built image:
#
#   make_test_installer.ps1 -Variant old|broken [-Image <image>] [-Out <folder>]
#
#   old     FoodOptimizer-0.0.1-Windows-old.exe: version 0.0.1, with files an
#           older lock and app had (a stale package, an old module, a stale
#           shell DLL at the root and under runtimes) that an upgrade must
#           not leave behind.
#   broken  FoodOptimizer-<ver>-Windows-broken.exe: gpytorch raises on import,
#           as if security software blocked a file, so the pre-warm fails.
#
# -Image defaults to desktop\windows\build\FoodOptimizer (build_installer.ps1);
# -Out to C:\work\l2\installers. Prints the installer's path. Test only.
# Windows PowerShell 5.1 rules: ASCII only, $LASTEXITCODE checked.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][ValidateSet('old', 'broken')][string]$Variant,
    [string]$Image = '',
    [string]$Out = 'C:\work\l2\installers'
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$WindowsDir = Split-Path -Parent $PSScriptRoot
$Utf8 = New-Object Text.UTF8Encoding $false
if ($Image -eq '') { $Image = Join-Path $WindowsDir 'build\FoodOptimizer' }
if (-not (Test-Path -LiteralPath (Join-Path $Image 'FoodOptimizer.exe'))) {
    [Console]::Error.WriteLine("no built image at $Image (run build_installer.ps1 -TestBuild)")
    exit 2
}

# Inno Setup's compiler, in build_installer.ps1's order.
function Find-Iscc {
    $candidates = @()
    if ($env:FOODOPT_ISCC) { $candidates += $env:FOODOPT_ISCC }
    $onPath = Get-Command ISCC.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($null -ne $onPath) { $candidates += @($onPath)[0].Source }
    if ($env:LOCALAPPDATA) { $candidates += (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe') }
    if (${env:ProgramFiles(x86)}) { $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe') }
    if ($env:ProgramFiles) { $candidates += (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe') }
    foreach ($c in $candidates) {
        if ($c -and (Test-Path -LiteralPath $c -PathType Leaf)) { return $c }
    }
    return ''
}
$Iscc = Find-Iscc
if ($Iscc -eq '') {
    [Console]::Error.WriteLine('Inno Setup 6.3 or later (ISCC.exe) not found')
    exit 2
}

function Write-Lf([string]$path, [string]$text) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path) | Out-Null
    [IO.File]::WriteAllText($path, $text, $Utf8)
}

# 1. A private copy of the image.
$copy = Join-Path ('C:\work\l2\img-' + $Variant) 'FoodOptimizer'
New-Item -ItemType Directory -Force -Path $copy | Out-Null
& robocopy.exe $Image $copy /MIR /NFL /NDL /NJH /NJS /NP /R:2 /W:1 | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy $Image -> $copy failed ($LASTEXITCODE)" }

$ver = ([IO.File]::ReadAllText((Join-Path (Split-Path -Parent $WindowsDir) 'VERSION'), $Utf8)).Trim()
$site = Join-Path $copy 'python\Lib\site-packages'
if ($Variant -eq 'old') {
    # 2. What an older lock and app left: a package, its metadata, two
    #    modules and two shell DLLs.
    Write-Lf (Join-Path $site 'stale_pkg\__init__.py') "STALE = True`n"
    Write-Lf (Join-Path $site 'stale_pkg-1.0.dist-info\METADATA') "Metadata-Version: 2.1`nName: stale-pkg`nVersion: 1.0`n"
    Write-Lf (Join-Path $copy 'app\old_module.py') "OLD = True`n"
    Write-Lf (Join-Path $copy 'shared\launcher\old_helper.py') "OLD = True`n"
    Copy-Item -LiteralPath (Join-Path $copy 'python\python3.dll') -Destination (Join-Path $copy 'stale_shell.dll') -Force
    $native = Join-Path $copy 'runtimes\win-x64\native'
    New-Item -ItemType Directory -Force -Path $native | Out-Null
    Copy-Item -LiteralPath (Join-Path $copy 'python\python3.dll') -Destination (Join-Path $native 'FoodOptimizer.Stale.dll') -Force
    $appVersion = '0.0.1'
    $name = 'FoodOptimizer-0.0.1-Windows-old'
} else {
    # 3. gpytorch blocked: its package raises, and its unchecked-hash .pyc
    #    (which would still load) is gone.
    Write-Lf (Join-Path $site 'gpytorch\__init__.py') "raise ImportError(`"blocked by test`")`n"
    $pyc = Join-Path $site 'gpytorch\__pycache__\__init__.cpython-313.pyc'
    if (Test-Path -LiteralPath $pyc) { Remove-Item -LiteralPath $pyc -Force }
    $appVersion = $ver
    $name = "FoodOptimizer-$ver-Windows-broken"
}

# Both variants need the release's disk figure, from size_gate.json.
$gate = Get-Content -LiteralPath (Join-Path $WindowsDir 'size_gate.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$diskFreeGb = 2
if ($null -ne $gate.disk_free_gb) { $diskFreeGb = [int]$gate.disk_free_gb }

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$exe = Join-Path $Out ($name + '.exe')
if (Test-Path -LiteralPath $exe) { Remove-Item -LiteralPath $exe -Force }
& $Iscc '/Qp' "/DAppVersion=$appVersion" "/DImageDir=$copy" "/DDiskFreeGB=$diskFreeGb" "/O$Out" "/F$name" (Join-Path $WindowsDir 'setup.iss') | Out-Null
if ($LASTEXITCODE -ne 0) { throw ('ISCC exit ' + $LASTEXITCODE) }
[Console]::Out.WriteLine($exe)
exit 0
