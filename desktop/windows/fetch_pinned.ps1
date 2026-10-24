# Downloads one pinned file from pins.json into the cache folder, verifies its
# sha256, and prints its full path on stdout. Build time only (never shipped).
# Mirrors desktop/fetch_uv.sh: a cached file is re-hashed on every run, and a
# mismatch removes it and fails, so a corrupted or tampered cache is never
# trusted.
#
#   fetch_pinned.ps1 -Name python|vc_runtime|wix [-CacheDir DIR]
#
# Exit codes: 0 verified (path printed), 1 download or verification failed,
# 2 bad arguments.
param(
    [string]$Name = '',
    [string]$CacheDir = ''
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Fail([int]$code, [string]$message) {
    [Console]::Error.WriteLine($message)
    exit $code
}

function Get-Sha256([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (@('python', 'vc_runtime', 'wix') -notcontains $Name) {
    Fail 2 'usage: fetch_pinned.ps1 -Name python|vc_runtime|wix [-CacheDir DIR]'
}
if ($CacheDir -eq '') {
    $CacheDir = Join-Path $PSScriptRoot '.cache'
}

$pins = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'pins.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$pin = $pins.$Name
$expected = $pin.sha256.ToLowerInvariant()

if (-not (Test-Path -LiteralPath $CacheDir)) {
    New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
}
$CacheDir = (Resolve-Path -LiteralPath $CacheDir).ProviderPath
$target = Join-Path $CacheDir $pin.file
$part = $target + '.part'

if (Test-Path -LiteralPath $target) {
    if ((Get-Sha256 $target) -ne $expected) {
        Remove-Item -LiteralPath $target -Force
        Fail 1 ('cached ' + $pin.file + ' failed verification; removed - run again')
    }
    Write-Output $target
    exit 0
}

if (Test-Path -LiteralPath $part) {
    Remove-Item -LiteralPath $part -Force
}
[Console]::Error.WriteLine('downloading ' + $pin.url)
try {
    Invoke-WebRequest -Uri $pin.url -OutFile $part -UseBasicParsing
} catch {
    if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
    Fail 1 ('download of ' + $Name + ' failed: ' + $_.Exception.Message)
}
$got = Get-Sha256 $part
if ($got -ne $expected) {
    Remove-Item -LiteralPath $part -Force
    Fail 1 ('download of ' + $Name + ' does not match pins.json (got ' + $got + ')')
}
Move-Item -LiteralPath $part -Destination $target -Force
Write-Output $target
exit 0
