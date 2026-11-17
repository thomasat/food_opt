# Stages an install folder for the shell's UI tests: phase 3b's image (or,
# before it exists, the spike Python plus the repo's shared parts), the
# published FoodOptimizer.exe on top, and the probe app or the real app.
# ASCII only: Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI.
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Image = "",
    [string]$Python = "C:\spike\py\tools",
    [ValidateSet("probe", "real")][string]$App = "probe",
    [string]$Publish = ""
)
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
if (-not $Image) { $Image = Join-Path $repo "desktop\windows\build\FoodOptimizer" }
if (-not $Publish) { $Publish = Join-Path $repo "desktop\.cache\shell\publish\FoodOptimizer.Shell\release_win-x64" }

function Mirror($from, $to) {
    robocopy $from $to /MIR /NFL /NDL /NJH /NJS /NP /R:2 /W:1 | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy $from -> $to failed ($LASTEXITCODE)" }
}
function CopyFiles($from, $to, [string[]]$names) {
    robocopy $from $to $names /NFL /NDL /NJH /NJS /NP /R:2 /W:1 | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy $from -> $to failed ($LASTEXITCODE)" }
}

if (-not (Test-Path -LiteralPath (Join-Path $Publish "FoodOptimizer.exe"))) {
    throw "no FoodOptimizer.exe in $Publish (run dotnet publish FoodOptimizer.Shell -c Release)"
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null

if (Test-Path -LiteralPath (Join-Path $Image "python\python.exe")) {
    $mode = "3b"
    foreach ($d in "python", "app", "shared") { Mirror (Join-Path $Image $d) (Join-Path $Out $d) }
    robocopy $Image $Out /LEV:1 /NFL /NDL /NJH /NJS /NP /R:2 /W:1 | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy of the image root failed ($LASTEXITCODE)" }
} else {
    $mode = "fallback"
    $py = Join-Path $Out "python"
    $stamp = Join-Path $py "stage.stamp"
    $same = (Test-Path -LiteralPath $stamp) -and ((Get-Content -LiteralPath $stamp -Raw).Trim() -eq $Python)
    if (-not $same) {
        Mirror $Python $py
        Set-Content -LiteralPath $stamp -Value $Python -Encoding ASCII
    }
    $site = Join-Path $py "Lib\site-packages"
    $pth = (Join-Path $Out "app") + "`r`n" + (Join-Path $Out "shared") + "`r`n"
    [IO.File]::WriteAllText((Join-Path $site "foodopt_stage.pth"), $pth)
    $shared = Join-Path $repo "desktop\shared"
    Mirror (Join-Path $shared "launcher") (Join-Path $Out "shared\launcher")
    Mirror (Join-Path $shared "pages") (Join-Path $Out "shared\pages")
    Copy-Item -LiteralPath (Join-Path $shared "messages.json") -Destination (Join-Path $Out "shared\messages.json") -Force
    $appDir = Join-Path $Out "app"
    if (Test-Path -LiteralPath $appDir) { Remove-Item -LiteralPath $appDir -Recurse -Force }
    foreach ($line in Get-Content -LiteralPath (Join-Path $shared "bundle_manifest.txt")) {
        $rel = $line.Trim()
        if (-not $rel -or $rel.StartsWith("#")) { continue }
        $dest = Join-Path $appDir $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        Copy-Item -LiteralPath (Join-Path $repo $rel) -Destination $dest -Force
    }
    Copy-Item -LiteralPath (Join-Path $repo "desktop\windows\start_here.txt") -Destination (Join-Path $Out "Start Here.txt") -Force
}

robocopy $Publish $Out /E /IS /IT /NFL /NDL /NJH /NJS /NP /R:2 /W:1 | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy of the shell failed ($LASTEXITCODE)" }

$appPy = Join-Path $Out "app\app.py"
$real = Join-Path $Out "app\app.py.real"
if ($App -eq "probe") {
    if (-not (Test-Path -LiteralPath $real)) { Copy-Item -LiteralPath $appPy -Destination $real }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "probe_app.py") -Destination $appPy -Force
} elseif (Test-Path -LiteralPath $real) {
    Copy-Item -LiteralPath $real -Destination $appPy -Force
    Remove-Item -LiteralPath $real -Force
}
Write-Output "staged $Out (image=$mode, app=$App)"
