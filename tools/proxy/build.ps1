# Build the proxy onnxruntime.dll (x64, static CRT).
# Output goes to tools\runtime\onnxruntime.dll
$ErrorActionPreference = "Stop"

$vs = "C:\Program Files\Microsoft Visual Studio\18\Professional"
if (-not (Test-Path -LiteralPath "$vs\VC\Auxiliary\Build\vcvars64.bat")) {
    throw "vcvars64.bat not found; adjust `$vs in build.ps1"
}

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out  = Join-Path (Split-Path -Parent $here) "runtime"
New-Item -ItemType Directory -Path $out -Force | Out-Null

$src = Join-Path $here "proxy_ort.cpp"
$inc = Join-Path $here "include"
$def = Join-Path $here "proxy_ort.def"

Push-Location $out
try {
    $cmd = 'call "' + $vs + '\VC\Auxiliary\Build\vcvars64.bat" >nul && ' +
           'cl /nologo /std:c++17 /O2 /MT /EHsc /D_USRDLL /LD ' +
           '/I "' + $inc + '" "' + $src + '" ' +
           '/Fe:onnxruntime.dll /Foonnxruntime_proxy.obj ' +
           '/link /DEF:"' + $def + '" /MACHINE:X64'
    cmd /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "cl failed with $LASTEXITCODE" }
} finally {
    Pop-Location
}

Get-Item -LiteralPath "$out\onnxruntime.dll" | Select-Object FullName, Length, LastWriteTime