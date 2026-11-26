param([switch]$CpuOnly, [switch]$Gpu)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno virtual.' }
}
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
# Para probar fotografías no hace falta una GPU. CUDA se instala solo con -Gpu.
$wheelIndex = if ($Gpu -and -not $CpuOnly) { 'https://download.pytorch.org/whl/cu130' } else { 'https://download.pytorch.org/whl/cpu' }
& $projectPython -m pip install 'torch==2.10.0' 'torchvision==0.25.0' --index-url $wheelIndex
if ($LASTEXITCODE -ne 0) { throw 'No se pudo instalar PyTorch. Revisa Python, red y espacio disponible.' }
& $projectPython -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias del proyecto.' }
& $projectPython -m emotion_recognition doctor
if ($LASTEXITCODE -ne 0) { throw 'La verificación del entorno falló.' }
