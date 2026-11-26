# El proyecto usa su propio entorno y puede iniciarse desde cualquier carpeta.
param([int]$Port = 5000, [switch]$SinNavegador)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$projectPython = '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Primero ejecuta scripts/setup.ps1.' }
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$arguments = @('-u', 'ejecutar.py', '--port', $Port)
if ($SinNavegador) { $arguments += '--no-browser' }
& $projectPython @arguments
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 130) { throw 'No se pudo iniciar la aplicacion. Revisa la salida anterior.' }
