# Cuarto caso: ResNet18 con suavizado y pesos por clase en la perdida de train.
param(
    [ValidateRange(1, 1000)][int]$Epochs = 60,
    [ValidateRange(2, 1024)][int]$BatchSize = 64,
    [ValidateRange(0.000001, 0.999999)][double]$LabelSmoothing = 0.05,
    [ValidateRange(0, 1000)][int]$Patience = 8,
    [ValidateRange(0, 1000)][int]$MinEpochs = 0,
    [switch]$Amp,
    [switch]$SoloPreparar
)
# PASO 1. Seleccionar el entorno del proyecto y verificar los pesos iniciales.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$projectPython = '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Falta el Python de .venv.' }
if (-not (Test-Path -LiteralPath 'data/processed_exact/manifest.csv')) { throw 'Falta el manifiesto de duplicados exactos.' }
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'

& $projectPython scripts/guardar_resultados.py
if ($LASTEXITCODE -ne 0) { throw 'No se pudo guardar la bitacora anterior.' }
& $projectPython -u -m emotion_recognition pretrained-dacl --output-dir data/pretrained
if ($LASTEXITCODE -ne 0) { throw 'Fallo la verificacion de los pesos base.' }
if ($SoloPreparar) {
    Write-Output 'Resultados registrados y pesos verificados; no se inicio entrenamiento.'
    return
}

# PASO 2. Crear una salida nueva; los ensayos anteriores se conservan.
$runLabel = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$runOutput = Join-Path 'artifacts' "resnet18_dacl_suavizado_pesos_$runLabel"
$smoothingArgument = $LabelSmoothing.ToString([System.Globalization.CultureInfo]::InvariantCulture)
Write-Output "Cuarto caso: suavizado=$smoothingArgument + pesos N/(7*n_k) calculados solo con train."
Write-Output "Datos: processed_exact; batch=$BatchSize; seleccion por menor val_loss; paciencia=$Patience."
Write-Output 'Val conserva CE sin pesos y sin suavizado. A, B y C se conservan.'
# PASO 3. Definir la receta. Solo esta llamada explicita inicia entrenamiento.
$trainingArguments = @(
    '-u', '-m', 'emotion_recognition', 'train-dacl-baseline',
    '--manifest', 'data/processed_exact/manifest.csv',
    '--data-root', 'data/raw/rafdb', '--output-dir', $runOutput,
    '--pretrained-weights', 'data/pretrained/resnet18_msceleb_dacl.pth',
    '--device', 'cuda', '--epochs', $Epochs, '--batch-size', $BatchSize,
    '--seed', 42, '--lr', 0.01, '--momentum', 0.9,
    '--weight-decay', 0.0005, '--step-size', 20, '--gamma', 0.1,
    '--class-weighting', 'balanced', '--label-smoothing', $smoothingArgument,
    '--selection-metric', 'loss', '--patience', $Patience, '--min-epochs', $MinEpochs
)
if ($Amp) { $trainingArguments += '--amp' }
& $projectPython @trainingArguments
if ($LASTEXITCODE -ne 0) { throw "Fallo el entrenamiento. Salida: $runOutput" }
# PASO 4. Evaluar el mejor checkpoint en validacion y guardar curvas/resultados.
$checkpoint = Join-Path $runOutput 'best.pt'
& $projectPython -m emotion_recognition evaluate --manifest data/processed_exact/manifest.csv --data-root data/raw/rafdb --checkpoint $checkpoint --output-dir (Join-Path $runOutput 'validation') --split val --device cuda --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw 'Fallo la evaluacion de validacion.' }
& $projectPython scripts/plot_history.py --history (Join-Path $runOutput 'history.csv') --output (Join-Path $runOutput 'training_curves.png')
if ($LASTEXITCODE -ne 0) { throw 'Fallo la generacion de curvas.' }
& $projectPython scripts/guardar_resultados.py
if ($LASTEXITCODE -ne 0) { throw 'Fallo la actualizacion de resultados.' }
Write-Output "Terminado: $runOutput. Checkpoint elegido por menor val_loss; solo se evaluo validacion."
Write-Output 'La aplicacion conserva el checkpoint final fijado en config/modelo_aplicacion.json.'
