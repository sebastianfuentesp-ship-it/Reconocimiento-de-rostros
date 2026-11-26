# ResNet18 + suavizado; comparar con un baseline del mismo manifiesto.
param(
    [ValidateRange(1, 1000)][int]$Epochs = 60,
    [ValidateRange(2, 1024)][int]$BatchSize = 64,
    [ValidateRange(0.000001, 0.999999)][double]$LabelSmoothing = 0.05,
    [switch]$Amp,
    [ValidateRange(0, 1000)][int]$Patience = 8,
    [ValidateRange(0, 1000)][int]$MinEpochs = 0,
    [switch]$SoloPreparar
)
# PASO 1. Seleccionar el entorno del proyecto y verificar los pesos iniciales.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$projectPython = '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Falta el Python de .venv.' }
$env:OMP_NUM_THREADS = '4'
$env:PYTHONIOENCODING = 'utf-8'
& $projectPython -u -m emotion_recognition pretrained-dacl --output-dir data/pretrained
if ($LASTEXITCODE -ne 0) { throw 'Fallo la verificacion de los pesos base.' }
if ($SoloPreparar) {
    Write-Output 'Pesos verificados; no se inicio entrenamiento.'
    return
}

# PASO 2. Crear una salida nueva; los ensayos anteriores se conservan.
$runLabel = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$runOutput = Join-Path 'artifacts' "resnet18_dacl_suavizado_$runLabel"
Write-Output 'Datos: data/processed_exact/manifest.csv.'
Write-Output "Nueva prueba desde pesos base MSCeleb: label smoothing=$LabelSmoothing solo en train."
Write-Output 'Se conserva la receta ResNet18 completa + SGD; compara con un baseline entrenado con este mismo manifiesto.'
Write-Output 'Val usa CE sin suavizado. Compara val_loss y F1 macro; subir train_loss no prueba menor sobreajuste.'
$smoothingArgument = $LabelSmoothing.ToString([System.Globalization.CultureInfo]::InvariantCulture)
# PASO 3. Definir la receta. Solo esta llamada explicita inicia entrenamiento.
$trainingArguments = @(
    '-u', '-m', 'emotion_recognition', 'train-dacl-baseline',
    '--manifest', 'data/processed_exact/manifest.csv',
    '--data-root', 'data/raw/rafdb', '--output-dir', $runOutput,
    '--pretrained-weights', 'data/pretrained/resnet18_msceleb_dacl.pth',
    '--device', 'cuda', '--epochs', $Epochs, '--batch-size', $BatchSize,
    '--seed', 42, '--lr', 0.01, '--momentum', 0.9,
    '--weight-decay', 0.0005, '--step-size', 20, '--gamma', 0.1,
    '--label-smoothing', $smoothingArgument, '--selection-metric', 'loss', '--patience', $Patience, '--min-epochs', $MinEpochs
)
if ($Amp) { $trainingArguments += '--amp' }
& $projectPython @trainingArguments
if ($LASTEXITCODE -ne 0) { throw "El entrenamiento fallo. Salida: $runOutput" }
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
