# Reconocimiento de expresiones faciales

Aplicación web local para subir una fotografía, localizar rostros y estimar una
de siete expresiones: sorpresa, miedo, asco, felicidad, tristeza, enojo y neutral.
Utiliza los pesos ya entrenados del caso **D: ResNet18 con suavizado y pesos por clase**.

## Abrir y probar

En un equipo nuevo instala **Python 3.12 de 64 bits** y Git. Abre PowerShell:

```powershell
git clone https://github.com/sebastianfuentesp-ship-it/Reconocimiento-de-rostros.git
cd Reconocimiento-de-rostros
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe ejecutar.py
```

La instalación usa CPU por defecto y no requiere GPU. Para instalar CUDA en un
equipo compatible, ejecuta `scripts/setup.ps1 -Gpu`. Se incluye el checkpoint D:
no hace falta descargar RAF-DB, entrenar ni tener acceso a la computadora original.

Abre esta carpeta en Visual Studio Code y ejecuta en su terminal:

```powershell
.\.venv\Scripts\python.exe ejecutar.py
```

Se abre **http://127.0.0.1:5000**. También puedes pulsar **F5** y elegir
«Abrir aplicación de fotografías». Para cerrar, pulsa **Ctrl+C** en la terminal.
El inicio de la aplicación no entrena ni modifica los pesos.

1. Selecciona o arrastra una imagen JPG, PNG o WEBP, de hasta 12 MB y 24 megapíxeles.
2. Usa una foto cercana, frontal y bien iluminada. Pulsa «Analizar expresión».
3. Revisa la expresión estimada y las puntuaciones de las siete clases.
4. Si aparecen varias caras, selecciona cada rostro para ver su resultado.
   Se analizan como máximo los cinco rostros más grandes.
5. Si la imagen ya contiene un rostro recortado, marca la casilla correspondiente.

La primera predicción tarda más porque carga el checkpoint. Por defecto se usa
CPU; la GPU queda disponible para otras tareas. Opciones:

```powershell
.\.venv\Scripts\python.exe ejecutar.py --device cuda --port 5001
.\.venv\Scripts\python.exe ejecutar.py --no-browser
```

Las fotografías se procesan en memoria en este equipo; la aplicación no las
guarda ni las envía a un servicio externo. El servidor escucha únicamente en
`127.0.0.1`. El detector frontal puede fallar con perfiles, oclusiones o caras
muy pequeñas. El recorte automático no reproduce la alineación de RAF-DB;
las métricas de ese dataset no garantizan el mismo rendimiento en fotos propias.
Las puntuaciones softmax no representan certeza calibrada ni describen el estado
emocional interno de una persona.

## Organización del código

Los módulos principales contienen comentarios en español con el flujo paso a paso.

| Archivo o carpeta | Responsabilidad |
| --- | --- |
| `ejecutar.py` | Arrancar y cerrar la aplicación local. |
| `app/web.py` | Validar la carga HTTP y devolver la predicción. |
| `app/templates/` y `app/static/` | Pantalla, estilos y comportamiento de la interfaz. |
| `src/emotion_recognition/inference.py` | Coordinar detección, recorte y clasificación, sin depender de Flask. |
| `src/emotion_recognition/face_detection.py` | Localizar caras mediante el detector Haar de OpenCV. |
| `src/emotion_recognition/predict.py` | Aplicar el preprocesamiento del checkpoint y obtener puntuaciones. |
| `src/emotion_recognition/model.py` | Definir y cargar ResNet18 y su clasificador de siete salidas. |
| `src/emotion_recognition/prepare.py` | Auditar imágenes y generar las particiones reproducibles. |
| `src/emotion_recognition/train_dacl_baseline.py` | Entrenar A, B, C y D cuando se ejecutan explícitamente. |
| `src/emotion_recognition/early_stopping.py` | Contar épocas sin mejorar la pérdida de validación. |
| `src/emotion_recognition/evaluate.py` | Evaluar un checkpoint de forma separada. |
| `config/modelo_aplicacion.json` | Fijar D y verificar su SHA256 antes de cargarlo. |
| `artifacts/` y `documentos/` | Conservar checkpoints, curvas, predicciones y resultados finales. |

Flujo de una fotografía:
`interfaz → web.py → inference.py → detección/recorte → predict.py → ResNet18`.
Cambiar la interfaz no exige cambiar el entrenamiento. Los módulos auxiliares de
entrenamiento conservados en `src/` permiten mantener sus dependencias y pruebas.

## Datos y resultados conservados

RAF-DB, siete clases. Manifiesto activo: `data/processed_exact/manifest.csv`.
Se excluyeron únicamente **tres duplicados exactos**; las imágenes distintas pero
similares se conservaron. Los originales y los pesos de preentrenamiento permanecen
en `data/`. División: **9812 entrenamiento / 2457 validación / 3067 test**.
Entrenamiento y validación se dividieron de forma estratificada con semilla 42.

En cada variante se seleccionó el checkpoint con **menor `val_loss`** y se aplicó
parada temprana tras ocho épocas sin disminuirla. La validación empleó entropía
cruzada sin pesos ni suavizado en todos los casos. Los resultados finales son:

| Caso | Cambio en la pérdida de entrenamiento | Época elegida | Accuracy val | F1 macro val | Recall macro val |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Baseline, entropía cruzada | 4 | 83.44 % | 0.745876 | 73.23 % |
| B | Suavizado de etiquetas 0.05 | 7 | 85.14 % | 0.773550 | 75.20 % |
| C | Pesos por clase `N / (7 × n_k)` | 7 | 83.27 % | 0.764196 | 77.40 % |
| D | Suavizado 0.05 y pesos por clase | 28 | 85.80 % | 0.782546 | 79.61 % |

B y D ya se evaluaron en el mismo test depurado de 3067 imágenes:

| Caso | Accuracy test | F1 macro test | Recall macro test | CE test |
| --- | ---: | ---: | ---: | ---: |
| B | 83.86 % | 0.754298 | 73.82 % | 0.473949 |
| D | 84.68 % | 0.767331 | 78.47 % | 0.564463 |

Se fijó D para la aplicación por el mayor F1 macro observado entre esos dos casos.
Su ventaja no es concluyente: el intervalo bootstrap del cambio de F1 incluye cero
(`[-0.007333, 0.034022]`). D mantiene una brecha importante entre entrenamiento y
validación; no se afirma que desapareció el sobreajuste. El test ya fue consultado
y no se usa para seguir ajustando hiperparámetros.

Los detalles por clase, pérdidas y procedencia están en
`documentos/Analisis_ABCD_seleccion_por_perdida.txt` y
`documentos/Comparacion_final_test_B_D.txt`, con sus correspondientes JSON.
`scripts/comparar_test_B_D.py` conserva el cálculo de la comparación a partir de
predicciones ya guardadas, sin ejecutar los modelos.
Los cuatro directorios finales bajo `artifacts/` conservan íntegros sus resultados.
Los archivos obsoletos se apartaron en un archivo externo; los inventarios
`documentos/Archivo_*.json` indican su ubicación.

## Reproducir el entrenamiento, solo si se necesita

La aplicación carga D desde `config/modelo_aplicacion.json`, independientemente
de cualquier entrenamiento posterior. Las recetas reproducibles son:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\probar_baseline_dacl.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\probar_suavizado_etiquetas.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\probar_pesos_clase.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\probar_suavizado_pesos.ps1
```

Cada ejecución crea una carpeta nueva. Comparten ResNet18 con transfer learning
desde pesos faciales MS-Celeb enlazados por los autores de DACL, ajuste de todas las
capas, aumentos aleatorios solo en entrenamiento, batch 64, SGD con momentum 0.9,
lr 0.01, regularización L2 de 0.0005, StepLR cada 20 épocas con factor 0.1 y máximo
60 épocas. **D combina pérdidas; no añade el módulo de atención DACL.**

No hace falta preparar de nuevo los datos para usar la aplicación. El comando de
preparación, para una reproducción explícita, es:

```powershell
.\.venv\Scripts\python.exe -m emotion_recognition prepare --data-root data/raw/rafdb --output-dir data/processed_exact --near-policy exact --seed 42 --val-fraction 0.2
```

## Entorno y comprobaciones

El entorno `.venv` es local y no se publica. `scripts/setup.ps1` crea uno nuevo e
instala PyTorch CPU y este proyecto. `requirements-lock.txt` registra el entorno
original con CUDA en Windows; usa el instalador para otro equipo.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Las pruebas de la aplicación verifican cargas válidas, errores, límites, detección,
recortes y comprobación del checkpoint, sin volver a entrenar.

## Desplegar manualmente en Vercel

1. En Vercel, importa este repositorio desde GitHub.
2. Selecciona **Root Directory: `deploy/vercel`**.
3. Selecciona el preset **Flask** si no se detecta automáticamente. Conserva
   los comandos predeterminados y no configures una carpeta de salida estática.
4. Pulsa **Deploy**. No requiere variables de entorno, una API externa ni GPU.

Esa carpeta incluye la interfaz y el mismo modelo D exportado a **ONNX float32**,
con ONNX Runtime en CPU. No instala PyTorch ni descarga los pesos al recibir una
foto. El límite de carga allí es **4 MB** para respetar el límite de peticiones
de Vercel. La aplicación local conserva su límite de 12 MB.

La exportación se comprobó sobre seis imágenes sintéticas de distintos tamaños:
preprocesamiento equivalente, mismas clases predichas y diferencia máxima de
puntuaciones `2.384185791015625e-7`. Esto verifica la conversión; no constituye una
nueva evaluación del test. Registro: `deploy/vercel/models/verificacion_exportacion.json`.

Para actualizar el paquete después de cambiar la interfaz o los pesos:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[export]"
.\.venv\Scripts\python.exe scripts/preparar_vercel.py
```

El paquete se genera a partir de `app/` y de los módulos compartidos en `src/`.
No edites sus copias generadas directamente. La configuración de Vercel está en
`deploy/vercel/vercel.json`. Documentación oficial:
[Python en Vercel](https://vercel.com/docs/functions/runtimes/python) y
[límites de funciones](https://vercel.com/docs/functions/limitations).
