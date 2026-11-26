# Despliegue en Vercel

En la importación del repositorio elige **Root Directory: `deploy/vercel`** y
**Framework Preset: Flask**. Conserva los comandos predeterminados. La entrada es
`index.py`, que expone `app`; no se necesita `ejecutar.py` ni una GPU.

El modelo D y su verificación SHA256 están en `models/`. El formato ONNX reduce
las dependencias del servidor y mantiene los pesos float32. Se aceptan fotos de
hasta 4 MB y 24 megapíxeles. Las fotos se procesan en memoria y no se guardan.

Esta carpeta se genera con `scripts/preparar_vercel.py` desde la raíz del
repositorio. Cambia la interfaz y la lógica en los archivos originales y vuelve
a generar el paquete. Las dependencias de despliegue están en `requirements.txt`;
el entrenamiento y PyTorch pertenecen al proyecto local.
