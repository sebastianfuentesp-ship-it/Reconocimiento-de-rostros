"""Lógica de la aplicación, independiente de Flask y de la interfaz.

Flujo: foto RGB -> detección -> recorte -> preprocesamiento del checkpoint
-> ResNet18 en evaluación -> siete puntuaciones -> expresión estimada.
Este módulo nunca entrena ni consulta los datasets de entrenamiento/test.
"""

import hashlib
import json
from pathlib import Path
from threading import Lock

from PIL import Image

from .constants import CLASS_NAMES
from .face_detection import FaceDetector
from .inference_core import NoFaceFound, analyze_faces
from .model import load_checkpoint, resolve_device
from .predict import predict_face

PROJECT_ROOT = Path(__file__).resolve().parents[2]
__all__ = ["CLASS_NAMES", "EmotionRecognitionService", "NoFaceFound", "load_deployment"]


def load_deployment() -> dict:
    """Leer la selección fija del modelo final, sin depender del último entrenamiento."""
    path = PROJECT_ROOT / "config/modelo_aplicacion.json"
    return json.loads(path.read_text(encoding="utf-8"))


class EmotionRecognitionService:
    def __init__(self, device="cpu"):
        # 1. Verificar que usamos exactamente los pesos evaluados del caso D.
        deployment = load_deployment()
        checkpoint = PROJECT_ROOT / deployment["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError("No está disponible el checkpoint final de la aplicación.")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != deployment["sha256"]:
            raise ValueError("El checkpoint no coincide con el modelo final registrado.")

        # 2. Cargar una vez la red y reutilizarla entre fotografías.
        # CPU es suficiente para fotos y deja libre la GPU; no cambia los pesos.
        self.device = resolve_device(device)
        self.model, self.metadata = load_checkpoint(checkpoint, self.device)
        self.detector = FaceDetector()
        self._lock = Lock()

    def analyze(self, image: Image.Image, already_cropped=False) -> dict:
        return analyze_faces(
            image, self.detector,
            lambda face: predict_face(face, self.model, self.metadata, self.device),
            self._lock, already_cropped=already_cropped,
        )
