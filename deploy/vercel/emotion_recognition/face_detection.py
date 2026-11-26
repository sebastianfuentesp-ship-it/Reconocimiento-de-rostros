"""Paso previo a la CNN: localizar rostros y preparar sus recortes.

El detector Haar de OpenCV busca caras frontales; no clasifica emociones.
ResNet18 recibe después cada rostro, no la fotografía completa.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FaceBox:
    """Rectángulo del rostro en píxeles de la imagen ya orientada."""

    x: int
    y: int
    width: int
    height: int


class FaceDetector:
    def __init__(self):
        # El XML viene incluido en OpenCV: la aplicación funciona sin descargarlo.
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError("No se pudo cargar el detector de rostros de OpenCV.")

    def detect(self, image: Image.Image) -> list[FaceBox]:
        # 1. Limitar el tamaño de búsqueda para que una foto grande no demore demasiado.
        width, height = image.size
        scale = min(1.0, 1280 / max(width, height))
        reduced = image.resize((round(width * scale), round(height * scale)))
        gray = cv2.cvtColor(np.asarray(reduced), cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        # 2. Buscar ventanas que coincidan con un rostro frontal.
        detected = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40),
        )
        boxes = []
        for x, y, w, h in detected:
            left, top = max(0, round(x / scale)), max(0, round(y / scale))
            right = min(width, round((x + w) / scale))
            bottom = min(height, round((y + h) / scale))
            boxes.append(FaceBox(left, top, right - left, bottom - top))
        # El rostro más grande aparece primero en la interfaz.
        return sorted(boxes, key=lambda box: box.width * box.height, reverse=True)


def crop_face(image: Image.Image, box: FaceBox) -> Image.Image:
    """Recortar un cuadrado con margen, manteniéndolo dentro de la fotografía."""
    width, height = image.size
    side = min(round(max(box.width, box.height) * 1.15), width, height)
    left = max(0, min(round(box.x + box.width / 2 - side / 2), width - side))
    top = max(0, min(round(box.y + box.height / 2 - side / 2), height - side))
    return image.crop((left, top, left + side, top + side))
