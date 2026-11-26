"""Ejecutar los pesos exportados del caso D sin instalar PyTorch en Vercel."""

import hashlib
import json
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort
from PIL import Image

from .constants import CLASS_NAMES
from .face_detection import FaceDetector
from .inference_core import analyze_faces


def preprocess(image):
    """Reproducir Resize(256), CenterCrop(224), ToTensor y Normalize de DACL."""
    image = image.convert("RGB")
    width, height = image.size
    size = (256, int(256 * height / width)) if width <= height else (int(256 * width / height), 256)
    image = image.resize(size, Image.Resampling.BILINEAR)
    left, top = round((size[0] - 224) / 2), round((size[1] - 224) / 2)
    image = image.crop((left, top, left + 224, top + 224))
    tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    mean = np.array([0.5752, 0.4495, 0.4012], dtype=np.float32)[:, None, None]
    std = np.array([0.2086, 0.1911, 0.1827], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(((tensor - mean) / std)[None])


class OnnxEmotionService:
    def __init__(self, model_dir=None):
        # 1. Verificar el modelo empaquetado y cargar una sesión CPU reutilizable.
        directory = Path(model_dir) if model_dir else Path(__file__).resolve().parents[1] / "models"
        config = json.loads((directory / "modelo_onnx.json").read_text(encoding="utf-8"))
        model = directory / "modelo_d.onnx"
        if hashlib.sha256(model.read_bytes()).hexdigest() != config["sha256"]:
            raise RuntimeError("El modelo ONNX no coincide con su registro.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.detector = FaceDetector()
        self.lock = Lock()

    def predict(self, image):
        # 2. Inferencia y softmax estable; se conserva el orden RAF-DB de las clases.
        logits = self.session.run(None, {"images": preprocess(image)})[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        label = int(probabilities.argmax())
        return {"label": label, "class_name": CLASS_NAMES[label],
                "probabilities": dict(zip(CLASS_NAMES, probabilities.tolist()))}

    def analyze(self, image, already_cropped=False):
        # 3. Compartir exactamente la detección y los recortes de la aplicación local.
        return analyze_faces(image, self.detector, self.predict, self.lock,
                             already_cropped=already_cropped)
