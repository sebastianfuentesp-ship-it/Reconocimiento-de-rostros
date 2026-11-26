"""Clasificar un rostro recortado reutilizando el preprocesamiento del modelo.

Esta función es compartida por la consola y el servicio de la aplicación.
Detectar/localizar el rostro es una tarea previa, separada de clasificarlo.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

from .model import build_transforms, load_checkpoint, resolve_device


def predict_face(
    image: Image.Image, model: Any, metadata: dict[str, Any], device: Any
) -> dict[str, Any]:
    """Classify an already cropped face; face detection belongs to the caller."""
    # 1. Recuperar el tamaño y la normalización guardados, sin aumentos aleatorios.
    transform = build_transforms(
        train=False, image_size=metadata.get("image_size", 224),
        profile=metadata.get("preprocessing_profile", "imagenet"),
    )
    image = ImageOps.exif_transpose(image).convert("RGB")
    # 2. Añadir la dimensión de lote: una imagen pasa a tener forma [1, 3, 224, 224].
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    # 3. Desactivar gradientes y usar BatchNorm en evaluación; los pesos no cambian.
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    # 4. Softmax expresa las puntuaciones relativas; argmax elige una de las siete clases.
    label = max(range(len(probabilities)), key=probabilities.__getitem__)
    names = metadata["class_names"]
    return {
        "label": label,
        "class_name": names[label],
        "probabilities": {name: float(value) for name, value in zip(names, probabilities)},
        "note": "Expresión facial estimada a partir de un rostro recortado; los valores softmax no son confianza calibrada.",
    }


def predict(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device=device)
    with Image.open(args.image) as source:
        result = predict_face(source, model, metadata, device)
    result["image"] = str(Path(args.image).resolve())
    result["checkpoint"] = str(Path(args.checkpoint).resolve())
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("predict", help="Clasificar una foto de un rostro ya recortado.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=None, help="Guardar el resultado JSON.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.set_defaults(func=predict)
    return parser
