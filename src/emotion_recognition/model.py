"""Arquitectura y preprocesamiento comunes a entrenamiento e inferencia.

1. Construir ResNet18 y cargar su inicialización cuando se va a entrenar.
2. Cambiar la cabeza por siete salidas, conservando el orden de las clases.
3. Aplicar aumentos solo a train y transformaciones fijas a val/test/fotografías.
4. Para la aplicación, restaurar directamente el checkpoint, sin descargar pesos.
"""

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models, transforms

from .constants import CLASS_NAMES, NUM_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DACL_MEAN = (0.5752, 0.4495, 0.4012)
DACL_STD = (0.2086, 0.1911, 0.1827)


class RandomFiveCrop:
    """Select one corner/center crop, as in the authors' DACL training code."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        return transforms.functional.five_crop(image, self.size)[random.randint(0, 4)]


def preprocessing_config(profile: str = "imagenet", image_size: int = 224) -> dict:
    """Describe and validate preprocessing before any image/model is evaluated."""
    if profile not in {"imagenet", "dacl_baseline"}:
        raise ValueError(f"Perfil de preprocesamiento desconocido: {profile}")
    if image_size < 32 or (profile == "dacl_baseline" and image_size != 224):
        raise ValueError("image_size debe ser >=32; el perfil dacl_baseline requiere 224.")
    dacl = profile == "dacl_baseline"
    return {
        "profile": profile, "image_size": image_size,
        "resize_size": 256 if dacl else image_size,
        "evaluation_crop": "center" if dacl else None,
        "mean": list(DACL_MEAN if dacl else IMAGENET_MEAN),
        "std": list(DACL_STD if dacl else IMAGENET_STD),
    }


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Choose CUDA when available and fail clearly if it was explicitly requested."""
    if isinstance(device, torch.device):
        resolved = device
    elif device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device in {"cpu", "cuda"}:
        resolved = torch.device(device)
    else:
        raise ValueError("device debe ser auto, cpu o cuda")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. Comprueba PyTorch/CUDA o usa --device cpu.")
    return resolved


def build_model(
    num_classes: int = NUM_CLASSES, pretrained: bool = True,
    pretrained_source: str = "imagenet", pretrained_weights: str | Path | None = None,
) -> nn.Module:
    """Construir ResNet18 con una cabeza nueva de siete salidas.

    La red entrega logits. CrossEntropyLoss aplica log-softmax internamente;
    softmax solo se aplica para presentar las puntuaciones de una predicción.
    """
    if num_classes < 2:
        raise ValueError("Se requieren al menos dos clases.")
    if pretrained_source not in {"imagenet", "msceleb", "msceleb_dacl"}:
        raise ValueError("pretrained-source debe ser imagenet, msceleb o msceleb_dacl.")
    face_pretrained = pretrained_source in {"msceleb", "msceleb_dacl"}
    if face_pretrained and not pretrained:
        raise ValueError("No combines msceleb con --no-pretrained.")
    if pretrained_weights is not None and not face_pretrained:
        raise ValueError("pretrained-weights requiere --pretrained-source msceleb.")
    weights = models.ResNet18_Weights.DEFAULT if pretrained and pretrained_source == "imagenet" else None
    initialization = {
        "source": "imagenet" if weights is not None else "random",
        "dataset": "ImageNet-1K" if weights is not None else None,
        "weights": str(weights) if weights is not None else None,
    }
    if face_pretrained:
        if pretrained_source == "msceleb_dacl":
            from .dacl_pretrained import DACL_FILENAME as filename
            from .dacl_pretrained import verify_dacl_weights as verify_weights
        else:
            from .pretrained import MSCELEB_FILENAME as filename
            from .pretrained import verify_msceleb_weights as verify_weights

        path = Path(pretrained_weights or Path("data/pretrained") / filename)
        initialization = verify_weights(path)
    model = models.resnet18(weights=weights)
    # ResNet18 estándar de torchvision; no se inventan capas para la aplicación:
    # - Entrada RGB: conv 7x7, 3->64 canales, stride=2, padding=3; BatchNorm y ReLU.
    # - MaxPool 3x3, stride=2, padding=1.
    # - layer1/2/3/4: dos bloques residuales cada uno, con 64/128/256/512 canales.
    #   Cada bloque usa dos conv 3x3, padding=1. El primero de layer2/3/4 reduce
    #   tamaño con stride=2 y adapta el atajo mediante conv 1x1; los demás usan 1.
    # - Average pooling global: un valor por canal -> vector de 512 características.
    # - fc: combina esas características para producir siete logits.
    # Las convoluciones aprenden sus filtros durante el ajuste; pooling no tiene pesos.
    if face_pretrained:
        # Cargar estrictamente todos los pesos originales, incluida su cabeza.
        # Después se sustituye la cabeza; así se comprueba el backbone completo.
        payload = torch.load(initialization["path"], map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise ValueError("Pesos MSCeleb inválidos: falta state_dict.")
        model.load_state_dict(payload["state_dict"], strict=True)
        del payload
    # La representación tiene 512 características; la nueva cabeza entrega 7 logits.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.initialization = initialization
    return model


def build_transforms(
    train: bool = False, image_size: int = 224, profile: str = "imagenet",
) -> transforms.Compose:
    """Mantener el mismo perfil del checkpoint al entrenar, evaluar y subir fotos."""
    config = preprocessing_config(profile, image_size)
    if profile == "dacl_baseline":
        # 1. Escalar el lado corto a 256, como en el código de los autores.
        # 2. Train elige un recorte aleatorio; val/test/fotos usan el centro fijo.
        # 3. Solo train permite espejo horizontal; luego se convierte a tensor
        #    y se normaliza cada canal con la media/desviación del perfil DACL.
        return transforms.Compose([
            transforms.Resize(256, antialias=True),
            RandomFiveCrop(224) if train else transforms.CenterCrop(224),
            *([transforms.RandomHorizontalFlip(p=0.5)] if train else []),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"]),
        ])
    operations: list[Any] = [transforms.Resize((image_size, image_size), antialias=True)]
    if train:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8, fill=(128, 128, 128)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(operations)


def load_checkpoint(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load this project's checkpoint without downloading pretrained weights.

    ``weights_only=True`` restricts deserialization to tensors and safe primitive
    metadata. The saved class order must match the project's RAF-DB mapping.
    """
    resolved = resolve_device(device)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint inválido: falta model_state_dict.")
    names = payload.get("class_names")
    if names is None or tuple(names) != tuple(CLASS_NAMES):
        raise ValueError("El orden de clases del checkpoint no coincide con RAF-DB.")
    architecture = payload.get("architecture", "resnet18")
    if architecture not in {"resnet18", "dacl_resnet18"}:
        raise ValueError("La arquitectura del checkpoint no es ResNet18 ni DACL ResNet18.")
    image_size = int(payload.get("image_size", 224))
    profile = payload.get("preprocessing_profile", "imagenet")
    preprocessing_config(profile, image_size)
    if architecture == "dacl_resnet18":
        if profile != "dacl_baseline":
            raise ValueError("DACL completo requiere el perfil de preprocesamiento dacl_baseline.")
        from .dacl_model import build_dacl_model

        model = build_dacl_model(pretrained=False)
    else:
        model = build_model(num_classes=len(names), pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(resolved).eval()
    training_state_keys = {"model_state_dict", "optimizer_state_dict", "center_state_dict"}
    metadata = {key: value for key, value in payload.items() if key not in training_state_keys}
    metadata["image_size"] = image_size
    metadata["preprocessing_profile"] = profile
    return model, metadata
