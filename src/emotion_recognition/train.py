"""Utilidades compartidas de datos, semillas y actualización por lotes.

Las recetas finales A/B/C/D importan read_manifest, ManifestDataset y _run_epoch
desde este módulo. Su coordinación está en train_dacl_baseline.py.
El entrenador antiguo de dos etapas se conserva para compatibilidad.
"""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .constants import CLASS_NAMES, NUM_CLASSES
from .model import build_model, build_transforms, resolve_device


def _safe_image_path(data_root: Path, relative_path: str) -> Path:
    path = (data_root / relative_path).resolve()
    if not path.is_relative_to(data_root):
        raise ValueError(f"Ruta fuera del directorio de datos: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Imagen del manifest inexistente: {path}")
    return path


def read_manifest(manifest: str | Path) -> list[dict[str, Any]]:
    """Read labels and check split separation before touching image tensors."""
    with Path(manifest).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_path", "label", "class_name", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requiere columnas: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("El manifest está vacío.")
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        label = int(row["label"])
        split = row["split"]
        if label not in range(NUM_CLASSES) or row["class_name"] != CLASS_NAMES[label]:
            raise ValueError(f"Etiqueta/nombre de clase inconsistentes: {row['image_path']}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Partición desconocida: {split}")
        row["label"] = label
        # Pixel hashes also catch images re-encoded into a different file format.
        for key in ("image_path", "sha256", "pixel_sha256", "group_id"):
            value = row.get(key)
            if value:
                identity = (key, value)
                previous = seen.get(identity)
                if previous is not None and previous != split:
                    raise ValueError(f"Fuga de datos entre {previous} y {split}: {row['image_path']}")
                seen[identity] = split
    return rows


class ManifestDataset(Dataset):
    """Leer cada imagen y etiqueta del manifiesto, aplicando su transformación."""

    def __init__(self, rows: list[dict[str, Any]], data_root: Path, transform: Any):
        self.rows = rows
        self.paths = [_safe_image_path(data_root, row["image_path"]) for row in rows]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # Abrir en RGB y aplicar los aumentos al leerla: no se crean copias en disco.
        # Si cambia un original y falla su lectura, se informa en vez de omitirlo.
        with Image.open(self.paths[index]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = self.transform(image)
        return image, self.rows[index]["label"]


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _limit_training_rows(rows: list[dict[str, Any]], limit: int | None, seed: int):
    if limit is None or limit >= len(rows):
        return rows
    if limit < NUM_CLASSES:
        raise ValueError(f"--max-train-samples debe ser al menos {NUM_CLASSES}.")
    rng = random.Random(seed)
    by_label = {label: [row for row in rows if row["label"] == label] for label in range(NUM_CLASSES)}
    # Keep at least one example per class, then sample the remaining pool.
    mandatory = []
    remaining = []
    for label, candidates in by_label.items():
        rng.shuffle(candidates)
        if not candidates:
            raise ValueError(f"Falta la clase {CLASS_NAMES[label]} en entrenamiento.")
        mandatory.append(candidates[0])
        remaining.extend(candidates[1:])
    selected = mandatory + rng.sample(remaining, limit - NUM_CLASSES)
    rng.shuffle(selected)
    return selected


def _configure_phase(
    model: nn.Module, fine_tune: bool, lr: float, backbone_lr: float,
    weight_decay: float = 1e-4, fine_tune_from: str = "layer4", layer3_lr: float = 1e-5,
):
    if fine_tune_from not in {"layer3", "layer4"}:
        raise ValueError("fine-tune-from debe ser layer3 o layer4.")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    groups = [{"params": model.fc.parameters(), "lr": lr}]
    if fine_tune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        groups.append({"params": model.layer4.parameters(), "lr": backbone_lr})
        if fine_tune_from == "layer3":
            for parameter in model.layer3.parameters():
                parameter.requires_grad = True
            groups.append({"params": model.layer3.parameters(), "lr": layer3_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _set_training_mode(model, training, fine_tune=False, fine_tune_from="layer4"):
    """Update BatchNorm only in the stages selected for training."""
    model.eval()
    if training:
        if fine_tune and fine_tune_from == "all":
            model.train()
            return
        model.fc.train()
        if fine_tune:
            model.layer4.train()
            if fine_tune_from == "layer3":
                model.layer3.train()


def _run_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
    optimizer: Any = None, scaler: Any = None, fine_tune: bool = False,
    fine_tune_from: str = "layer4",
    gradient_clip: float | None = 5.0, amp: bool = True,
) -> dict[str, float]:
    """Recorrer una vez los lotes; solo train calcula gradientes y cambia pesos."""
    training = optimizer is not None
    _set_training_mode(model, training, fine_tune, fine_tune_from)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    with torch.set_grad_enabled(training):
        for batch_number, (images, labels) in enumerate(loader, 1):
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                # 1. Borrar gradientes del lote anterior antes de acumular los nuevos.
                optimizer.zero_grad(set_to_none=True)
            # Validación usa float32 igual que evaluate/predict; AMP solo entrena.
            with torch.autocast(device_type=device.type,
                                enabled=device.type == "cuda" and training and amp):
                # 2. Propagación hacia delante y pérdida multiclase de este lote.
                logits = model(images)
                losses = criterion(logits, labels)
                loss = losses.mean()
            if training:
                # 3. Retropropagar: derivar la pérdida respecto de los parámetros.
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], gradient_clip,
                    )
                # 4. Actualizar una vez por lote con el optimizador recibido (SGD
                #    en A/B/C/D). En validación se omite todo este bloque.
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(losses.detach().float().sum().cpu())
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
            if batch_number % 50 == 0 or batch_number == len(loader):
                phase_name = "Entrenamiento" if training else "Validación"
                print(f"  {phase_name}: lote {batch_number}/{len(loader)}", flush=True)
    return {
        "loss": total_loss / len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)),
    }


def _make_scheduler(optimizer, args, fine_tune):
    """Start plateau tracking only after the optimizer changes to fine tuning."""
    if args.scheduler == "none" or not fine_tune:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.scheduler_factor,
        patience=args.scheduler_patience, threshold=args.scheduler_threshold,
        threshold_mode="abs", min_lr=args.min_lr,
    )


def train(args: argparse.Namespace) -> int:
    if args.epochs < 1 or args.warmup_epochs < 0 or args.patience < 1:
        raise ValueError("epochs/patience deben ser positivos y warmup-epochs no negativo.")
    if args.batch_size < 2 or args.num_workers < 0 or args.lr <= 0 or args.backbone_lr <= 0:
        raise ValueError("Usa batch-size >= 2, num-workers >= 0 y tasas de aprendizaje positivas.")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay debe ser un número finito no negativo.")
    if args.pretrained_source == "msceleb" and args.no_pretrained:
        raise ValueError("No combines --pretrained-source msceleb con --no-pretrained.")
    if args.pretrained_weights is not None and args.pretrained_source != "msceleb":
        raise ValueError("--pretrained-weights requiere --pretrained-source msceleb.")
    if (args.fine_tune_from == "layer3"
            and (not np.isfinite(args.layer3_lr) or not 0 < args.layer3_lr <= args.backbone_lr)):
        raise ValueError("layer3-lr debe ser positiva, finita y no mayor que backbone-lr.")
    if (not 0 < args.scheduler_factor < 1 or args.scheduler_patience < 0
            or args.scheduler_threshold < 0 or not 0 <= args.min_lr <= min(args.lr, args.backbone_lr)):
        raise ValueError("Scheduler: factor entre 0 y 1, paciencia/umbral no negativos y min-lr <= tasas iniciales.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("best.pt", "history.csv", "config.json")) and not args.overwrite:
        raise FileExistsError("Ya existe una ejecución en output-dir. Usa otra carpeta o --overwrite.")
    manifest_path = Path(args.manifest).resolve()
    data_root = Path(args.data_root).resolve()
    rows = read_manifest(manifest_path)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("Se requieren particiones train y val no vacías.")
    original_train_count = len(train_rows)
    train_rows = _limit_training_rows(train_rows, args.max_train_samples, args.seed)
    if (
        args.image_size == 32
        and not args.freeze_backbone
        and args.epochs > args.warmup_epochs
        and len(train_rows) % args.batch_size == 1
    ):
        raise ValueError(
            "Con image-size 32, el último lote de una imagen no permite ajustar "
            "BatchNorm de layer4. Usa --image-size 64 o un batch-size sin lote final unitario."
        )
    counts = Counter(row["label"] for row in train_rows)
    if any(counts[label] == 0 for label in range(NUM_CLASSES)):
        raise ValueError("Entrenamiento debe contener las siete clases.")
    missing_val = [CLASS_NAMES[label] for label in range(NUM_CLASSES) if not any(row["label"] == label for row in val_rows)]
    if missing_val:
        raise ValueError(f"Validación debe contener las siete clases; faltan: {missing_val}")
    _seed_everything(args.seed)
    device = resolve_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    train_data = ManifestDataset(train_rows, data_root, build_transforms(train=True, image_size=args.image_size))
    val_data = ManifestDataset(val_rows, data_root, build_transforms(train=False, image_size=args.image_size))
    loader_options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                      "pin_memory": device.type == "cuda", "worker_init_fn": _seed_worker}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    model = build_model(
        pretrained=not args.no_pretrained, pretrained_source=args.pretrained_source,
        pretrained_weights=args.pretrained_weights,
    ).to(device)
    class_weights = [len(train_rows) / (NUM_CLASSES * counts[label]) for label in range(NUM_CLASSES)]
    weights = None if args.no_class_weights else torch.tensor(class_weights, dtype=torch.float32, device=device)
    train_criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
    val_criterion = nn.CrossEntropyLoss(reduction="none")
    config = {key: value for key, value in vars(args).items() if key != "func"}
    config.update({
        "architecture": "resnet18", "class_names": list(CLASS_NAMES),
        "initialization": model.initialization,
        "normalization": {"name": "imagenet", "mean": [0.485, 0.456, 0.406],
                          "std": [0.229, 0.224, 0.225]},
        "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "data_root": str(data_root), "device_used": str(device),
        "class_counts_train": {CLASS_NAMES[label]: counts[label] for label in range(NUM_CLASSES)},
        "class_weights": None if args.no_class_weights else class_weights,
        "train_samples": len(train_rows), "train_samples_before_limit": original_train_count,
        "val_samples": len(val_rows), "selection_metric": "val_macro_f1",
        "torch_version": str(torch.__version__),
        "fine_tune_stages": ([] if args.freeze_backbone else
                             (["layer3", "layer4"] if args.fine_tune_from == "layer3" else ["layer4"])),
        "batchnorm_policy": "Solo se actualizan estadísticas en las etapas descongeladas durante train; val siempre eval.",
        "warning": "Ejecución reducida para comprobación; no es un resultado final." if args.max_train_samples is not None else None,
    })
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -1.0
    no_improvement = 0
    previous_phase = None
    optimizer = None
    scheduler = None
    history_fields = ["epoch", "phase", "train_loss", "train_accuracy", "train_macro_f1", "val_loss", "val_accuracy", "val_macro_f1", "lr_head", "lr_layer4", "next_lr_head", "next_lr_layer4", "lr_reduced", "lr_layer3", "next_lr_layer3"]
    print(f"Dispositivo: {device}; train={len(train_rows)}; val={len(val_rows)}; test reservado.")
    print(f"Inicialización: {model.initialization['source']}; "
          f"dataset: {model.initialization['dataset']}", flush=True)
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as history_stream:
        writer = csv.DictWriter(history_stream, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            fine_tune = not args.freeze_backbone and epoch > args.warmup_epochs
            phase = ("fine_tune_layer3_layer4" if args.fine_tune_from == "layer3" else "fine_tune_layer4") if fine_tune else "head_only"
            if phase != previous_phase:
                optimizer = _configure_phase(
                    model, fine_tune, args.lr, args.backbone_lr, weight_decay=args.weight_decay,
                    fine_tune_from=args.fine_tune_from, layer3_lr=args.layer3_lr,
                )
                scheduler = _make_scheduler(optimizer, args, fine_tune)
                previous_phase = phase
                no_improvement = 0
                print(f"Fase: {phase}; tasas iniciales por grupo: "
                      f"{[group['lr'] for group in optimizer.param_groups]}", flush=True)
            used_rates = [group["lr"] for group in optimizer.param_groups]
            train_metrics = _run_epoch(
                model, train_loader, train_criterion, device, optimizer, scaler, fine_tune,
                fine_tune_from=args.fine_tune_from,
            )
            val_metrics = _run_epoch(model, val_loader, val_criterion, device)
            if scheduler is not None:
                scheduler.step(val_metrics["macro_f1"])
            next_rates = [group["lr"] for group in optimizer.param_groups]
            record = {"epoch": epoch, "phase": phase}
            record.update({f"train_{key}": value for key, value in train_metrics.items()})
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            reduced = any(after < before for before, after in zip(used_rates, next_rates))
            record.update({"lr_head": used_rates[0], "lr_layer4": used_rates[1] if fine_tune else "",
                           "next_lr_head": next_rates[0], "next_lr_layer4": next_rates[1] if fine_tune else "",
                           "lr_reduced": reduced,
                           "lr_layer3": used_rates[2] if len(used_rates) == 3 else "",
                           "next_lr_layer3": next_rates[2] if len(next_rates) == 3 else ""})
            writer.writerow(record)
            history_stream.flush()
            print(f"Época {epoch}/{args.epochs} [{phase}] loss={train_metrics['loss']:.4f} val_F1_macro={val_metrics['macro_f1']:.4f} val_accuracy={val_metrics['accuracy']:.4f}", flush=True)
            if reduced:
                print(f"  Scheduler redujo las tasas para la siguiente época: {next_rates}", flush=True)
            if val_metrics["macro_f1"] > best_score:
                best_score = val_metrics["macro_f1"]
                no_improvement = 0
                checkpoint = {
                    "architecture": "resnet18", "model_state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES), "image_size": args.image_size,
                    "epoch": epoch, "val_macro_f1": best_score, "config": config,
                    "learning_rates_next_epoch": next_rates,
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                }
                temporary_path = output_dir / "best.pt.tmp"
                torch.save(checkpoint, temporary_path)
                temporary_path.replace(output_dir / "best.pt")
            else:
                no_improvement += 1
            if no_improvement >= args.patience:
                print(f"Parada temprana tras {args.patience} épocas sin mejorar F1 macro de validación.")
                break
    print(f"Modelo seleccionado: {output_dir / 'best.pt'} (val F1 macro={best_score:.4f}).")
    return 0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("train", help="Entrenar ResNet18 sin utilizar el conjunto de prueba.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="artifacts/resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--fine-tune-from", choices=("layer4", "layer3"), default="layer4",
                        help="Etapas a ajustar: layer4 (referencia) o layer3 y layer4.")
    parser.add_argument("--layer3-lr", type=float, default=1e-5,
                        help="Tasa inicial para layer3; solo se usa con --fine-tune-from layer3.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Decaimiento de pesos de AdamW; referencia 0.0001, prueba 0.001.")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler", choices=("none", "plateau"), default="none")
    parser.add_argument("--scheduler-factor", type=float, default=0.3)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-threshold", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--no-pretrained", action="store_true", help="Solo para pruebas sin descarga o experimentos desde cero.")
    parser.add_argument("--pretrained-source", choices=("imagenet", "msceleb"), default="imagenet",
                        help="Pesos iniciales: ImageNet o reconocimiento facial MSCeleb (DAN).")
    parser.add_argument("--pretrained-weights", default=None,
                        help="Archivo MSCeleb verificado; por defecto data/pretrained/resnet18_msceleb.pth.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Entrenar únicamente la capa final.")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limitar train para una prueba técnica; no usar para el reporte final.")
    parser.add_argument("--overwrite", action="store_true", help="Reemplazar salidas de entrenamiento existentes.")
    parser.set_defaults(func=train)
    return parser
