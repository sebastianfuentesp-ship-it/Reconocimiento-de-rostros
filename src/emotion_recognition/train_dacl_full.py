"""DACL attention and sparse center loss with an independent validation split.

The training objective is CE + center_lambda * sparse_center_loss. Validation
uses raw CE and the backbone logits, so train_loss and val_loss describe
different objectives; train_ce_loss is recorded separately for comparison.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .constants import CLASS_NAMES, NUM_CLASSES
from .dacl_loss import SparseCenterLoss
from .dacl_model import build_dacl_model
from .early_stopping import EarlyStopping
from .model import build_transforms, resolve_device
from .train import ManifestDataset, _run_epoch, _seed_everything, _seed_worker, read_manifest
from .train_dacl_baseline import (
    DACL_COMMIT,
    DACL_MEAN,
    DACL_PAPER_URL,
    DACL_SOURCE_URL,
    DACL_STD,
    IMAGE_SIZE,
    PROFILE,
    _split_rows,
)

HISTORY_FIELDS = [
    "epoch", "phase", "train_loss", "train_ce_loss", "train_center_loss",
    "train_accuracy", "train_macro_f1", "val_loss", "val_accuracy", "val_macro_f1",
    "lr_head", "lr_layer4", "next_lr_head", "next_lr_layer4", "lr_reduced",
]


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0:
        raise ValueError("Usa epochs >= 1, batch-size >= 2 y num-workers >= 0.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr debe ser positiva y finita.")
    if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        raise ValueError("momentum debe ser finito y estar en [0, 1).")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay debe ser finito y no negativo.")
    if args.step_size < 1 or not math.isfinite(args.gamma) or not 0 < args.gamma < 1:
        raise ValueError("StepLR requiere step-size >= 1 y gamma finito entre 0 y 1.")
    if not math.isfinite(args.alpha) or not 0 < args.alpha <= 1:
        raise ValueError("alpha debe ser finito y estar en (0, 1].")
    if not math.isfinite(args.center_lambda) or args.center_lambda < 0:
        raise ValueError("center-lambda debe ser finito y no negativo.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed debe estar entre 0 y 2**32 - 1.")
    if args.selection_metric not in {"loss", "macro_f1", "accuracy"}:
        raise ValueError("selection-metric debe ser loss, macro_f1 o accuracy.")
    for name in ("patience", "min_epochs"):
        value = getattr(args, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("patience y min-epochs deben ser enteros no negativos.")


def _validate_train_batches(train_rows: list[dict[str, Any]], batch_size: int) -> None:
    # The attention branch contains BatchNorm1d and cannot train on N=1.
    # Reject instead of silently discarding a manifest example with drop_last.
    if len(train_rows) % batch_size == 1:
        raise ValueError(
            "El último lote de train tendría una sola imagen y la atención DACL "
            "usa BatchNorm1d. Elige otro --batch-size; no se descartan imágenes."
        )


def _configure_optimizer(model: nn.Module, args: argparse.Namespace):
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma,
    )
    return optimizer, scheduler


def _ensure_finite(**tensors: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"DACL produjo valores no finitos en {name}.")


def _run_train_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    center_criterion: SparseCenterLoss, device: torch.device, optimizer,
    center_lambda: float, alpha: float,
) -> dict[str, float]:
    """Optimize all model parameters in FP32; update centers independently."""
    model.train()
    totals = {"loss": 0.0, "ce_loss": 0.0, "center_loss": 0.0}
    targets: list[int] = []
    predictions: list[int] = []
    for batch_number, (images, labels) in enumerate(loader, 1):
        images = images.to(device, dtype=torch.float32, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        features, logits, attention = model.forward_with_attention(images)
        _ensure_finite(features=features, logits=logits, attention=attention)
        ce_loss = criterion(logits, labels)
        center_loss = center_criterion(features, attention, labels)
        loss = ce_loss + center_lambda * center_loss
        _ensure_finite(ce_loss=ce_loss, center_loss=center_loss, total_loss=loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                _ensure_finite(**{f"gradient:{name}": parameter.grad})
        optimizer.step()
        # These are the pre-SGD features/attention from this batch, detached.
        # The moving average has its own alpha; lambda only weights the loss.
        center_criterion.update_centers(
            features.detach(), attention.detach(), labels, alpha=alpha,
        )
        _ensure_finite(**dict(center_criterion.named_buffers()))
        batch_size = labels.numel()
        for name, value in (("loss", loss), ("ce_loss", ce_loss), ("center_loss", center_loss)):
            totals[name] += float(value.detach().cpu()) * batch_size
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        if batch_number % 50 == 0 or batch_number == len(loader):
            print(f"  Entrenamiento DACL: lote {batch_number}/{len(loader)}", flush=True)
    if not targets:
        raise ValueError("No hay ejemplos en el loader de entrenamiento.")
    return {
        **{name: value / len(targets) for name, value in totals.items()},
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(
            targets, predictions, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0,
        )),
    }


def _build_config(
    args: argparse.Namespace, manifest_path: Path, data_root: Path,
    train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]],
    initialization: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    counts = Counter(row["label"] for row in train_rows)
    deviations = [
        ("Se utiliza el manifest del proyecto y su validación interna; "
         "el paper usa 12271 imágenes de entrenamiento y 3068 de test oficial. "
         "El test del proyecto permanece reservado."),
        ("La semilla del proyecto es explícita y se solicitan algoritmos deterministas; "
         "el código público de DACL no activa determinismo por defecto."),
    ]
    if args.batch_size != 128:
        deviations.append(
            f"Batch size {args.batch_size}; el paper indica 128 y el código público "
            "usa 64 por defecto. El valor 32 permite comenzar con una GPU de 6 GB."
        )
    metric_description = {"loss": "menor pérdida", "macro_f1": "F1 macro", "accuracy": "accuracy"}[args.selection_metric]
    deviations.append(
        f"El checkpoint se selecciona por {metric_description} de validación interna; "
        "el código público selecciona por accuracy sobre el test oficial."
    )
    paper_values = {
        "epochs": 60, "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005,
        "step_size": 20, "gamma": 0.1, "alpha": 0.5, "center_lambda": 0.01,
    }
    for name, expected in paper_values.items():
        if getattr(args, name) != expected:
            deviations.append(f"{name}={getattr(args, name)}; valor de referencia: {expected}.")
    if args.patience > 0:
        deviations.append(
            f"Parada temprana por val_{args.selection_metric}, paciencia={args.patience}; "
            f"se empieza a contar después de {args.min_epochs} épocas. "
            "El artículo entrena 60 épocas sin esta parada temprana."
        )
    config = {key: value for key, value in vars(args).items() if key != "func"}
    config.update({
        "architecture": "dacl_resnet18", "training_method": "dacl_full",
        "class_names": list(CLASS_NAMES), "image_size": IMAGE_SIZE,
        "preprocessing_profile": PROFILE,
        "normalization": {"name": PROFILE, "mean": list(DACL_MEAN), "std": list(DACL_STD)},
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "data_root": str(data_root), "device_used": str(device),
        "pretrained_source": "msceleb_dacl", "initialization": initialization,
        "train_samples": len(train_rows), "train_samples_before_limit": len(train_rows),
        "val_samples": len(val_rows), "test_used": False,
        "class_counts_train": {CLASS_NAMES[label]: counts[label] for label in range(NUM_CLASSES)},
        "class_weighting": "none", "class_weights": None,
        "label_smoothing": 0.0, "train_label_smoothing": 0.0, "validation_label_smoothing": 0.0,
        "loss": "cross_entropy_plus_sparse_center_loss",
        "train_loss_definition": "CE media + center_lambda * sparse center loss; objetivo total.",
        "train_ce_loss_definition": "Cross entropy sin pesos ni suavizado, media por ejemplo.",
        "center_loss_definition": "sum(attention * (features - class_center)^2) / (2 * N).",
        "val_loss_definition": "Cross entropy sin pesos ni suavizado de logits del backbone.",
        "loss_comparison": "train_loss incluye centros; comparar train_ce_loss con val_loss.",
        "center_update": "Media móvil explícita con alpha, independiente de center_lambda y N.",
        "center_update_rule": "Media móvil explícita con alpha, independiente de center_lambda y N.",
        "attention_usage": "Solo entrenamiento; validación y fotos usan logits del backbone.",
        "optimizer": "SGD", "scheduler": "StepLR", "warmup_epochs": 0,
        "early_stopping": args.patience > 0,
        "early_stopping_metric": f"val_{args.selection_metric}",
        "early_stopping_mode": "min" if args.selection_metric == "loss" else "max",
        "early_stopping_patience": args.patience,
        "early_stopping_min_epochs": args.min_epochs,
        "early_stopping_rule": "Mejora estricta; paciencia solo después de min_epochs; 0 desactiva.",
        "fine_tune_from": "all",
        "fine_tune_stages": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc",
                             "attention", "attention_heads"],
        "batchnorm_policy": "Todas las estadísticas BatchNorm se actualizan en train; val usa eval.",
        "drop_last": False, "gradient_clip": None, "amp_enabled": False, "precision": "float32",
        "selection_metric": f"val_{args.selection_metric}", "torch_version": str(torch.__version__),
        "paper_url": DACL_PAPER_URL, "source_url": DACL_SOURCE_URL, "source_commit": DACL_COMMIT,
        "paper_recipe": {**paper_values, "batch_size": 128, "optimizer": "SGD",
                         "pretrained_dataset": "MS-Celeb-1M"},
        "source_code_default_batch_size": 64, "deviations": deviations,
        "warning": "Adaptación del método DACL completo al split del proyecto; "
                   "no es una reproducción exacta del protocolo del paper ni garantiza su accuracy.",
    })
    return config


def _checkpoint_payload(model, center_criterion, config, epoch, val_metrics, optimizer, scheduler):
    return {
        "architecture": "dacl_resnet18", "training_method": "dacl_full",
        "model_state_dict": model.state_dict(),
        "center_state_dict": center_criterion.state_dict(),
        "class_names": list(CLASS_NAMES), "image_size": IMAGE_SIZE,
        "preprocessing_profile": PROFILE, "config": config, "epoch": epoch,
        "val_macro_f1": val_metrics["macro_f1"], "val_accuracy": val_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "selection_metric": config["selection_metric"],
        "selection_score": val_metrics[config["selection_metric"].removeprefix("val_")],
        "learning_rates_next_epoch": [group["lr"] for group in optimizer.param_groups],
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "attention_usage": config["attention_usage"],
    }


def _inference_payload(model, checkpoint):
    """Export a plain ResNet18 that the existing photo/evaluation loader accepts."""
    excluded = {"model_state_dict", "center_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
    payload = {key: value for key, value in checkpoint.items() if key not in excluded}
    payload.update({
        "architecture": "resnet18", "training_architecture": "dacl_resnet18",
        "model_state_dict": model.backbone.state_dict(),
        "config": {**checkpoint["config"], "architecture": "resnet18",
                   "training_architecture": "dacl_resnet18"},
        "export_note": "Backbone entrenado con DACL; atención y centros se usan solo en entrenamiento.",
    })
    return payload


def _fit_epochs(
    args, model, center_criterion, train_loader, val_loader, optimizer, scheduler,
    config, output_dir: Path, device: torch.device,
) -> float:
    train_criterion = nn.CrossEntropyLoss(reduction="mean")
    val_criterion = nn.CrossEntropyLoss(reduction="none")
    best_score = math.inf if args.selection_metric == "loss" else -math.inf
    best_epoch = None
    completed_epochs = 0
    stopped_early = False
    stopper = EarlyStopping(patience=args.patience, min_epochs=args.min_epochs,
                            mode="min" if args.selection_metric == "loss" else "max")
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            used_rate = optimizer.param_groups[0]["lr"]
            train_metrics = _run_train_epoch(
                model, train_loader, train_criterion, center_criterion, device, optimizer,
                center_lambda=args.center_lambda, alpha=args.alpha,
            )
            val_metrics = _run_epoch(model, val_loader, val_criterion, device, amp=False)
            for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                if any(not math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(f"Métricas no finitas en {split}, época {epoch}.")
            scheduler.step()
            next_rate = optimizer.param_groups[0]["lr"]
            record = {"epoch": epoch, "phase": "fine_tune_all"}
            record.update({f"train_{key}": value for key, value in train_metrics.items()})
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            record.update({"lr_head": used_rate, "next_lr_head": next_rate,
                           "lr_layer4": "", "next_lr_layer4": "", "lr_reduced": next_rate < used_rate})
            writer.writerow(record)
            stream.flush()
            print(f"Época {epoch}/{args.epochs} [DACL completo]: "
                  f"train_total={train_metrics['loss']:.4f} "
                  f"train_CE={train_metrics['ce_loss']:.4f} "
                  f"train_center={train_metrics['center_loss']:.4f}; "
                  f"val_CE={val_metrics['loss']:.4f} "
                  f"val_accuracy={val_metrics['accuracy']:.4f} "
                  f"val_F1_macro={val_metrics['macro_f1']:.4f}; lr={used_rate:g}", flush=True)
            score = val_metrics[args.selection_metric]
            if (score < best_score if args.selection_metric == "loss" else score > best_score):
                best_score = score
                best_epoch = epoch
                checkpoint = _checkpoint_payload(
                    model, center_criterion, config, epoch, val_metrics, optimizer, scheduler,
                )
                torch.save(checkpoint, output_dir / "best.pt.tmp")
                torch.save(_inference_payload(model, checkpoint), output_dir / "inference.pt.tmp")
                (output_dir / "best.pt.tmp").replace(output_dir / "best.pt")
                (output_dir / "inference.pt.tmp").replace(output_dir / "inference.pt")
            completed_epochs = epoch
            if stopper.update(score, epoch):
                stopped_early = True
                print(
                    f"Parada temprana en época {epoch}: {args.patience} épocas sin mejorar "
                    f"val_{args.selection_metric}, contadas después de la época {args.min_epochs}. "
                    f"Se conserva el mejor modelo de la época {best_epoch}.",
                    flush=True,
                )
                break
    summary = {
        "requested_epochs": args.epochs, "completed_epochs": completed_epochs,
        "best_epoch": best_epoch, "best_score": best_score,
        "selection_metric": f"val_{args.selection_metric}",
        "stopped_early": stopped_early,
        "stop_reason": "early_stopping" if stopped_early else "max_epochs_reached",
        "stopped_epoch": completed_epochs if stopped_early else None,
        "patience": args.patience, "min_epochs": args.min_epochs,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return best_score


def train(args: argparse.Namespace) -> int:
    _validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    if not args.overwrite and any(
        (output_dir / name).exists() for name in ("config.json", "history.csv", "best.pt", "inference.pt")
    ):
        raise FileExistsError("Ya existe una ejecución en output-dir. Usa otra carpeta o --overwrite.")
    manifest_path = Path(args.manifest).resolve()
    data_root = Path(args.data_root).resolve()
    train_rows, val_rows = _split_rows(read_manifest(manifest_path))
    _validate_train_batches(train_rows, args.batch_size)
    _seed_everything(args.seed)
    device = resolve_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    train_data = ManifestDataset(
        train_rows, data_root, build_transforms(train=True, image_size=IMAGE_SIZE, profile=PROFILE),
    )
    val_data = ManifestDataset(
        val_rows, data_root, build_transforms(train=False, image_size=IMAGE_SIZE, profile=PROFILE),
    )
    loader_options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                      "pin_memory": device.type == "cuda", "worker_init_fn": _seed_worker}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    model = build_dacl_model(pretrained=True, pretrained_weights=args.pretrained_weights).float().to(device)
    center_criterion = SparseCenterLoss(num_classes=NUM_CLASSES, feat_dim=512).float().to(device)
    optimizer, scheduler = _configure_optimizer(model, args)
    config = _build_config(
        args, manifest_path, data_root, train_rows, val_rows, model.initialization, device,
    )
    config["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    config["attention_parameter_count"] = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"DACL completo: atención + sparse center loss; SGD en todos los parámetros; {device}.", flush=True)
    print(f"train={len(train_rows)}; val={len(val_rows)}; test reservado. "
          f"Selección: val_{args.selection_metric}; FP32 sin AMP.", flush=True)
    print("train_loss = CE + center_lambda * center_loss; val_loss = CE del backbone. "
          "El CSV guarda train_ce_loss para comparar CE.", flush=True)
    if args.patience > 0:
        print(f"Parada temprana: paciencia={args.patience}; se cuenta después de "
              f"{args.min_epochs} épocas; máximo={args.epochs}.", flush=True)
    else:
        print("Parada temprana desactivada (patience=0).", flush=True)
    best_score = _fit_epochs(
        args, model, center_criterion, train_loader, val_loader, optimizer, scheduler,
        config, output_dir, device,
    )
    print(f"Modelo seleccionado: {output_dir / 'best.pt'} "
          f"(val_{args.selection_metric}={best_score:.4f}). "
          f"Para fotos: {output_dir / 'inference.pt'}.", flush=True)
    return 0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "train-dacl-full", help="Entrenar DACL completo: ResNet18, atención y sparse center loss.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="artifacts/dacl_full")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8,
                        help="Épocas sin mejora estricta de validación; 0 desactiva la parada.")
    parser.add_argument("--min-epochs", type=int, default=0,
                        help="No acumular paciencia hasta terminar estas épocas; da margen a StepLR.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="32 para comenzar en 6 GB; paper 128 y código público 64.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--center-lambda", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-weights", default=None)
    parser.add_argument("--selection-metric", choices=("loss", "macro_f1", "accuracy"), default="loss")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(func=train)
    return parser
