"""Entrenamiento reproducible de los cuatro casos finales A, B, C y D.

A: entropía cruzada. B: añade suavizado. C: añade pesos por clase.
D: combina suavizado y pesos; sigue siendo ResNet18 sin atención auxiliar.
La validación siempre usa CE sin pesos ni suavizado para comparar las recetas.
Se guarda la menor val_loss; la paciencia vigila ese mínimo, no la última época.
El comando solo entrena al ejecutarlo explícitamente. La aplicación no lo importa.
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
from torch import nn
from torch.utils.data import DataLoader

from .constants import CLASS_NAMES, NUM_CLASSES
from .early_stopping import EarlyStopping
from .model import build_model, build_transforms, resolve_device
from .train import ManifestDataset, _run_epoch, _seed_everything, _seed_worker, read_manifest

DACL_COMMIT = "c41e9e81de9b70a84cae3d7c19102b5095964d5b"
DACL_SOURCE_URL = f"https://github.com/amirhfarzaneh/dacl/tree/{DACL_COMMIT}"
DACL_PAPER_URL = (
    "https://openaccess.thecvf.com/content/WACV2021/papers/"
    "Farzaneh_Facial_Expression_Recognition_in_the_Wild_via_Deep_Attentive_Center_"
    "WACV_2021_paper.pdf"
)
DACL_MEAN = [0.5752, 0.4495, 0.4012]
DACL_STD = [0.2086, 0.1911, 0.1827]
IMAGE_SIZE = 224
PROFILE = "dacl_baseline"
HISTORY_FIELDS = [
    "epoch", "phase", "train_loss", "train_accuracy", "train_macro_f1",
    "val_loss", "val_accuracy", "val_macro_f1", "lr_head", "lr_layer4",
    "next_lr_head", "next_lr_layer4", "lr_reduced",
]


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0:
        raise ValueError("Usa epochs >= 1, batch-size >= 2 y num-workers >= 0.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr debe ser una tasa positiva y finita.")
    if not math.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        raise ValueError("momentum debe ser finito y estar en [0, 1).")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay debe ser finito y no negativo.")
    if not math.isfinite(args.label_smoothing) or not 0 <= args.label_smoothing < 1:
        raise ValueError("label-smoothing debe ser finito y estar en [0, 1).")
    if getattr(args, "class_weighting", "none") not in {"none", "balanced"}:
        raise ValueError("class-weighting debe ser none o balanced.")
    if args.step_size < 1 or not math.isfinite(args.gamma) or not 0 < args.gamma < 1:
        raise ValueError("StepLR requiere step-size >= 1 y gamma finito entre 0 y 1.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed debe estar entre 0 y 2**32 - 1.")
    if args.selection_metric not in {"loss", "macro_f1", "accuracy"}:
        raise ValueError("selection-metric debe ser loss, macro_f1 o accuracy.")
    for name in ("patience", "min_epochs"):
        value = getattr(args, name, 8 if name == "patience" else 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name.replace('_', '-')} debe ser un entero >= 0.")


def _split_rows(rows: list[dict[str, Any]]) -> tuple[list, list]:
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("Se requieren particiones train y val no vacías.")
    for split, selected in (("train", train_rows), ("val", val_rows)):
        present = {row["label"] for row in selected}
        missing = [CLASS_NAMES[label] for label in range(NUM_CLASSES) if label not in present]
        if missing:
            raise ValueError(f"{split} debe contener las siete clases; faltan: {missing}")
    return train_rows, val_rows


def _configure_optimizer(model: nn.Module, args: argparse.Namespace):
    """Use one SGD group for the entire backbone and the fresh classifier."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma,
    )
    return optimizer, scheduler


def _balanced_class_weights(train_rows: list[dict[str, Any]]) -> list[float]:
    """Use N/(K*n_k), in RAF class order, from training examples exclusively."""
    if not train_rows:
        raise ValueError("No hay imágenes de entrenamiento para calcular pesos de clase.")
    for row in train_rows:
        if row.get("split") != "train":
            raise ValueError("Los pesos de clase se calculan únicamente con filas de train.")
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in range(NUM_CLASSES):
            raise ValueError("Etiqueta inválida al calcular pesos de clase.")
    counts = Counter(row["label"] for row in train_rows)
    if any(counts[label] == 0 for label in range(NUM_CLASSES)):
        raise ValueError("Se requieren las siete clases en train para calcular sus pesos.")
    return [len(train_rows) / (NUM_CLASSES * counts[label]) for label in range(NUM_CLASSES)]


def _build_criteria(
    args: argparse.Namespace, class_weights: list[float] | None = None,
    device: str | torch.device = "cpu",
):
    """Weight training examples only; validation remains comparable raw NLL."""
    weight = None
    if getattr(args, "class_weighting", "none") == "balanced":
        if (class_weights is None or len(class_weights) != NUM_CLASSES
                or any(not math.isfinite(value) or value <= 0 for value in class_weights)):
            raise ValueError("Se requieren siete pesos de clase finitos y positivos.")
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    elif class_weights is not None:
        raise ValueError("No pases pesos de clase con class-weighting=none.")
    return (
        nn.CrossEntropyLoss(weight=weight, reduction="none", label_smoothing=args.label_smoothing),
        nn.CrossEntropyLoss(reduction="none", label_smoothing=0.0),
    )


def _build_config(
    args: argparse.Namespace, manifest_path: Path, data_root: Path,
    train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]],
    initialization: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    counts = Counter(row["label"] for row in train_rows)
    class_weighting = getattr(args, "class_weighting", "none")
    class_weights = _balanced_class_weights(train_rows) if class_weighting == "balanced" else None
    amp_enabled = bool(args.amp and device.type == "cuda")
    deviations = [
        ("Se usa el manifest del proyecto y su validación interna; "
         "el paper usa 12271 imágenes de entrenamiento y 3068 de test oficial. "
         "El test del proyecto permanece reservado."),
        ("Se usa ResNet18 estándar con cross entropy; no se incorpora la rama de "
         "atención ni sparse center loss del método DACL completo."),
        ("La semilla del proyecto es explícita y se solicitan algoritmos deterministas; "
         "el código público de DACL no activa determinismo por defecto."),
    ]
    if args.batch_size != 128:
        deviations.append(
            f"Batch size {args.batch_size}; el paper indica 128 y el código público "
            "de workspace.py usa 64 por defecto."
        )
    if args.selection_metric == "loss":
        deviations.append("El checkpoint se selecciona por menor pérdida de validación, no por accuracy del test.")
    elif args.selection_metric == "macro_f1":
        deviations.append(
            "El checkpoint se selecciona por F1 macro de validación interna; "
            "el código público selecciona por accuracy sobre el test oficial."
        )
    else:
        deviations.append(
            "El checkpoint se selecciona por accuracy de validación interna; "
            "el código público consulta el test oficial cada época."
        )
    paper_values = {
        "epochs": 60, "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005,
        "step_size": 20, "gamma": 0.1,
    }
    for name, expected in paper_values.items():
        if getattr(args, name) != expected:
            deviations.append(f"{name}={getattr(args, name)}; valor del paper: {expected}.")
    if amp_enabled:
        deviations.append("AMP activado por opción; el código publicado usa float32 sin AMP.")
    if args.label_smoothing > 0:
        deviations.append(
            f"Regularización adicional: label_smoothing={args.label_smoothing} solo en train; "
            "el baseline publicado no usa suavizado. Validación conserva CE sin suavizado."
        )
    if class_weights is not None:
        deviations.append(
            "Ponderación adicional de clases solo en train: N/(K*n_k), calculada con "
            "el train del manifest. El baseline publicado no usa estos pesos. "
            "Val conserva CE sin ponderación."
        )
    patience = getattr(args, "patience", 8)
    min_epochs = getattr(args, "min_epochs", 0)
    if patience:
        deviations.append(
            f"Parada temprana por val_{args.selection_metric}: paciencia {patience}, "
            f"contada solo después de {min_epochs} épocas; el paper completa 60 épocas."
        )
    config = {key: value for key, value in vars(args).items() if key != "func"}
    config.update({
        "architecture": "resnet18", "class_names": list(CLASS_NAMES),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "data_root": str(data_root), "device_used": str(device),
        "image_size": IMAGE_SIZE, "preprocessing_profile": PROFILE,
        "pretrained_source": "msceleb_dacl", "initialization": initialization,
        "normalization": {"name": PROFILE, "mean": list(DACL_MEAN), "std": list(DACL_STD)},
        "train_samples": len(train_rows), "train_samples_before_limit": len(train_rows),
        "val_samples": len(val_rows), "test_used": False,
        "class_counts_train": {CLASS_NAMES[label]: counts[label] for label in range(NUM_CLASSES)},
        "class_weights": class_weights, "loss": "cross_entropy", "optimizer": "SGD",
        "class_weighting": class_weighting,
        "class_weight_formula": "N/(K*n_k), train only" if class_weights is not None else None,
        "class_weight_names": list(CLASS_NAMES) if class_weights is not None else None,
        "train_loss_reduction": "Mean of per-example losses over the batch (no batch weight renormalization).",
        "train_label_smoothing": args.label_smoothing, "validation_label_smoothing": 0.0,
        "train_loss_definition": (
            f"Cross entropy; ponderación={class_weighting}; "
            f"label_smoothing={args.label_smoothing}; media por ejemplo."
        ),
        "val_loss_definition": "Cross entropy sin suavizado ni pesos de clase (NLL).",
        "scheduler": "StepLR", "warmup_epochs": 0, "early_stopping": patience > 0,
        "early_stopping_metric": f"val_{args.selection_metric}",
        "early_stopping_mode": "min" if args.selection_metric == "loss" else "max",
        "early_stopping_patience": patience, "early_stopping_min_epochs": min_epochs,
        "early_stopping_rule": (
            "Mayor score de validación desde época 1; empate no mejora. La paciencia "
            "solo acumula después de min_epochs; una mejora estricta reinicia el contador. "
            "patience=0 desactiva la parada temprana."
        ),
        "fine_tune_from": "all", "fine_tune_stages": ["conv1", "bn1", "layer1", "layer2",
                                                       "layer3", "layer4", "fc"],
        "batchnorm_policy": "Todas las estadísticas BatchNorm se actualizan en train; val usa eval.",
        "gradient_clip": None, "amp_enabled": amp_enabled,
        "precision": "amp" if amp_enabled else "float32",
        "selection_metric": f"val_{args.selection_metric}", "torch_version": str(torch.__version__),
        "paper_url": DACL_PAPER_URL, "source_url": DACL_SOURCE_URL,
        "source_commit": DACL_COMMIT, "paper_sections": ["4.2", "Table 1"],
        "paper_recipe": {**paper_values, "batch_size": 128, "optimizer": "SGD",
                         "pretrained_dataset": "MS-Celeb-1M"},
        "source_code_default_batch_size": 64, "deviations": deviations,
        "warning": "Adaptación del baseline softmax de DACL; no es una reproducción exacta "
                   "de su protocolo ni una garantía del resultado publicado.",
    })
    return config


def _checkpoint_payload(model, config, epoch, val_metrics, optimizer, scheduler):
    return {
        "architecture": "resnet18", "model_state_dict": model.state_dict(),
        "class_names": list(CLASS_NAMES), "image_size": IMAGE_SIZE,
        "preprocessing_profile": PROFILE, "config": config, "epoch": epoch,
        "val_macro_f1": val_metrics["macro_f1"], "val_accuracy": val_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "selection_metric": config["selection_metric"],
        "selection_score": val_metrics[config["selection_metric"].removeprefix("val_")],
        "learning_rates_next_epoch": [group["lr"] for group in optimizer.param_groups],
        "scheduler_state_dict": scheduler.state_dict(),
    }


def train(args: argparse.Namespace) -> int:
    # PASO 1. Validar parámetros y evitar sobrescribir una ejecución existente.
    _validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    if not args.overwrite and any(
        (output_dir / name).exists() for name in ("config.json", "history.csv", "best.pt")
    ):
        raise FileExistsError("Ya existe una ejecución en output-dir. Usa otra carpeta o --overwrite.")
    # PASO 2. Leer la división fija. Test se comprueba como metadato y no entra al entrenamiento.
    manifest_path = Path(args.manifest).resolve()
    data_root = Path(args.data_root).resolve()
    # read_manifest checks labels, paths/hashes and group separation, including test metadata.
    train_rows, val_rows = _split_rows(read_manifest(manifest_path))
    _seed_everything(args.seed)
    device = resolve_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    # PASO 3. Preparar lotes: train con aumentos, validación con recorte central fijo.
    train_data = ManifestDataset(
        train_rows, data_root, build_transforms(train=True, image_size=IMAGE_SIZE, profile=PROFILE),
    )
    val_data = ManifestDataset(
        val_rows, data_root, build_transforms(train=False, image_size=IMAGE_SIZE, profile=PROFILE),
    )
    loader_options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                      "pin_memory": device.type == "cuda", "worker_init_fn": _seed_worker}
    # At 224px the final ResNet BatchNorm2d sees a 7x7 map, so N=1 is valid.
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    # PASO 4. Transferencia de aprendizaje: pesos faciales previos y cabeza nueva de 7 clases.
    model = build_model(
        pretrained_source="msceleb_dacl", pretrained_weights=args.pretrained_weights,
    ).to(device)
    # PASO 5. Ajustar toda la red con SGD; StepLR cambia la tasa, no calcula gradientes.
    optimizer, scheduler = _configure_optimizer(model, args)
    # Los pesos N/(7*n_k) usan exclusivamente train. La pérdida de val nunca se pondera.
    class_weights = _balanced_class_weights(train_rows) if args.class_weighting == "balanced" else None
    train_criterion, val_criterion = _build_criteria(args, class_weights=class_weights, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    config = _build_config(
        args, manifest_path, data_root, train_rows, val_rows, model.initialization, device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Baseline DACL: ResNet18 + CE; SGD completo desde época 1; {device}.", flush=True)
    print(f"train={len(train_rows)}; val={len(val_rows)}; test reservado. "
          f"Selección: val_{args.selection_metric}; AMP={config['amp_enabled']}.", flush=True)
    print(f"Label smoothing: train={args.label_smoothing}; val=0.0. "
          "Compara val_loss y F1; train_loss cambia de objetivo si hay suavizado.", flush=True)
    print(f"Pesos por clase: {args.class_weighting}; "
          f"{dict(zip(CLASS_NAMES, class_weights)) if class_weights is not None else 'sin ponderación'}; "
          "val sin pesos.", flush=True)
    best_score = math.inf if args.selection_metric == "loss" else -math.inf
    best_epoch = None
    completed_epochs = 0
    stopper = EarlyStopping(
        patience=config["early_stopping_patience"], min_epochs=config["early_stopping_min_epochs"],
        mode="min" if args.selection_metric == "loss" else "max",
    )
    if config["early_stopping"]:
        print(f"Parada temprana: val_{args.selection_metric}; paciencia={stopper.patience}; "
              f"el contador empieza después de la época {stopper.min_epochs}.", flush=True)
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as history_stream:
        writer = csv.DictWriter(history_stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        # PASO 6. En cada época: entrenar -> validar -> registrar -> guardar el mejor -> comprobar paro.
        for epoch in range(1, args.epochs + 1):
            used_rate = optimizer.param_groups[0]["lr"]
            train_metrics = _run_epoch(
                model, train_loader, train_criterion, device, optimizer=optimizer, scaler=scaler,
                fine_tune=True, fine_tune_from="all", gradient_clip=None, amp=args.amp,
            )
            val_metrics = _run_epoch(
                model, val_loader, val_criterion, device, fine_tune_from="all",
                gradient_clip=None, amp=False,
            )
            scheduler.step()
            next_rate = optimizer.param_groups[0]["lr"]
            record = {"epoch": epoch, "phase": "fine_tune_all"}
            record.update({f"train_{key}": value for key, value in train_metrics.items()})
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            record.update({"lr_head": used_rate, "next_lr_head": next_rate,
                           "lr_layer4": "", "next_lr_layer4": "",
                           "lr_reduced": next_rate < used_rate})
            writer.writerow(record)
            history_stream.flush()
            print(f"Época {epoch}/{args.epochs} [fine_tune_all]: "
                  f"train_loss={train_metrics['loss']:.4f} "
                  f"train_accuracy={train_metrics['accuracy']:.4f} "
                  f"train_F1_macro={train_metrics['macro_f1']:.4f}; "
                  f"val_loss={val_metrics['loss']:.4f} "
                  f"val_accuracy={val_metrics['accuracy']:.4f} "
                  f"val_F1_macro={val_metrics['macro_f1']:.4f}; lr={used_rate:g}", flush=True)
            score = val_metrics[args.selection_metric]
            should_stop = stopper.update(score, epoch)
            completed_epochs = epoch
            if (score < best_score if args.selection_metric == "loss" else score > best_score):
                best_score = score
                best_epoch = epoch
                temporary = output_dir / "best.pt.tmp"
                torch.save(_checkpoint_payload(
                    model, config, epoch, val_metrics, optimizer, scheduler,
                ), temporary)
                temporary.replace(output_dir / "best.pt")
            if should_stop:
                print(f"Parada temprana en época {epoch}: {stopper.stale_epochs} épocas "
                      f"sin mejorar val_{args.selection_metric}; se conserva el mejor "
                      f"modelo de la época {best_epoch}.", flush=True)
                break
    summary = {
        "requested_epochs": args.epochs, "completed_epochs": completed_epochs,
        "best_epoch": best_epoch, "best_score": best_score,
        "selection_metric": config["selection_metric"],
        "stopped_early": stopper.stopped_epoch is not None and completed_epochs < args.epochs,
        "stop_reason": "early_stopping" if stopper.stopped_epoch is not None else "max_epochs",
        "stopped_epoch": stopper.stopped_epoch,
        "patience": stopper.patience, "min_epochs": stopper.min_epochs,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Modelo seleccionado: {output_dir / 'best.pt'} "
          f"(val_{args.selection_metric}={best_score:.4f}).", flush=True)
    return 0


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "train-dacl-baseline", help="Entrenar ResNet18 + CE con la receta del baseline DACL.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="artifacts/dacl_baseline")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8,
                        help="Épocas sin mejora después de min-epochs; 0 desactiva la parada.")
    parser.add_argument("--min-epochs", type=int, default=0,
                        help="No acumular paciencia antes de completar estas épocas.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="64 en el código público; el paper indica 128.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Suavizado solo en train; 0 mantiene el baseline, prueba adicional 0.05.")
    parser.add_argument("--class-weighting", choices=("none", "balanced"), default="none",
                        help="Pesos solo en train: balanced usa N/(7*n_k); none conserva el baseline.")
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-weights", default=None)
    parser.add_argument("--selection-metric", choices=("loss", "macro_f1", "accuracy"), default="loss")
    parser.add_argument("--amp", action="store_true",
                        help="Activar AMP solo en entrenamiento CUDA; por defecto float32.")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(func=train)
    return parser
