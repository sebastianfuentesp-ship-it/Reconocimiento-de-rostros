"""Evaluar checkpoints sin ajustar sus parámetros.

Flujo: verificar manifiesto -> preprocesar partición -> inferencia -> métricas
-> matrices de confusión y predicciones -> registrar procedencia del resultado.
La comparación final de B/D ya está guardada; abrir la app no vuelve a evaluar test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .constants import CLASS_NAMES


def export_metrics(y_true, y_pred, output_dir, class_names=CLASS_NAMES) -> dict:
    """Write all-class metrics, counts and row-normalized confusion matrices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    count = len(class_names)
    if truth.ndim != 1 or prediction.shape != truth.shape or not len(truth):
        raise ValueError("Metric inputs must be nonempty vectors with matching shapes")
    if np.any((truth < 0) | (truth >= count) | (prediction < 0) | (prediction >= count)):
        raise ValueError("Metric inputs contain an unknown class label")
    labels = list(range(count))
    matrix = confusion_matrix(truth, prediction, labels=labels)
    support = matrix.sum(axis=1)
    normalized = np.divide(
        matrix,
        support[:, None],
        out=np.zeros_like(matrix, dtype=float),
        where=support[:, None] != 0,
    )
    report = classification_report(
        truth, prediction, labels=labels, target_names=list(class_names), output_dict=True,
        zero_division=0,
    )
    metrics = {
        "samples": len(truth),
        "class_names": list(class_names),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(normalized.diagonal()[support > 0].mean()),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "macro_f1_definition": "Mean of seven class F1 scores; missing classes receive zero.",
        "balanced_accuracy_definition": "Mean recall over classes with nonzero true support.",
        "present_true_classes": [class_names[i] for i in labels if support[i] > 0],
        "per_class": {name: report[name] for name in class_names},
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for name, values, format_value in (
        ("confusion_matrix", matrix, "d"),
        ("confusion_matrix_normalized", normalized, ".2f"),
    ):
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true/predicted", *class_names])
            for label, row in zip(class_names, values):
                writer.writerow([label, *row.tolist()])
        figure, axis = plt.subplots(figsize=(9, 8))
        heatmap = axis.imshow(values, cmap="Blues", vmin=0)
        axis.set(
            xticks=range(count), yticks=range(count), xticklabels=class_names,
            yticklabels=class_names, xlabel="Predicted class", ylabel="True class",
            title="Confusion matrix" + (" (row normalized)" if "normalized" in name else ""),
        )
        plt.setp(axis.get_xticklabels(), rotation=40, ha="right")
        threshold = float(values.max()) / 2
        for row in range(count):
            for column in range(count):
                axis.text(
                    column, row, format(values[row, column], format_value), ha="center", va="center",
                    color="white" if values[row, column] > threshold else "black",
                )
        figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=160)
        plt.close(figure)
    return metrics


def export_predictions(rows, predictions, output_dir, class_names=CLASS_NAMES, probabilities=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if len(rows) != len(predictions):
        raise ValueError("Prediction count does not match manifest rows")
    fieldnames = ["image_path", "true_label", "true_class", "predicted_label", "predicted_class", "correct"]
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if probabilities.shape != (len(rows), len(class_names)):
            raise ValueError("Probability array does not match sample and class counts")
        fieldnames.extend(f"probability_{name}" for name in class_names)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, prediction) in enumerate(zip(rows, predictions)):
            true_label = int(row["label"])
            label = int(prediction)
            record = {
                "image_path": row["image_path"], "true_label": true_label,
                "true_class": class_names[true_label], "predicted_label": label,
                "predicted_class": class_names[label], "correct": int(label == true_label),
            }
            if probabilities is not None:
                record.update({f"probability_{name}": float(probabilities[index, i])
                               for i, name in enumerate(class_names)})
            writer.writerow(record)


class _EvaluationDataset:
    """Plain picklable Dataset protocol; importing this module does not import torch."""

    def __init__(self, rows, data_root, transform):
        self.rows = rows
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from .baseline import resolve_image

        row = self.rows[index]
        with Image.open(resolve_image(self.data_root, row["image_path"])) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label"])


def _verify_manifest_provenance(expected_hash, actual_hash):
    """A newly generated split must never masquerade as the original holdout."""
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            "Checkpoint has no manifest_sha256 provenance. Held-out evaluation requires "
            "a checkpoint produced by this project; use predict for external photos."
        )
    if expected_hash != actual_hash:
        raise ValueError(
            "The manifest does not match the one used to train this checkpoint "
            "(manifest_sha256 mismatch). Restore the original manifest or retrain; "
            "regenerating splits can move training images into the held-out set. "
            "Use predict for external photos."
        )


def _predict_resnet(checkpoint, rows, args, manifest_sha256, model_info=None):
    import torch
    from torch.utils.data import DataLoader

    from .model import build_transforms, load_checkpoint, resolve_device

    device = resolve_device(args.device)
    model, metadata = load_checkpoint(checkpoint, device=device)
    if model_info is not None:
        model_info.update({
            "architecture": metadata.get("architecture", "resnet18"),
            "training_method": metadata.get("training_method",
                                             metadata.get("config", {}).get("training_method")),
        })
    _verify_manifest_provenance(
        metadata.get("config", {}).get("manifest_sha256"), manifest_sha256
    )
    if tuple(metadata.get("class_names", ())) != tuple(CLASS_NAMES):
        raise ValueError("Checkpoint class mapping does not match this project")
    dataset = _EvaluationDataset(
        rows, args.data_root, build_transforms(
            train=False, image_size=metadata.get("image_size", 224),
            profile=metadata.get("preprocessing_profile", "imagenet"),
        )
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    batches = []
    model.eval()
    with torch.inference_mode():
        for inputs, _ in loader:
            logits = model(inputs.to(device, non_blocking=device.type == "cuda"))
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(batches)
    return probabilities.argmax(axis=1), probabilities


def run(args: argparse.Namespace) -> int:
    from .baseline import load_baseline, predict_baseline, read_manifest

    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and num-workers nonnegative")
    manifest = Path(args.manifest).resolve()
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = [row for row in read_manifest(manifest) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"No images in manifest split {args.split!r}")
    checkpoint = Path(args.checkpoint).resolve()
    model_info = {}
    if checkpoint.suffix.lower() in {".joblib", ".pkl"}:
        artifact = load_baseline(checkpoint)
        _verify_manifest_provenance(artifact.get("manifest_sha256"), manifest_sha256)
        predictions, probabilities = predict_baseline(artifact, rows, args.data_root)
        model_kind = "hog_logistic_v1"
    else:
        predictions, probabilities = _predict_resnet(
            checkpoint, rows, args, manifest_sha256, model_info=model_info,
        )
        model_kind = ("dacl_resnet18" if model_info.get("training_method") == "dacl_full"
                      else model_info.get("architecture", "resnet18"))
    output = Path(args.output_dir)
    truth = np.asarray([int(row["label"]) for row in rows])
    metrics = export_metrics(truth, predictions, output, CLASS_NAMES)
    export_predictions(rows, predictions, output, CLASS_NAMES, probabilities)
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_kind": model_kind, "checkpoint": str(checkpoint),
        **model_info,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "data_root": str(Path(args.data_root).resolve()),
        "split": args.split, "class_names": list(CLASS_NAMES),
        "note": "Held-out results are for reporting; do not tune models on test results.",
    }
    (output / "evaluation.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{args.split}: n={len(rows)}, accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}")
    print(f"Evaluation saved to {output}")
    return 0


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("evaluate", help="Evaluate a saved model on an explicit held-out split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.set_defaults(func=run)
    return parser
