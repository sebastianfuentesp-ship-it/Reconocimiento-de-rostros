"""Reproducible HOG + logistic regression baseline selected on validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import CLASS_NAMES

HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "feature_vector": True,
}


def read_manifest(manifest: str | Path) -> list[dict[str, str]]:
    """Read the prepared manifest without loading images from any split."""
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "label", "class_name", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("The manifest is empty")
    identities: dict[tuple[str, str], str] = {}
    for row in rows:
        label = int(row["label"])
        if not 0 <= label < len(CLASS_NAMES):
            raise ValueError(f"Invalid class label {label}: {row['image_path']}")
        if row["class_name"] != CLASS_NAMES[label]:
            raise ValueError(f"Class mapping mismatch: {row['image_path']}")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {row['split']!r}")
        # Inspect metadata for every split, without reading test pixels. A new
        # manifest must not reintroduce a known image/group across partitions.
        for field in ("image_path", "sha256", "pixel_sha256", "group_id"):
            value = row.get(field)
            if value:
                identity = (field, value)
                previous = identities.get(identity)
                if previous is not None and previous != row["split"]:
                    raise ValueError(
                        f"Data leakage between {previous} and {row['split']} detected via {field}"
                    )
                identities[identity] = row["split"]
    return rows


def resolve_image(data_root: str | Path, image_path: str) -> Path:
    """Only permit image paths inside the selected dataset directory."""
    root = Path(data_root).resolve()
    relative = Path(image_path)
    if relative.is_absolute():
        raise ValueError(f"Manifest image path must be relative: {image_path}")
    image = (root / relative).resolve()
    if not image.is_relative_to(root):
        raise ValueError(f"Manifest image path escapes data root: {image_path}")
    if not image.is_file():
        raise FileNotFoundError(image)
    return image


def extract_hog(image_path: str | Path, image_size: int = 96) -> np.ndarray:
    """Extract the same deterministic descriptor for training and inference."""
    if image_size < 16:
        raise ValueError("HOG image size must be at least 16 pixels")
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return hog(pixels, **HOG_CONFIG).astype(np.float32, copy=False)


def extract_features(
    rows: list[dict[str, str]], data_root: str | Path, image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("The selected manifest split is empty")
    features = []
    for index, row in enumerate(rows, start=1):
        features.append(extract_hog(resolve_image(data_root, row["image_path"]), image_size))
        if index % 1000 == 0:
            print(f"HOG: {index}/{len(rows)} images", flush=True)
    return np.stack(features), np.asarray([int(row["label"]) for row in rows])


def load_baseline(checkpoint: str | Path) -> dict:
    """Load a locally produced artifact. Joblib files must be trusted."""
    artifact = joblib.load(checkpoint)
    if artifact.get("format") != "hog_logistic_v1":
        raise ValueError("Unsupported baseline artifact format")
    if tuple(artifact.get("class_names", ())) != tuple(CLASS_NAMES):
        raise ValueError("Baseline class mapping does not match this project")
    if artifact.get("hog_config") != HOG_CONFIG:
        raise ValueError("Baseline HOG configuration does not match this project")
    return artifact


def predict_baseline(
    artifact: dict, rows: list[dict[str, str]], data_root: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    features, _ = extract_features(rows, data_root, artifact["image_size"])
    pipeline = artifact["pipeline"]
    predictions = pipeline.predict(features)
    probabilities = np.zeros((len(rows), len(CLASS_NAMES)), dtype=float)
    probabilities[:, pipeline.classes_.astype(int)] = pipeline.predict_proba(features)
    return predictions, probabilities


def run(args: argparse.Namespace) -> int:
    from .evaluate import export_metrics, export_predictions

    manifest = Path(args.manifest).resolve()
    rows = read_manifest(manifest)
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "val"]
    if not training or not validation:
        raise ValueError("Baseline needs nonempty train and val splits")
    present = {int(row["label"]) for row in training}
    if present != set(range(len(CLASS_NAMES))):
        raise ValueError("The training split must contain all seven classes")
    if args.pca_components < 0:
        raise ValueError("--pca-components must be zero (disabled) or a positive integer")
    if not args.c_values or any(value <= 0 for value in args.c_values):
        raise ValueError("All C values must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x_train, y_train = extract_features(training, args.data_root, args.image_size)
    x_val, y_val = extract_features(validation, args.data_root, args.image_size)
    if args.pca_components > min(x_train.shape):
        raise ValueError(f"PCA components cannot exceed {min(x_train.shape)}")

    # Fit feature normalization and dimensionality reduction using train only.
    preprocessing_steps = [("scale", StandardScaler())]
    if args.pca_components:
        preprocessing_steps.append(
            ("pca", PCA(n_components=args.pca_components, random_state=args.seed))
        )
    preprocessing = Pipeline(preprocessing_steps)
    transformed_train = preprocessing.fit_transform(x_train)
    transformed_val = preprocessing.transform(x_val)

    best_score = -1.0
    best_classifier = None
    best_predictions = None
    search = []
    labels = list(range(len(CLASS_NAMES)))
    for value in args.c_values:
        classifier = LogisticRegression(
            C=value,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=args.seed,
        )
        classifier.fit(transformed_train, y_train)
        predictions = classifier.predict(transformed_val)
        score = f1_score(y_val, predictions, labels=labels, average="macro", zero_division=0)
        search.append({"C": value, "validation_macro_f1": float(score)})
        print(f"C={value:g}, validation macro F1={score:.4f}", flush=True)
        if score > best_score:
            best_score = float(score)
            best_classifier = classifier
            best_predictions = predictions

    pipeline = Pipeline([*preprocessing.steps, ("classifier", best_classifier)])
    config = {
        "format": "hog_logistic_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "class_names": list(CLASS_NAMES),
        "image_size": args.image_size,
        "hog_config": HOG_CONFIG,
        "pca_components": args.pca_components,
        "seed": args.seed,
        "class_weight": "balanced",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "manifest_path": str(manifest),
        "data_root": str(Path(args.data_root).resolve()),
        "training_samples": len(training),
        "validation_samples": len(validation),
        "selected_C": best_classifier.C,
        "validation_macro_f1": best_score,
        "selection_metric": "macro F1 over all seven classes on val only",
        "search": search,
        "test_evaluated": False,
    }
    artifact = {**config, "pipeline": pipeline}
    joblib.dump(artifact, output / "baseline.joblib")
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    export_metrics(y_val, best_predictions, output / "validation", CLASS_NAMES)
    export_predictions(validation, best_predictions, output / "validation", CLASS_NAMES)
    print(f"Saved baseline to {output / 'baseline.joblib'}; test remains unevaluated.")
    return 0


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("baseline", help="Train the HOG + logistic regression baseline")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="artifacts/baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.set_defaults(func=run)
    return parser
