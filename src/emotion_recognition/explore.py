"""Describe prepared class distributions and show training-only image samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .baseline import read_manifest, resolve_image
from .constants import CLASS_NAMES


def run(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = Path(args.manifest).resolve()
    rows = read_manifest(manifest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = ("train", "val", "test")
    counts = {
        split: Counter(int(row["label"]) for row in rows if row["split"] == split)
        for split in splits
    }
    totals = {split: sum(counts[split].values()) for split in splits}
    distribution = []
    for label, name in enumerate(CLASS_NAMES):
        record = {"label": label, "class_name": name}
        for split in splits:
            record[f"{split}_count"] = counts[split][label]
            record[f"{split}_proportion"] = (
                counts[split][label] / totals[split] if totals[split] else 0.0
            )
        record["total_count"] = sum(counts[split][label] for split in splits)
        distribution.append(record)
    with (output / "distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distribution[0]))
        writer.writeheader()
        writer.writerows(distribution)

    figure, axis = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(CLASS_NAMES))
    width = 0.26
    for offset, split, color in zip((-1, 0, 1), splits, ("#2466A6", "#D78925", "#3C8D64")):
        values = [counts[split][label] for label in range(len(CLASS_NAMES))]
        bars = axis.bar(positions + offset * width, values, width, color=color,
                        label=f"{split} (n={totals[split]:,})")
        axis.bar_label(bars, padding=3, fontsize=8)
    axis.set(
        xticks=positions, xticklabels=CLASS_NAMES, xlabel="Expression class",
        ylabel="Number of images", title="Prepared dataset: class distribution by split",
    )
    axis.set_ylim(0, max(max(counter.values(), default=0) for counter in counts.values()) * 1.18 + 1)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output / "class_distribution.png", dpi=160)
    plt.close(figure)

    # Only training images are opened. Test metadata is counted above solely to
    # document the official evaluation set and its class imbalance.
    rng = random.Random(args.seed)
    sample_records = []
    figure, axes = plt.subplots(len(CLASS_NAMES), 3, figsize=(10, 18), squeeze=False)
    for label, name in enumerate(CLASS_NAMES):
        candidates = sorted(
            [row for row in rows if row["split"] == "train" and int(row["label"]) == label],
            key=lambda row: row["image_path"],
        )
        selected = rng.sample(candidates, min(3, len(candidates)))
        for column, axis in enumerate(axes[label]):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column == 0:
                axis.set_ylabel(f"{label}: {name}", rotation=90, labelpad=12, fontsize=11)
            if column >= len(selected):
                axis.text(0.5, 0.5, "No training sample", ha="center", va="center")
                continue
            row = selected[column]
            image_path = resolve_image(args.data_root, row["image_path"])
            with Image.open(image_path) as source:
                source_mode = source.mode
                source_format = source.format
                image = ImageOps.exif_transpose(source).convert("RGB")
                sample_records.append({
                    "image_path": row["image_path"], "split": "train", "label": label,
                    "class_name": name, "width": image.width, "height": image.height,
                    "source_mode": source_mode, "source_format": source_format,
                })
                axis.imshow(image)
            axis.set_title(Path(row["image_path"]).name, fontsize=9)
    figure.suptitle("Training samples only: three images per class", fontsize=16, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(output / "samples.png", dpi=140)
    plt.close(figure)

    train_counts = [counts["train"][label] for label in range(len(CLASS_NAMES))]
    train_nonzero = [count for count in train_counts if count]
    report = {
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "data_root": str(Path(args.data_root).resolve()),
        "class_mapping": {str(label): name for label, name in enumerate(CLASS_NAMES)},
        "split_counts": totals,
        "class_distribution": distribution,
        "train_missing_classes": [name for name, count in zip(CLASS_NAMES, train_counts) if not count],
        "train_imbalance_ratio_max_over_min": (
            max(train_counts) / min(train_counts) if train_counts and min(train_counts) > 0 else None
        ),
        "train_imbalance_ratio_present_classes": (
            max(train_nonzero) / min(train_nonzero) if train_nonzero else None
        ),
        "sample_seed": args.seed,
        "image_statistics_scope": "At most three train images per class; no val/test images opened.",
        "sampled_images": sample_records,
        "sampled_size_counts": dict(Counter(f"{row['width']}x{row['height']}" for row in sample_records)),
    }
    if args.cleaning_report:
        cleaning_path = Path(args.cleaning_report).resolve()
        cleaning = json.loads(cleaning_path.read_text(encoding="utf-8"))
        report["cleaning_report"] = str(cleaning_path)
        report["cleaning_summary"] = {
            key: cleaning[key] for key in (
                "retained_records", "excluded_records", "official_test_exclusions",
                "requires_manual_near_duplicate_review", "exact_duplicates_between_splits",
                "source_files_modified",
            ) if key in cleaning
        }
    (output / "eda.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Class counts: {totals}")
    print(f"Training imbalance ratio: {report['train_imbalance_ratio_max_over_min']}")
    print(f"Saved EDA to {output}; opened {len(sample_records)} training images only.")
    return 0


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("explore", help="Summarize cleaned class counts and training samples")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cleaning-report")
    parser.add_argument("--output-dir", default="artifacts/data_analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=run)
    return parser
