"""Check held-out isolation and complete-class reporting using synthetic images."""

import argparse
import csv
import json

import numpy as np
import pytest
from PIL import Image

from emotion_recognition import baseline, evaluate
from emotion_recognition.constants import CLASS_NAMES

FIELDS = ["image_path", "label", "class_name", "split", "sha256", "pixel_sha256", "group_id"]


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_corpus(tmp_path):
    root = tmp_path / "raw"
    rows = []
    rng = np.random.default_rng(42)
    for split, samples in (("train", 4), ("val", 2), ("test", 1)):
        for label, name in enumerate(CLASS_NAMES):
            for index in range(samples):
                relative = f"{split}/{name}/{index}.png"
                # Test images intentionally do not exist. Fitting and validation
                # selection must work without ever opening held-out pixels.
                if split != "test":
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pixels = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
                    pixels[:, label * 4:label * 4 + 3] = 255
                    Image.fromarray(pixels).save(path)
                rows.append({
                    "image_path": relative, "label": label, "class_name": name,
                    "split": split, "sha256": f"file-{relative}",
                    "pixel_sha256": f"pixels-{relative}", "group_id": f"group-{relative}",
                })
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    return root, manifest, rows


def fit_args(root, manifest, output):
    return argparse.Namespace(
        manifest=str(manifest), data_root=str(root), output_dir=str(output),
        seed=42, image_size=32, pca_components=5, c_values=[0.1, 1.0],
    )


def test_baseline_fits_train_only_and_test_changes_do_not_change_model(tmp_path):
    root, manifest, rows = synthetic_corpus(tmp_path)
    first = tmp_path / "baseline1"
    assert baseline.run(fit_args(root, manifest, first)) == 0
    artifact = baseline.load_baseline(first / "baseline.joblib")
    assert artifact["test_evaluated"] is False
    assert tuple(artifact["class_names"]) == tuple(CLASS_NAMES)
    assert artifact["training_samples"] == 28
    assert artifact["validation_samples"] == 14
    assert not (first / "test").exists()

    training = [row for row in baseline.read_manifest(manifest) if row["split"] == "train"]
    train_features, _ = baseline.extract_features(training, root, 32)
    np.testing.assert_allclose(
        artifact["pipeline"].named_steps["scale"].mean_, train_features.mean(axis=0),
        atol=1e-7,
    )
    assert artifact["pipeline"].named_steps["pca"].n_components_ == 5
    assert len(artifact["search"]) == 2

    # Change held-out labels and image paths. A baseline selected using train/val
    # must remain identical even though the manifest fingerprint changes.
    for row in rows:
        if row["split"] == "test":
            row["label"] = (row["label"] + 1) % len(CLASS_NAMES)
            row["class_name"] = CLASS_NAMES[row["label"]]
            row["image_path"] = "unavailable/" + row["image_path"]
    write_manifest(manifest, rows)
    second = tmp_path / "baseline2"
    assert baseline.run(fit_args(root, manifest, second)) == 0
    other = baseline.load_baseline(second / "baseline.joblib")
    assert artifact["manifest_sha256"] != other["manifest_sha256"]
    assert artifact["selected_C"] == other["selected_C"]
    np.testing.assert_allclose(
        artifact["pipeline"].named_steps["classifier"].coef_,
        other["pipeline"].named_steps["classifier"].coef_,
    )
    validation = [row for row in baseline.read_manifest(manifest) if row["split"] == "val"]
    first_predictions, first_probs = baseline.predict_baseline(artifact, validation, root)
    second_predictions, second_probs = baseline.predict_baseline(other, validation, root)
    np.testing.assert_array_equal(first_predictions, second_predictions)
    np.testing.assert_allclose(first_probs, second_probs)
    np.testing.assert_allclose(first_probs.sum(axis=1), 1.0)


def test_evaluate_baseline_exports_predictions_and_provenance_without_torch(tmp_path):
    root, manifest, _ = synthetic_corpus(tmp_path)
    model_dir = tmp_path / "baseline"
    baseline.run(fit_args(root, manifest, model_dir))
    output = tmp_path / "evaluation"
    args = argparse.Namespace(
        manifest=str(manifest), data_root=str(root), checkpoint=str(model_dir / "baseline.joblib"),
        output_dir=str(output), split="val", batch_size=8, device="cpu", num_workers=0,
    )
    assert evaluate.run(args) == 0
    provenance = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert provenance["split"] == "val"
    assert provenance["model_kind"] == "hog_logistic_v1"
    assert metrics["samples"] == 14
    with (output / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 14
    assert all(row["image_path"].startswith("val/") for row in predictions)
    assert all("probability_neutral" in row for row in predictions)


def test_export_metrics_keeps_all_seven_classes_when_some_are_absent(tmp_path):
    metrics = evaluate.export_metrics([0, 0, 1], [0, 1, 1], tmp_path)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(4 / 21)
    assert metrics["per_class"]["neutral"]["support"] == 0
    assert metrics["per_class"]["neutral"]["f1-score"] == 0
    assert np.asarray(metrics["confusion_matrix"]).shape == (7, 7)
    normalized = np.asarray(metrics["confusion_matrix_normalized"])
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized[2:], 0)
    for filename in ("metrics.json", "confusion_matrix.csv", "confusion_matrix.png",
                     "confusion_matrix_normalized.csv", "confusion_matrix_normalized.png"):
        assert (tmp_path / filename).is_file()
        assert (tmp_path / filename).stat().st_size > 0


@pytest.mark.parametrize("field", ["image_path", "sha256", "pixel_sha256", "group_id"])
@pytest.mark.parametrize("splits", [("train", "test"), ("val", "test"), ("train", "val")])
def test_manifest_rejects_leakage_between_every_pair_of_splits(tmp_path, field, splits):
    _, manifest, rows = synthetic_corpus(tmp_path)
    first = next(row for row in rows if row["split"] == splits[0])
    second = next(row for row in rows if row["split"] == splits[1])
    second[field] = first[field]
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="Data leakage"):
        baseline.read_manifest(manifest)


def test_manifest_rejects_empty_input(tmp_path):
    manifest = tmp_path / "empty.csv"
    write_manifest(manifest, [])
    with pytest.raises(ValueError, match="empty"):
        baseline.read_manifest(manifest)
