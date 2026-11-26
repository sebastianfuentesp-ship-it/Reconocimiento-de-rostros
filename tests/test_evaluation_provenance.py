"""Refuse repartitioned datasets before any held-out image is opened."""

import argparse
import csv
import hashlib

import pytest

from emotion_recognition import baseline, evaluate
from emotion_recognition.constants import CLASS_NAMES


def _evaluation_inputs(tmp_path, suffix):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "label", "class_name", "split"])
        writer.writeheader()
        writer.writerow({
            "image_path": "deliberately_missing_test_image.png", "label": 0,
            "class_name": CLASS_NAMES[0], "split": "test",
        })
    checkpoint = tmp_path / f"checkpoint{suffix}"
    checkpoint.write_bytes(b"placeholder; deserialization is monkeypatched")
    args = argparse.Namespace(
        manifest=str(manifest), data_root=str(tmp_path), checkpoint=str(checkpoint),
        output_dir=str(tmp_path / "evaluation"), split="test", batch_size=2,
        device="cpu", num_workers=0,
    )
    return args, manifest


@pytest.mark.parametrize("expected_hash", [None, "0" * 64])
def test_baseline_evaluation_rejects_missing_or_mismatched_provenance_before_images(
    tmp_path, monkeypatch, expected_hash,
):
    args, _ = _evaluation_inputs(tmp_path, ".joblib")
    artifact = {} if expected_hash is None else {"manifest_sha256": expected_hash}
    monkeypatch.setattr(baseline, "load_baseline", lambda checkpoint: artifact)

    def forbidden_prediction(*args, **kwargs):
        raise AssertionError("Image prediction must not run with invalid provenance.")

    monkeypatch.setattr(baseline, "predict_baseline", forbidden_prediction)
    with pytest.raises(ValueError, match="manifest_sha256"):
        evaluate.run(args)
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize("expected_hash", [None, "0" * 64])
def test_resnet_evaluation_rejects_missing_or_mismatched_provenance_before_images(
    tmp_path, monkeypatch, expected_hash,
):
    from emotion_recognition import model

    args, _ = _evaluation_inputs(tmp_path, ".pt")
    config = {} if expected_hash is None else {"manifest_sha256": expected_hash}
    metadata = {"class_names": list(CLASS_NAMES), "config": config, "image_size": 64}
    monkeypatch.setattr(model, "load_checkpoint", lambda *args, **kwargs: (None, metadata))
    with pytest.raises(ValueError, match="manifest_sha256"):
        evaluate.run(args)
    assert not (tmp_path / "evaluation").exists()


def test_matching_manifest_fingerprint_is_accepted(tmp_path):
    _, manifest = _evaluation_inputs(tmp_path, ".joblib")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    evaluate._verify_manifest_provenance(digest, digest)
