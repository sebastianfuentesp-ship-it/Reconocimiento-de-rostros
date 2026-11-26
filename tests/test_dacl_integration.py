"""Architecture routing and exported-checkpoint provenance without CNN execution."""

import argparse
import csv
import json

import numpy as np
import pytest
import torch
from torch import nn

from emotion_recognition import dacl_model, evaluate, model
from emotion_recognition.constants import CLASS_NAMES


class TinyDACL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 7, device="cpu")
        self.attention = nn.Linear(2, 2, device="cpu")

    def forward(self, *args):
        raise AssertionError("Checkpoint tests must not run model inference.")


@pytest.fixture(autouse=True)
def forbid_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No downloads, CUDA or neural network execution in these tests.")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(nn.Module, "_call_impl", forbidden)


def payload(state, architecture="dacl_resnet18", profile="dacl_baseline"):
    return {
        "architecture": architecture, "training_method": "dacl_full",
        "model_state_dict": state, "class_names": list(CLASS_NAMES),
        "image_size": 224, "preprocessing_profile": profile,
        "config": {"manifest_sha256": "test-manifest", "training_method": "dacl_full"},
    }


def test_full_checkpoint_loads_strictly_without_original_weights(tmp_path, monkeypatch):
    original = TinyDACL()
    path = tmp_path / "best.pt"
    torch.save(payload(original.state_dict()), path)
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return TinyDACL()

    monkeypatch.setattr(dacl_model, "build_dacl_model", factory)
    loaded, metadata = model.load_checkpoint(path, device="cpu")
    assert calls == [{"pretrained": False}]
    assert not loaded.training
    assert metadata["architecture"] == "dacl_resnet18"
    assert metadata["training_method"] == "dacl_full"
    assert metadata["preprocessing_profile"] == "dacl_baseline"
    for name, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def test_full_checkpoint_rejects_incomplete_attention_weights(tmp_path, monkeypatch):
    state = TinyDACL().state_dict()
    del state["attention.weight"]
    path = tmp_path / "incomplete.pt"
    torch.save(payload(state), path)
    monkeypatch.setattr(dacl_model, "build_dacl_model", lambda **kwargs: TinyDACL())
    with pytest.raises(RuntimeError, match="attention.weight"):
        model.load_checkpoint(path)


def test_full_checkpoint_requires_its_preprocessing_profile(tmp_path, monkeypatch):
    path = tmp_path / "wrong_profile.pt"
    torch.save(payload({}, profile="imagenet"), path)

    def forbidden(**kwargs):
        raise AssertionError("Reject incompatible metadata before allocating the full model.")

    monkeypatch.setattr(dacl_model, "build_dacl_model", forbidden)
    with pytest.raises(ValueError, match="dacl_baseline"):
        model.load_checkpoint(path)


def test_inference_export_loads_only_classifier_and_keeps_dacl_training_method(tmp_path, monkeypatch):
    original = nn.Linear(2, 7, device="cpu")
    path = tmp_path / "inference.pt"
    torch.save(payload(original.state_dict(), architecture="resnet18"), path)
    calls = []

    def plain_factory(**kwargs):
        calls.append(kwargs)
        return nn.Linear(2, 7, device="cpu")

    monkeypatch.setattr(model, "build_model", plain_factory)
    loaded, metadata = model.load_checkpoint(path)
    assert calls == [{"num_classes": 7, "pretrained": False}]
    assert metadata["training_method"] == "dacl_full"
    torch.testing.assert_close(loaded.weight, original.weight, rtol=0, atol=0)


def test_evaluation_records_dacl_method_for_lightweight_export(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    rows = [{"image_path": f"missing-{i}.jpg", "label": i, "class_name": name, "split": "val"}
            for i, name in enumerate(CLASS_NAMES)]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = tmp_path / "inference.pt"
    checkpoint.write_bytes(b"Fake checkpoint, predictions are mocked.")
    output = tmp_path / "validation"

    def fake_predictions(checkpoint, rows, args, manifest_sha256, model_info=None):
        model_info.update({"architecture": "resnet18", "training_method": "dacl_full"})
        return np.arange(7), np.eye(7)

    def fake_metrics(truth, predicted, path, names):
        path.mkdir()
        return {"accuracy": 1.0, "macro_f1": 1.0}

    monkeypatch.setattr(evaluate, "_predict_resnet", fake_predictions)
    monkeypatch.setattr(evaluate, "export_metrics", fake_metrics)
    args = argparse.Namespace(
        manifest=str(manifest), checkpoint=str(checkpoint), output_dir=str(output),
        data_root=str(tmp_path), split="val", batch_size=2, num_workers=0, device="cpu",
    )
    assert evaluate.run(args) == 0
    provenance = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert provenance["model_kind"] == "dacl_resnet18"
    assert provenance["architecture"] == "resnet18"
    assert provenance["training_method"] == "dacl_full"
    assert provenance["split"] == "val"
