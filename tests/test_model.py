"""Model integrity and data-leakage regressions; no pretrained downloads."""

import csv

import pytest
import torch
import torchvision.models._api

from emotion_recognition.constants import CLASS_NAMES
from emotion_recognition.model import build_model, load_checkpoint
from emotion_recognition.train import _configure_phase, _run_epoch, read_manifest


@pytest.fixture(autouse=True)
def no_downloads_and_single_cpu_thread(monkeypatch):
    def forbidden_download(*args, **kwargs):
        raise AssertionError("Las pruebas no deben descargar pesos preentrenados.")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden_download)
    monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", forbidden_download)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def test_checkpoint_roundtrip_preserves_predictions_and_metadata(tmp_path):
    torch.manual_seed(17)
    original = build_model(pretrained=False).eval()
    inputs = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        expected = original(inputs)
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "architecture": "resnet18",
        "model_state_dict": original.state_dict(),
        "class_names": list(CLASS_NAMES),
        "image_size": 64,
        "epoch": 3,
        "val_macro_f1": 0.25,
        "config": {"pretrained": False},
    }, checkpoint)

    loaded, metadata = load_checkpoint(checkpoint, device="cpu")
    with torch.inference_mode():
        actual = loaded(inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not loaded.training
    assert actual.shape == (2, 7)
    assert metadata["class_names"] == list(CLASS_NAMES)
    assert metadata["image_size"] == 64
    assert "model_state_dict" not in metadata


def test_checkpoint_rejects_different_class_order(tmp_path):
    checkpoint = tmp_path / "wrong_mapping.pt"
    torch.save({
        "model_state_dict": {},
        "class_names": list(reversed(CLASS_NAMES)),
    }, checkpoint)
    with pytest.raises(ValueError, match="orden de clases"):
        load_checkpoint(checkpoint)


def test_manifest_rejects_reencoded_image_leakage(tmp_path):
    manifest = tmp_path / "manifest.csv"
    fields = ["image_path", "label", "class_name", "split", "sha256", "pixel_sha256"]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"image_path": "train/a.jpg", "label": 0, "class_name": CLASS_NAMES[0],
             "split": "train", "sha256": "different-file-a", "pixel_sha256": "same-pixels"},
            {"image_path": "val/a.png", "label": 0, "class_name": CLASS_NAMES[0],
             "split": "val", "sha256": "different-file-b", "pixel_sha256": "same-pixels"},
        ])
    with pytest.raises(ValueError, match="Fuga de datos"):
        read_manifest(manifest)


def test_fine_tuning_only_unfreezes_last_block_and_classifier():
    model = build_model(pretrained=False)
    head_optimizer = _configure_phase(model, fine_tune=False, lr=1e-3, backbone_lr=1e-4)
    assert len(head_optimizer.param_groups) == 1
    assert all(parameter.requires_grad == name.startswith("fc.") for name, parameter in model.named_parameters())

    optimizer = _configure_phase(model, fine_tune=True, lr=1e-3, backbone_lr=1e-4)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]
    assert all(
        parameter.requires_grad == name.startswith(("layer4.", "fc."))
        for name, parameter in model.named_parameters()
    )


def test_head_training_does_not_update_frozen_batchnorm_statistics():
    torch.manual_seed(3)
    model = build_model(pretrained=False)
    optimizer = _configure_phase(model, fine_tune=False, lr=1e-3, backbone_lr=1e-4)
    before_mean = model.bn1.running_mean.clone()
    before_counter = model.bn1.num_batches_tracked.clone()
    before_head = model.fc.weight.detach().clone()
    inputs = torch.randn(2, 3, 64, 64)
    targets = torch.tensor([0, 1])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, targets), batch_size=2)

    metrics = _run_epoch(
        model, loader, torch.nn.CrossEntropyLoss(reduction="none"), torch.device("cpu"),
        optimizer=optimizer, scaler=torch.amp.GradScaler("cuda", enabled=False), fine_tune=False,
    )

    torch.testing.assert_close(model.bn1.running_mean, before_mean, rtol=0, atol=0)
    torch.testing.assert_close(model.bn1.num_batches_tracked, before_counter, rtol=0, atol=0)
    assert not torch.equal(model.fc.weight, before_head)
    assert 0 <= metrics["macro_f1"] <= 1
