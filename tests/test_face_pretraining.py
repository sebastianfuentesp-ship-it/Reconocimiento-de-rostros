"""Face initialization integrity checks: CPU only, no forward pass or downloads."""

import argparse

import pytest
import torch
import torchvision.models._api
from torch import nn
from torchvision import models

from emotion_recognition import pretrained
from emotion_recognition.constants import CLASS_NAMES
from emotion_recognition.model import build_model, load_checkpoint
from emotion_recognition.train import add_parser


@pytest.fixture(autouse=True)
def cpu_only_no_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not download weights or run a forward pass.")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", forbidden)
    monkeypatch.setattr(models.resnet.ResNet, "forward", forbidden)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


@pytest.fixture
def original_weights(tmp_path, monkeypatch):
    source = models.resnet18(weights=None)
    source.fc.weight.data.fill_(17.0)
    source.fc.bias.data.fill_(17.0)
    # The historical public file predates BatchNorm's num_batches_tracked.
    state = {key: value for key, value in source.state_dict().items()
             if not key.endswith("num_batches_tracked")}
    path = tmp_path / "synthetic_msceleb.pth"
    torch.save({"state_dict": state, "optimizer": {}, "epoch": 49}, path)

    def verified(candidate):
        assert candidate == path
        return {"path": str(path), "source": "msceleb", "dataset": "synthetic test fixture"}

    monkeypatch.setattr(pretrained, "verify_msceleb_weights", verified)
    return path, state


def test_face_weights_load_every_backbone_tensor_and_replace_original_head(original_weights):
    path, expected = original_weights
    model = build_model(pretrained_source="msceleb", pretrained_weights=path)
    assert isinstance(model.fc, nn.Linear)
    assert (model.fc.in_features, model.fc.out_features) == (512, 7)
    assert not torch.any(model.fc.weight == 17.0)
    assert model.initialization["source"] == "msceleb"
    for key, tensor in model.state_dict().items():
        if key.startswith("fc."):
            continue
        if key.endswith("num_batches_tracked"):
            assert tensor.item() == 0
        else:
            torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)


def test_external_deserialization_is_restricted_and_on_cpu(original_weights, monkeypatch):
    path, _ = original_weights
    real_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    build_model(pretrained_source="msceleb", pretrained_weights=path)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "payload"])
def test_incomplete_or_incompatible_backbone_cannot_silently_load(original_weights, corruption):
    path, state = original_weights
    state = dict(state)
    if corruption == "missing":
        del state["layer4.1.conv2.weight"]
    elif corruption == "unexpected":
        state["unknown.weight"] = torch.zeros(1)
    elif corruption == "shape":
        state["conv1.weight"] = torch.zeros(1)
    payload = {"state_dict": state} if corruption != "payload" else {"unrelated": state}
    torch.save(payload, path)
    error = ValueError if corruption == "payload" else RuntimeError
    with pytest.raises(error):
        build_model(pretrained_source="msceleb", pretrained_weights=path)


def test_integrity_check_happens_before_deserialization(tmp_path, monkeypatch):
    def reject(path):
        raise ValueError("SHA256 incorrecto")

    def forbidden(*args, **kwargs):
        raise AssertionError("Deserialization must not happen after failed verification.")

    monkeypatch.setattr(pretrained, "verify_msceleb_weights", reject)
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA256"):
        build_model(pretrained_source="msceleb", pretrained_weights=tmp_path / "bad.pth")


def test_trained_face_checkpoint_loads_without_original_external_file(original_weights, monkeypatch):
    path, _ = original_weights
    model = build_model(pretrained_source="msceleb", pretrained_weights=path)
    checkpoint = path.parent / "best.pt"
    torch.save({
        "architecture": "resnet18", "model_state_dict": model.state_dict(),
        "class_names": list(CLASS_NAMES), "image_size": 224,
        "config": {"initialization": model.initialization},
    }, checkpoint)
    path.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Trained checkpoints must not need external initial weights.")

    monkeypatch.setattr(pretrained, "verify_msceleb_weights", forbidden)
    restored, metadata = load_checkpoint(checkpoint, device="cpu")
    assert metadata["config"]["initialization"]["source"] == "msceleb"
    assert restored.fc.out_features == 7
    for key, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"pretrained_source": "unknown"},
    {"pretrained_source": "msceleb", "pretrained": False},
    {"pretrained_weights": "unexpected.pth"},
])
def test_ambiguous_initializations_fail_before_constructing_model(kwargs):
    with pytest.raises(ValueError):
        build_model(**kwargs)


def test_cli_keeps_previous_defaults_and_accepts_face_initialization():
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="command", required=True))
    required = ["train", "--manifest", "unused", "--data-root", "unused"]
    previous = parser.parse_args(required)
    assert previous.pretrained_source == "imagenet"
    assert previous.pretrained_weights is None
    face = parser.parse_args(required + [
        "--pretrained-source", "msceleb", "--pretrained-weights", "face.pth",
        "--fine-tune-from", "layer3",
    ])
    assert face.pretrained_source == "msceleb"
    assert face.pretrained_weights == "face.pth"
    assert face.fine_tune_from == "layer3"
