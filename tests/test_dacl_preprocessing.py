"""DACL image geometry and checkpoint compatibility without model inference."""

import numpy as np
import pytest
import torch
from PIL import Image

from emotion_recognition import model
from emotion_recognition.constants import CLASS_NAMES


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def coordinate_image(width, height):
    y, x = np.indices((height, width))
    pixels = np.stack((x % 256, y % 256, ((x + y) // 2) % 256), axis=-1).astype("uint8")
    return Image.fromarray(pixels)


@pytest.mark.parametrize("index,xy", [(0, (0, 0)), (1, (8, 0)), (2, (0, 6)),
                                    (3, (8, 6)), (4, (4, 3))])
def test_random_five_crop_selects_each_whole_corner_or_center(monkeypatch, index, xy):
    image = coordinate_image(12, 10)
    calls = []

    def choose(low, high):
        calls.append((low, high))
        return index

    monkeypatch.setattr(model.random, "randint", choose)
    result = model.RandomFiveCrop(4)(image)
    x, y = xy
    expected = np.asarray(image)[y:y + 4, x:x + 4]
    np.testing.assert_array_equal(np.asarray(result), expected)
    assert result.size == (4, 4)
    assert calls == [(0, 4)]


def test_dacl_validation_preserves_aspect_ratio_then_takes_center():
    image = coordinate_image(320, 256)
    transform = model.build_transforms(profile="dacl_baseline")
    first = transform(image)
    second = transform(image)
    assert first.shape == (3, 224, 224)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    # The short side already measures256: crop224 starts at x48,y16.
    mean = torch.tensor([0.5752, 0.4495, 0.4012])[:, None, None]
    std = torch.tensor([0.2086, 0.1911, 0.1827])[:, None, None]
    restored = (first * std + mean).permute(1, 2, 0).numpy() * 255
    np.testing.assert_allclose(restored, np.asarray(image)[16:240, 48:272], atol=0.00003)


def test_validation_uses_dacl_normalization_and_legacy_default_remains_imagenet():
    image = Image.new("RGB", (256, 256), (128, 64, 192))
    dacl = model.build_transforms(profile="dacl_baseline")(image)
    legacy = model.build_transforms()(image)
    rgb = torch.tensor([128, 64, 192], dtype=torch.float32) / 255
    expected_dacl = (rgb - torch.tensor([0.5752, 0.4495, 0.4012])) / torch.tensor(
        [0.2086, 0.1911, 0.1827]
    )
    expected_legacy = (rgb - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
        [0.229, 0.224, 0.225]
    )
    torch.testing.assert_close(dacl[:, 0, 0], expected_dacl)
    torch.testing.assert_close(dacl[:, -1, -1], expected_dacl)
    torch.testing.assert_close(legacy[:, 0, 0], expected_legacy)
    assert not torch.allclose(dacl, legacy)


def test_dacl_training_crop_and_flip_use_same_pixel_scale(monkeypatch):
    image = coordinate_image(256, 256)
    monkeypatch.setattr(model.random, "randint", lambda low, high: 1)  # top-right crop
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.0]))
    actual = model.build_transforms(train=True, profile="dacl_baseline")(image)
    crop_flipped = np.asarray(image)[:224, 32:256][:, ::-1].copy()
    expected = torch.from_numpy(crop_flipped).permute(2, 0, 1).float() / 255
    expected = (expected - torch.tensor([0.5752, 0.4495, 0.4012])[:, None, None]) / torch.tensor(
        [0.2086, 0.1911, 0.1827]
    )[:, None, None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("profile,size,message", [
    ("unknown", 224, "Perfil"),
    ("dacl_baseline", 256, "requiere 224"),
    ("dacl_baseline", 64, "requiere 224"),
    ("imagenet", 16, "image_size"),
])
def test_transform_rejects_unknown_or_incompatible_profile(profile, size, message):
    with pytest.raises(ValueError, match=message):
        model.build_transforms(profile=profile, image_size=size)


@pytest.mark.parametrize("saved_profile,size,expected_profile", [
    ("dacl_baseline", 224, "dacl_baseline"),
    ("imagenet", 64, "imagenet"),
    (None, 224, "imagenet"),
])
def test_checkpoint_preserves_preprocessing_without_pretrained_download(
    tmp_path, monkeypatch, saved_profile, size, expected_profile
):
    calls = []
    saved_state = {"test": torch.tensor([1.0])}
    payload = {"model_state_dict": saved_state, "class_names": list(CLASS_NAMES),
               "image_size": size, "architecture": "resnet18"}
    if saved_profile is not None:
        payload["preprocessing_profile"] = saved_profile

    class OfflineModel(torch.nn.Module):
        def load_state_dict(self, state_dict, strict=True):
            assert state_dict is saved_state
            assert strict is True
            calls.append("strict_load")

    def offline_build(*, num_classes, pretrained):
        assert num_classes == 7
        assert pretrained is False
        calls.append("build_without_pretraining")
        return OfflineModel()

    def read_checkpoint(path, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        return payload

    monkeypatch.setattr(model, "build_model", offline_build)
    monkeypatch.setattr(model.torch, "load", read_checkpoint)
    loaded, metadata = model.load_checkpoint(tmp_path / "synthetic.pt", device="cpu")
    assert not loaded.training
    assert metadata["preprocessing_profile"] == expected_profile
    assert metadata["image_size"] == size
    assert "model_state_dict" not in metadata
    assert calls == ["build_without_pretraining", "strict_load"]


@pytest.mark.parametrize("profile,size,message", [
    ("unknown", 224, "Perfil"),
    ("dacl_baseline", 256, "requiere 224"),
    ("imagenet", 0, "image_size"),
])
def test_invalid_checkpoint_profile_fails_before_model_construction(
    tmp_path, monkeypatch, profile, size, message
):
    payload = {"model_state_dict": {}, "class_names": list(CLASS_NAMES),
               "preprocessing_profile": profile, "image_size": size}
    monkeypatch.setattr(model.torch, "load", lambda *args, **kwargs: payload)

    def forbidden_build(*args, **kwargs):
        pytest.fail("Se debe validar el perfil antes de construir o descargar el modelo.")

    monkeypatch.setattr(model, "build_model", forbidden_build)
    with pytest.raises(ValueError, match=message):
        model.load_checkpoint(tmp_path / "invalid.pt", device="cpu")
