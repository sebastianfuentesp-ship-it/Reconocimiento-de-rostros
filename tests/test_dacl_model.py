"""DACL architecture inspection; no real CNN forward, training, downloads or CUDA."""

import math

import pytest
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet

from emotion_recognition import dacl_model
from emotion_recognition import model as shared_model


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def no_training_download_or_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No se permite ejecutar una CNN, entrenar, descargar ni usar CUDA.")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    # Even accidental calls to a real ResNet forward are forbidden.
    monkeypatch.setattr(ResNet, "forward", forbidden)


def meta_backbone():
    with torch.device("meta"):
        backbone = resnet18(weights=None, num_classes=7)
    backbone.initialization = {"source": "synthetic", "sha256": "fixture"}
    return backbone


def meta_wrapper():
    with torch.device("meta"):
        return dacl_model.DACLResNet18(meta_backbone())


def test_attention_dimensions_and_activations_match_published_resnet18_branch():
    wrapped = meta_wrapper()
    linears = [module for module in wrapped.attention if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (512 * 7 * 7, 3584), (3584, 512), (512, 64),
    ]
    normalizations = [module for module in wrapped.attention if isinstance(module, nn.BatchNorm1d)]
    assert [module.num_features for module in normalizations] == [3584, 512, 64]
    assert sum(isinstance(module, nn.ReLU) for module in wrapped.attention) == 2
    assert isinstance(wrapped.attention[-1], nn.Tanh)
    assert wrapped.attention_heads.in_features == 64
    assert wrapped.attention_heads.out_features == 1024
    assert wrapped.backbone.fc.in_features == 512
    assert wrapped.backbone.fc.out_features == 7
    assert all(parameter.device.type == "meta" for parameter in wrapped.parameters())


def test_wrapping_never_reinitializes_or_replaces_backbone_parameters(monkeypatch):
    backbone = meta_backbone()
    original_parameters = dict(backbone.named_parameters())
    original_buffers = dict(backbone.named_buffers())
    original_head = backbone.fc
    touched = []

    def record_weight(tensor, *args, **kwargs):
        touched.append(tensor)
        return tensor

    monkeypatch.setattr(nn.init, "kaiming_normal_", record_weight)
    with torch.device("meta"):
        wrapped = dacl_model.DACLResNet18(backbone)
    assert wrapped.backbone is backbone
    assert wrapped.backbone.fc is original_head
    assert wrapped.initialization is backbone.initialization
    assert {id(tensor) for tensor in touched} == {
        id(module.weight) for module in list(wrapped.attention) + [wrapped.attention_heads]
        if isinstance(module, nn.Linear)
    }
    assert all(wrapped.backbone.get_parameter(name) is parameter
               for name, parameter in original_parameters.items())
    assert all(wrapped.backbone.get_buffer(name) is buffer
               for name, buffer in original_buffers.items())


def test_attention_initializer_resets_only_linear_weight_and_bias():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(431)
        linear = nn.Linear(64, 32)
        with torch.no_grad():
            linear.weight.fill_(5)
            linear.bias.fill_(9)
        dacl_model._initialize_attention_linear(linear)
    assert torch.isfinite(linear.weight).all()
    assert not torch.all(linear.weight == 5)
    assert abs(linear.weight.detach().mean().item()) < 0.02
    assert linear.weight.detach().std().item() == pytest.approx(math.sqrt(2 / 64), rel=0.08)
    torch.testing.assert_close(linear.bias, torch.zeros(32), rtol=0, atol=0)
    normalization = nn.BatchNorm1d(3)
    with torch.no_grad():
        normalization.weight.fill_(2)
        normalization.bias.fill_(3)
    dacl_model._initialize_attention_linear(normalization)
    torch.testing.assert_close(normalization.weight, torch.full((3,), 2.0))
    torch.testing.assert_close(normalization.bias, torch.full((3,), 3.0))


def test_numeric_backbone_and_classifier_values_survive_wrapping():
    backbone = nn.Module()
    backbone.conv1 = nn.Conv2d(3, 4, kernel_size=1)
    backbone.fc = nn.Linear(512, 7)
    backbone.initialization = {"source": "tiny-numeric-backbone"}
    with torch.no_grad():
        backbone.conv1.weight.fill_(0.125)
        backbone.fc.weight.fill_(-0.25)
        backbone.fc.bias.fill_(0.75)
    before = {name: tensor.detach().clone() for name, tensor in backbone.state_dict().items()}
    with torch.device("meta"):
        wrapped = dacl_model.DACLResNet18(backbone)
    for name, expected in before.items():
        torch.testing.assert_close(wrapped.backbone.state_dict()[name], expected, rtol=0, atol=0)


def test_factory_selects_face_weights_and_offline_checkpoint_mode(monkeypatch):
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return meta_backbone()

    monkeypatch.setattr(shared_model, "build_model", fake_build)
    with torch.device("meta"):
        face = dacl_model.build_dacl_model(pretrained_weights="provided.pth")
        offline = dacl_model.build_dacl_model(pretrained=False)
    assert isinstance(face, dacl_model.DACLResNet18)
    assert isinstance(offline, dacl_model.DACLResNet18)
    assert calls == [
        {"pretrained": True, "pretrained_source": "msceleb_dacl",
         "pretrained_weights": "provided.pth"},
        {"pretrained": False},
    ]


class ToyClassifier(nn.Module):
    def forward(self, features):
        return features[:, :7] + 1


class ForbiddenAttention(nn.Module):
    def forward(self, value):
        pytest.fail("La ruta para fotos/evaluación no debe calcular atención.")


def toy_dispatch(monkeypatch):
    """Replace all image processing by a predefined small mathematical fixture."""
    wrapped = meta_wrapper()
    wrapped.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
    wrapped.backbone.fc = ToyClassifier()
    maps = torch.arange(2 * 512, dtype=torch.float32).reshape(2, 512, 1, 1).expand(-1, -1, 7, 7)
    token = object()
    calls = []

    def no_cnn(images):
        assert images is token
        calls.append("predefined_feature_maps")
        return maps

    monkeypatch.setattr(wrapped, "_feature_maps", no_cnn)
    return wrapped, maps, token, calls


def test_logits_only_dispatch_skips_attention_without_executing_a_cnn(monkeypatch):
    wrapped, maps, token, calls = toy_dispatch(monkeypatch)
    wrapped.attention = ForbiddenAttention()
    wrapped.attention_heads = ForbiddenAttention()
    actual = wrapped.forward(token)
    expected = maps[:, :7, 0, 0] + 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 7)
    assert calls == ["predefined_feature_maps"]


def test_training_dispatch_returns_features_logits_and_pairwise_softmax_attention(monkeypatch):
    wrapped, maps, token, calls = toy_dispatch(monkeypatch)

    class ToyAttention(nn.Module):
        def forward(self, flattened):
            assert flattened.shape == (2, 512 * 7 * 7)
            return flattened[:, :64]

    class ToyHeads(nn.Module):
        def forward(self, embedding):
            assert embedding.shape == (2, 64)
            pairs = torch.tensor([0.0, math.log(3.0)]).expand(2, 512, 2).clone()
            return pairs.flatten(1)

    wrapped.attention = ToyAttention()
    wrapped.attention_heads = ToyHeads()
    features, logits, attention = wrapped.forward_with_attention(token)
    assert features.shape == attention.shape == (2, 512)
    assert logits.shape == (2, 7)
    torch.testing.assert_close(features, maps[:, :, 0, 0], rtol=0, atol=0)
    torch.testing.assert_close(logits, features[:, :7] + 1, rtol=0, atol=0)
    torch.testing.assert_close(attention, torch.full((2, 512), 0.75))
    assert calls == ["predefined_feature_maps"]


def test_attention_rejects_feature_maps_for_incompatible_image_size(monkeypatch):
    wrapped = meta_wrapper()
    monkeypatch.setattr(wrapped, "_feature_maps", lambda image: torch.empty(2, 512, 4, 4))
    wrapped.attention = ForbiddenAttention()
    with pytest.raises(ValueError, match="224x224"):
        wrapped.forward_with_attention(object())
