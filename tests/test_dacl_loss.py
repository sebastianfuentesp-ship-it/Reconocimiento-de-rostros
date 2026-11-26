"""Tensor-only checks of DACL loss and center updates; no network or image use."""

import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from emotion_recognition.dacl_loss import SparseCenterLoss


@pytest.fixture(autouse=True)
def no_networks_downloads_or_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Only tiny CPU tensor calculations are allowed in these tests.")

    monkeypatch.setattr(nn.Linear, "forward", forbidden)
    monkeypatch.setattr(nn.Conv2d, "forward", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)


def example(requires_grad=False):
    loss = SparseCenterLoss(num_classes=3, feat_dim=2).double()
    with torch.no_grad():
        loss.centers.copy_(torch.tensor([[0, 0], [2, -1], [8, 9]], dtype=torch.float64))
    features = torch.tensor([[1, 2], [3, 1]], dtype=torch.float64, requires_grad=requires_grad)
    attention = torch.tensor([[0.25, 0.5], [0.75, 0.25]], dtype=torch.float64,
                             requires_grad=requires_grad)
    labels = torch.tensor([0, 1], dtype=torch.long)
    return loss, features, attention, labels


def test_loss_matches_hand_calculated_scalar_and_keeps_centers_out_of_optimizer():
    loss, features, attention, labels = example()
    # Weighted squared distances are 2.25 and 1.75; divide their sum by 2*N=4.
    assert loss(features, attention, labels).item() == 1.0
    assert list(loss.parameters()) == []
    assert dict(loss.named_buffers())["centers"] is loss.centers
    assert loss.centers.requires_grad is False


def test_initialization_matches_authors_transposed_kaiming_normal():
    torch.manual_seed(17)
    expected = torch.empty(3, 2)
    nn.init.kaiming_normal_(expected.t())
    torch.manual_seed(17)
    actual = SparseCenterLoss(3, 2)
    torch.testing.assert_close(actual.centers, expected, rtol=0, atol=0)


def test_combined_cross_entropy_and_sparse_loss_have_manual_feature_attention_gradients():
    loss, features, attention, labels = example(requires_grad=True)
    logits = torch.tensor([[1.0, -0.5, 0.2], [-0.2, 0.8, 0.3]], dtype=torch.float64,
                          requires_grad=True)
    coefficient = 0.01
    differences = features.detach() - loss.centers.index_select(0, labels)
    expected_features = coefficient * attention.detach() * differences / 2
    expected_attention = coefficient * differences.square() / 4
    expected_logits = (logits.detach().softmax(dim=1) - F.one_hot(labels, 3)) / 2
    (F.cross_entropy(logits, labels) + coefficient * loss(features, attention, labels)).backward()
    torch.testing.assert_close(features.grad, expected_features, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(attention.grad, expected_attention, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(logits.grad, expected_logits, rtol=1e-12, atol=1e-12)
    assert loss.centers.grad is None


def test_repeated_labels_use_one_plus_class_count_and_leave_absent_classes_unchanged():
    loss = SparseCenterLoss(3, 2).double()
    with torch.no_grad():
        loss.centers.copy_(torch.tensor([[0, 0], [1, 1], [9, 9]], dtype=torch.float64))
    features = torch.tensor([[2, 0], [4, 1], [-1, 2]], dtype=torch.float64, requires_grad=True)
    attention = torch.tensor([[1, 0.5], [0.5, 1], [0.25, 0.5]], dtype=torch.float64,
                             requires_grad=True)
    labels = torch.tensor([0, 0, 1], dtype=torch.int32)
    absent_before = loss.centers[2].clone()
    loss.update_centers(features, attention, labels, alpha=0.5)
    expected = torch.tensor([[2 / 3, 1 / 6], [0.875, 1.125], [9, 9]], dtype=torch.float64)
    torch.testing.assert_close(loss.centers, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(loss.centers[2], absent_before, rtol=0, atol=0)
    assert loss.centers.grad_fn is None
    assert features.grad is attention.grad is None


@pytest.mark.parametrize("coefficient", [0.0, 0.01, 1.0])
def test_explicit_center_updates_are_independent_of_loss_coefficient(coefficient):
    loss, features, attention, labels = example(requires_grad=True)
    original = loss.centers.clone()
    expected = original.clone()
    expected[0] -= 0.5 * torch.tensor([-0.25, -1.0], dtype=torch.float64) / 2
    expected[1] -= 0.5 * torch.tensor([-0.75, -0.5], dtype=torch.float64) / 2
    (coefficient * loss(features, attention, labels)).backward()
    # Features and attention remain the detached, pre-optimizer values from the batch.
    loss.update_centers(features.detach(), attention.detach(), labels, alpha=0.5)
    torch.testing.assert_close(loss.centers, expected, rtol=1e-12, atol=1e-12)
    assert loss.centers.grad is None


def test_single_sample_is_valid_for_loss_and_explicit_update():
    loss = SparseCenterLoss(2, 2).double()
    loss.centers.zero_()
    features = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    attention = torch.tensor([[0.5, 0.25]], dtype=torch.float64)
    labels = torch.tensor([1])
    assert loss(features, attention, labels).item() == 3.0
    loss.update_centers(features, attention, labels, alpha=1.0)
    torch.testing.assert_close(loss.centers, torch.tensor([[0, 0], [0.5, 0.5]], dtype=torch.float64))


def test_mixed_input_dtype_preserves_buffer_dtype_and_updates_without_gradients():
    loss, features, attention, labels = example()
    loss.float()
    assert loss(features, attention, labels).dtype == torch.float64
    loss.update_centers(features, attention, labels)
    assert loss.centers.dtype == torch.float32
    assert loss.centers.grad is None


def test_center_state_roundtrip_keeps_buffers_without_parameters():
    loss, features, attention, labels = example()
    loss.update_centers(features, attention, labels)
    stream = io.BytesIO()
    torch.save(loss.state_dict(), stream)
    stream.seek(0)
    restored = SparseCenterLoss(3, 2).double()
    restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True), strict=True)
    torch.testing.assert_close(restored.centers, loss.centers, rtol=0, atol=0)
    assert list(restored.parameters()) == []
    assert set(restored.state_dict()) == {"centers"}


@pytest.mark.parametrize("num_classes,feat_dim", [(0, 2), (3, 0), (-1, 2), (True, 2), (3, 2.5)])
def test_invalid_constructor_dimensions_fail(num_classes, feat_dim):
    with pytest.raises(ValueError, match="enteros positivos"):
        SparseCenterLoss(num_classes, feat_dim)


@pytest.mark.parametrize("alpha", [0, -0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_alpha_never_changes_centers(alpha):
    loss, features, attention, labels = example()
    before = loss.centers.clone()
    with pytest.raises(ValueError, match="alpha"):
        loss.update_centers(features, attention, labels, alpha=alpha)
    torch.testing.assert_close(loss.centers, before, rtol=0, atol=0)


@pytest.mark.parametrize("problem", [
    "feature_rank", "feature_width", "attention_shape", "labels_rank", "labels_length",
    "empty", "integer_features", "integer_attention", "float_labels", "negative_label",
    "large_label", "nan_features", "infinite_attention", "nan_centers", "device",
])
@pytest.mark.parametrize("operation", ["forward", "update_centers"])
def test_invalid_batch_is_rejected_by_both_loss_and_update(problem, operation):
    loss, features, attention, labels = example()
    if problem == "feature_rank":
        features = features.flatten()
    elif problem == "feature_width":
        features = torch.zeros(2, 3)
    elif problem == "attention_shape":
        attention = attention[:, :1]
    elif problem == "labels_rank":
        labels = labels[:, None]
    elif problem == "labels_length":
        labels = labels[:1]
    elif problem == "empty":
        features, attention, labels = features[:0], attention[:0], labels[:0]
    elif problem == "integer_features":
        features = features.long()
    elif problem == "integer_attention":
        attention = attention.long()
    elif problem == "float_labels":
        labels = labels.float()
    elif problem == "negative_label":
        labels[0] = -1
    elif problem == "large_label":
        labels[0] = 3
    elif problem == "nan_features":
        features[0, 0] = float("nan")
    elif problem == "infinite_attention":
        attention[0, 0] = float("inf")
    elif problem == "nan_centers":
        loss.centers[0, 0] = float("nan")
    elif problem == "device":
        # A meta tensor checks the guard without allocating any CUDA memory.
        features = torch.empty_like(features, device="meta")
    with pytest.raises(ValueError):
        getattr(loss, operation)(features, attention, labels)
