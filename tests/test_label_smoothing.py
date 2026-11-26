"""Loss mathematics and experiment isolation; no network execution or training."""

import argparse
import hashlib
import math

import pytest
import torch
from torch.nn import functional as F

from emotion_recognition import train_dacl_baseline as dacl
from emotion_recognition.constants import CLASS_NAMES


def arguments(*extra):
    parser = argparse.ArgumentParser()
    dacl.add_parser(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args([
        "train-dacl-baseline", "--manifest", "unused.csv", "--data-root", "unused", "--selection-metric", "macro_f1", "--min-epochs", "40", *extra,
    ])


def test_default_loss_is_exactly_the_existing_baseline():
    args = arguments()
    assert args.label_smoothing == 0.0
    train_loss, validation_loss = dacl._build_criteria(args)
    logits = torch.tensor([[1.0, -2, 0, 0.2, 0.3, 1.2, -0.5]], dtype=torch.float64)
    targets = torch.tensor([0])
    expected = F.cross_entropy(logits, targets, reduction="none")
    torch.testing.assert_close(train_loss(logits, targets), expected, rtol=0, atol=0)
    torch.testing.assert_close(validation_loss(logits, targets), expected, rtol=0, atol=0)


def test_smoothing_mixes_the_training_target_with_uniform_distribution_only():
    train_loss, validation_loss = dacl._build_criteria(arguments("--label-smoothing", "0.05"))
    logits = torch.tensor([[4.0, -2, -1, 0, -0.5, 1, 0.5],
                           [0.5, 2.0, -0.5, 3.0, 0.4, -1, -2]], dtype=torch.float64)
    targets = torch.tensor([0, 3])
    log_probabilities = logits.log_softmax(dim=1)
    hard_loss = -log_probabilities[torch.arange(2), targets]
    expected_train_loss = 0.95 * hard_loss - 0.05 * log_probabilities.mean(dim=1)
    actual_train_loss = train_loss(logits, targets)
    torch.testing.assert_close(actual_train_loss, expected_train_loss)
    torch.testing.assert_close(validation_loss(logits, targets), hard_loss)
    assert train_loss.weight is None and validation_loss.weight is None
    assert validation_loss.label_smoothing == 0.0
    assert logits.grad is None


def test_smoothed_objective_penalizes_extreme_certainty_even_when_argmax_is_correct():
    train_loss, validation_loss = dacl._build_criteria(arguments("--label-smoothing", "0.05"))
    optimal = torch.full((1, 7), 0.05 / 7, dtype=torch.float64)
    optimal[0, 0] += 0.95
    extreme = torch.full((1, 7), 0.0001 / 6, dtype=torch.float64)
    extreme[0, 0] = 0.9999
    label = torch.tensor([0])
    assert train_loss(extreme.log(), label).item() > train_loss(optimal.log(), label).item()
    assert validation_loss(extreme.log(), label).item() < validation_loss(optimal.log(), label).item()
    assert extreme.argmax(1).item() == optimal.argmax(1).item() == 0


@pytest.mark.parametrize("value", ["-0.1", "1", "2", "nan", "inf"])
def test_invalid_smoothing_is_rejected_before_data_or_model_execution(value):
    with pytest.raises(ValueError, match="label-smoothing"):
        dacl._validate_args(arguments("--label-smoothing", value))


def test_smoothing_is_recorded_as_additional_regularization_with_comparable_validation(tmp_path):
    args = arguments("--label-smoothing", "0.05")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic provenance fixture", encoding="utf-8")
    rows = [{"label": index, "class_name": name} for index, name in enumerate(CLASS_NAMES)]
    cfg = dacl._build_config(
        args, manifest, tmp_path, rows, rows, {"source": "synthetic"}, torch.device("cpu"),
    )
    assert cfg["train_label_smoothing"] == cfg["label_smoothing"] == 0.05
    assert cfg["validation_label_smoothing"] == 0.0
    assert cfg["selection_metric"] == "val_macro_f1"
    assert cfg["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert cfg["class_weights"] is None
    assert math.isclose(cfg["weight_decay"], 0.0005)
    assert cfg["optimizer"] == "SGD" and cfg["scheduler"] == "StepLR"
    assert any("label_smoothing=0.05" in text for text in cfg["deviations"])
