"""Train-only class weights and CE mathematics; no model execution or training."""

import argparse
import json
import math

import pytest
import torch

from emotion_recognition import train_dacl_baseline as dacl
from emotion_recognition.constants import CLASS_NAMES

RAF_COUNTS = (986, 219, 561, 3596, 1530, 544, 1952)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def forbid_training_inference_downloads_and_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Estas pruebas solo validan configuración y matemática de pérdidas en CPU.")

    monkeypatch.setattr(dacl, "build_model", forbidden)
    monkeypatch.setattr(dacl, "_run_epoch", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)


def arguments(*extra):
    parser = argparse.ArgumentParser()
    dacl.add_parser(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args([
        "train-dacl-baseline", "--manifest", "unused.csv", "--data-root", "unused", "--selection-metric", "macro_f1", "--min-epochs", "40", *extra,
    ])


def rows_for(counts=RAF_COUNTS, split="train"):
    return [
        {"label": label, "class_name": CLASS_NAMES[label], "split": split}
        for label, count in enumerate(counts) for _ in range(count)
    ]


def make_config(tmp_path, args, train_rows=None, val_rows=None):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic manifest provenance", encoding="utf-8")
    return dacl._build_config(
        args, manifest, tmp_path,
        train_rows if train_rows is not None else rows_for(),
        val_rows if val_rows is not None else rows_for((1, 50, 2, 3, 4, 5, 6), "val"),
        {"source": "synthetic"}, torch.device("cpu"),
    )


def test_cli_keeps_original_baseline_unweighted_and_supports_balanced_option():
    assert arguments().class_weighting == "none"
    balanced = arguments("--class-weighting", "balanced")
    assert balanced.class_weighting == "balanced"
    assert balanced.label_smoothing == 0.0
    dacl._validate_args(balanced)
    with pytest.raises(SystemExit):
        arguments("--class-weighting", "oversample")


def test_actual_train_counts_produce_balanced_weights_in_raf_class_order():
    actual = dacl._balanced_class_weights(rows_for())
    assert isinstance(actual, list)
    assert len(actual) == len(CLASS_NAMES) == 7
    expected = [9388 / (7 * count) for count in RAF_COUNTS]
    assert actual == pytest.approx(expected, rel=1e-12)
    assert actual[1] == max(actual)  # fear is the least frequent train class
    assert actual[3] == min(actual)  # happiness is the most frequent train class
    assert all(isinstance(weight, float) and math.isfinite(weight) for weight in actual)


def test_balanced_weights_have_unit_population_mean_and_equal_total_class_mass():
    weights = dacl._balanced_class_weights(rows_for())
    masses = [count * weight for count, weight in zip(RAF_COUNTS, weights, strict=True)]
    assert masses == pytest.approx([9388 / 7] * 7, rel=1e-12)
    assert sum(masses) / 9388 == pytest.approx(1.0, rel=1e-12)
    # An unweighted mean across the seven class weights need not equal one.
    assert sum(weights) / 7 != pytest.approx(1.0)


def test_row_order_does_not_reorder_class_weights():
    rows = rows_for((2, 1, 4, 8, 3, 5, 6))
    expected = [29 / (7 * count) for count in (2, 1, 4, 8, 3, 5, 6)]
    assert dacl._balanced_class_weights(list(reversed(rows))) == pytest.approx(expected)
    assert dacl._balanced_class_weights(rows[7:] + rows[:7]) == pytest.approx(expected)


@pytest.mark.parametrize("split", ["val", "test", "unknown"])
def test_foreign_partition_cannot_enter_weight_estimation(split):
    rows = rows_for((1,) * 7)
    rows.append({"label": 1, "class_name": CLASS_NAMES[1], "split": split})
    with pytest.raises(ValueError):
        dacl._balanced_class_weights(rows)


@pytest.mark.parametrize("missing", range(7))
def test_missing_class_is_rejected_instead_of_producing_infinite_or_shifted_weights(missing):
    rows = [row for row in rows_for((1,) * 7) if row["label"] != missing]
    with pytest.raises(ValueError):
        dacl._balanced_class_weights(rows)


def test_empty_training_partition_is_rejected():
    with pytest.raises(ValueError):
        dacl._balanced_class_weights([])


@pytest.mark.parametrize("bad_label", [-1, 7, "bad", None, 1.5, True, False])
def test_invalid_label_is_rejected_before_loss_construction(bad_label):
    rows = rows_for((1,) * 7)
    rows.append({"label": bad_label, "class_name": CLASS_NAMES[0], "split": "train"})
    with pytest.raises(ValueError):
        dacl._balanced_class_weights(rows)


def test_balanced_criteria_require_explicit_train_weights():
    with pytest.raises(ValueError):
        dacl._build_criteria(arguments("--class-weighting", "balanced"))


@pytest.mark.parametrize("weights", [
    [], [1.0] * 6, [1.0] * 8,
    [0.0] + [1.0] * 6, [-1.0] + [1.0] * 6,
    [float("nan")] + [1.0] * 6, [float("inf")] + [1.0] * 6,
])
def test_balanced_criteria_reject_wrong_shape_or_nonpositive_nonfinite_weights(weights):
    with pytest.raises(ValueError):
        dacl._build_criteria(arguments("--class-weighting", "balanced"), weights)


def test_training_ce_weights_each_sample_without_batch_weight_renormalization():
    weights = dacl._balanced_class_weights(rows_for((2, 1, 4, 8, 3, 5, 6)))
    train_loss, val_loss = dacl._build_criteria(
        arguments("--class-weighting", "balanced"), weights, device="cpu",
    )
    assert train_loss.reduction == val_loss.reduction == "none"
    assert train_loss.weight.dtype == torch.float32
    assert train_loss.weight.device.type == "cpu"
    torch.testing.assert_close(train_loss.weight, torch.tensor(weights, dtype=torch.float32))
    assert val_loss.weight is None
    assert train_loss.label_smoothing == val_loss.label_smoothing == 0.0
    logits = torch.zeros((2, 7), dtype=torch.float32)
    logits[0, 1] = math.log(2)
    logits[1, 3] = math.log(3)
    targets = torch.tensor([1, 3])
    # Target probabilities are2/8 and3/9, respectively.
    raw = torch.tensor([math.log(4), math.log(3)])
    expected = raw * torch.tensor([weights[1], weights[3]])
    actual = train_loss(logits, targets)
    torch.testing.assert_close(actual, expected)
    assert actual.mean().item() == pytest.approx(expected.sum().item() / 2)
    batch_renormalized = expected.sum() / (weights[1] + weights[3])
    assert actual.mean().item() != pytest.approx(batch_renormalized.item())
    torch.testing.assert_close(val_loss(logits, targets), raw)
    assert logits.grad is None


def test_validation_loss_stays_raw_when_class_distribution_changes():
    args = arguments("--class-weighting", "balanced")
    weights_a = dacl._balanced_class_weights(rows_for((2, 1, 4, 8, 3, 5, 6)))
    weights_b = dacl._balanced_class_weights(rows_for((1, 8, 3, 2, 7, 9, 4)))
    train_a, val_a = dacl._build_criteria(args, weights_a, device=torch.device("cpu"))
    train_b, val_b = dacl._build_criteria(args, weights_b, device=torch.device("cpu"))
    logits = torch.zeros((2, 7))
    targets = torch.tensor([1, 3])
    torch.testing.assert_close(val_a(logits, targets), torch.full((2,), math.log(7)))
    torch.testing.assert_close(val_a(logits, targets), val_b(logits, targets), rtol=0, atol=0)
    assert not torch.allclose(train_a(logits, targets), train_b(logits, targets))


def test_optional_smoothing_retains_pytorch_weighted_target_semantics_only_in_train():
    args = arguments("--class-weighting", "balanced", "--label-smoothing", "0.05")
    weights = dacl._balanced_class_weights(rows_for((2, 1, 4, 8, 3, 5, 6)))
    train_loss, val_loss = dacl._build_criteria(args, weights, device="cpu")
    logits = torch.tensor([[1.0, -1.5, 2.0, -0.5, 0.0, 0.5, 1.25]])
    targets = torch.tensor([1])
    soft_target = torch.full((1, 7), 0.05 / 7)
    soft_target[0, 1] += 0.95
    log_probabilities = logits.log_softmax(1)
    expected = -(soft_target * torch.tensor(weights) * log_probabilities).sum(1)
    torch.testing.assert_close(train_loss(logits, targets), expected)
    torch.testing.assert_close(val_loss(logits, targets), -log_probabilities[:, 1])
    assert val_loss.weight is None and val_loss.label_smoothing == 0.0


def test_balanced_configuration_serializes_only_train_counts_and_correct_loss_definition(tmp_path):
    args = arguments("--class-weighting", "balanced")
    train_rows = rows_for()
    first = make_config(tmp_path, args, train_rows, rows_for((1,) * 7, "val"))
    second = make_config(tmp_path, args, train_rows, rows_for((30, 1, 2, 8, 9, 50, 100), "val"))
    expected = [9388 / (7 * count) for count in RAF_COUNTS]
    assert first["class_weighting"] == "balanced"
    assert first["class_weights"] == pytest.approx(expected)
    assert second["class_weights"] == first["class_weights"]
    assert first["class_counts_train"] == dict(zip(CLASS_NAMES, RAF_COUNTS, strict=True))
    assert first["train_label_smoothing"] == first["label_smoothing"] == 0.0
    assert first["validation_label_smoothing"] == 0.0
    assert first["test_used"] is False
    assert first["val_loss_definition"] == second["val_loss_definition"]
    definition = first["train_loss_definition"].lower()
    assert "peso" in definition or "ponder" in definition
    assert any("balanced" in item or "pesos" in item for item in first["deviations"])
    serialized = json.loads(json.dumps(first))
    assert serialized["class_weighting"] == "balanced"
    assert serialized["class_weights"] == pytest.approx(expected)


def test_default_arguments_keep_original_unweighted_objective(tmp_path):
    args = arguments()
    train_loss, val_loss = dacl._build_criteria(args)
    assert train_loss.weight is None and val_loss.weight is None
    assert train_loss.label_smoothing == val_loss.label_smoothing == 0.0
    cfg = make_config(tmp_path, args)
    assert cfg["class_weighting"] == "none"
    assert cfg["class_weights"] is None
    assert cfg["optimizer"] == "SGD" and cfg["scheduler"] == "StepLR"
    assert cfg["weight_decay"] == 0.0005


def test_legacy_namespace_without_weighting_option_is_backward_compatible(tmp_path):
    args = arguments()
    del args.class_weighting
    train_loss, val_loss = dacl._build_criteria(args)
    assert train_loss.weight is None and val_loss.weight is None
    cfg = make_config(tmp_path, args)
    assert cfg["class_weighting"] == "none" and cfg["class_weights"] is None
