"""DACL wiring and metadata tests; training, CNN forwards and CUDA are forbidden."""

import argparse
import csv
import hashlib
import json

import pytest
import torch
from torch import nn

from emotion_recognition import train_dacl_full as dacl
from emotion_recognition.constants import CLASS_NAMES
from emotion_recognition.train import read_manifest


@pytest.fixture(autouse=True)
def forbid_training_inference_downloads_and_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("These tests must not train, infer, download or initialize CUDA.")

    monkeypatch.setattr(nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    monkeypatch.setattr(dacl, "_run_train_epoch", forbidden)
    monkeypatch.setattr(dacl, "_run_epoch", forbidden)
    monkeypatch.setattr(dacl, "build_dacl_model", forbidden)
    monkeypatch.setattr(dacl, "ManifestDataset", forbidden)
    monkeypatch.setattr(dacl, "_seed_everything", forbidden)


def parse_args(*extra):
    parser = argparse.ArgumentParser()
    dacl.add_parser(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args([
        "train-dacl-full", "--manifest", "unused.csv", "--data-root", "unused", "--selection-metric", "macro_f1", "--min-epochs", "40", *extra,
    ])


@pytest.fixture
def manifest_data(tmp_path):
    rows = [
        {"image_path": f"{split}/{label}.jpg", "label": label, "class_name": name,
         "split": split, "group_id": f"{split}-{label}"}
        for split in ("train", "val", "test")
        for label, name in enumerate(CLASS_NAMES)
    ]
    path = tmp_path / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_rows, val_rows = dacl._split_rows(read_manifest(path))
    return path, train_rows, val_rows


def make_config(manifest_data, args=None):
    path, train_rows, val_rows = manifest_data
    initialization = {"source": "msceleb_dacl", "sha256": "test-digest", "path": "base.pth"}
    return dacl._build_config(
        args or parse_args(), path, path.parent, train_rows, val_rows,
        initialization, torch.device("cpu"),
    )


def tiny_modules():
    model = nn.Module()
    model.backbone = nn.Linear(2, len(CLASS_NAMES), device="cpu")
    model.attention = nn.Linear(2, 2, device="cpu")
    centers = nn.Module()
    centers.register_buffer("centers", torch.zeros(len(CLASS_NAMES), 512, device="cpu"))
    return model, centers


def test_cli_exposes_dacl_recipe_without_optional_regularization_or_amp():
    args = parse_args()
    assert args.func is dacl.train
    assert (args.epochs, args.batch_size, args.num_workers, args.seed) == (60, 32, 0, 42)
    assert (args.lr, args.momentum, args.weight_decay) == (0.01, 0.9, 0.0005)
    assert (args.step_size, args.gamma, args.alpha, args.center_lambda) == (20, 0.1, 0.5, 0.01)
    assert args.output_dir == "artifacts/dacl_full"
    assert args.selection_metric == "macro_f1"
    assert args.device == "auto"
    assert args.pretrained_weights is None
    assert args.overwrite is False
    assert (args.patience, args.min_epochs) == (8, 40)
    for unsupported in ("amp", "label_smoothing", "class_weighting", "warmup_epochs", "split"):
        assert not hasattr(args, unsupported)
    dacl._validate_args(args)
    for flag in ("--amp", "--label-smoothing", "--class-weighting", "--split"):
        with pytest.raises(SystemExit):
            parse_args(flag)


@pytest.mark.parametrize("field,value", [
    ("epochs", 0), ("batch_size", 1), ("num_workers", -1),
    ("lr", 0), ("lr", float("nan")), ("lr", float("inf")),
    ("momentum", -0.1), ("momentum", 1), ("weight_decay", -1),
    ("weight_decay", float("inf")), ("step_size", 0), ("gamma", 0), ("gamma", 1),
    ("alpha", -0.01), ("alpha", 0), ("alpha", 1.01), ("alpha", float("nan")),
    ("center_lambda", -0.01), ("center_lambda", float("inf")),
    ("seed", -1), ("seed", 2**32), ("selection_metric", "test_accuracy"),
    ("patience", -1), ("patience", True), ("patience", 1.5),
    ("min_epochs", -1), ("min_epochs", True), ("min_epochs", 1.5),
])
def test_invalid_options_are_rejected_without_execution(field, value):
    args = parse_args()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        dacl._validate_args(args)


def test_training_batchnorm_singleton_is_rejected_without_dropping_samples(manifest_data):
    _, train_rows, val_rows = manifest_data
    assert all(row["split"] == "train" for row in train_rows)
    assert all(row["split"] == "val" for row in val_rows)
    with pytest.raises(ValueError, match="BatchNorm1d"):
        dacl._validate_train_batches(train_rows, batch_size=2)
    dacl._validate_train_batches(train_rows, batch_size=4)
    dacl._validate_train_batches([{}] * 9388, batch_size=32)
    assert len(train_rows) == len(val_rows) == 7


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step.*:UserWarning")
def test_sgd_includes_attention_and_backbone_without_training_or_changing_centers():
    model, centers = tiny_modules()
    model.requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    center_before = centers.centers.clone()
    optimizer, scheduler = dacl._configure_optimizer(model, parse_args())
    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert (group["lr"], group["momentum"], group["weight_decay"]) == (0.01, 0.9, 0.0005)
    assert {id(parameter) for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters()
    }
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert list(centers.parameters()) == []
    rates = []
    for _ in range(60):
        rates.append(group["lr"])
        scheduler.step()
    assert rates == pytest.approx([0.01] * 20 + [0.001] * 20 + [0.0001] * 20)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.testing.assert_close(centers.centers, center_before, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_config_records_full_method_distinct_losses_and_protocol(manifest_data):
    config = make_config(manifest_data)
    assert config["architecture"] == "dacl_resnet18"
    assert config["training_method"] == "dacl_full"
    assert config["initialization"]["sha256"] == "test-digest"
    assert config["manifest_sha256"] == hashlib.sha256(manifest_data[0].read_bytes()).hexdigest()
    assert config["class_counts_train"] == dict.fromkeys(CLASS_NAMES, 1)
    assert config["train_samples"] == config["val_samples"] == 7
    assert config["class_weights"] is None
    assert config["label_smoothing"] == 0
    assert config["alpha"] == 0.5 and config["center_lambda"] == 0.01
    assert config["loss"] == "cross_entropy_plus_sparse_center_loss"
    assert "center_lambda" in config["train_loss_definition"]
    assert "train_ce_loss" in config["loss_comparison"]
    assert "attention" in config["fine_tune_stages"]
    assert config["amp_enabled"] is False and config["precision"] == "float32"
    assert config["early_stopping"] is True and config["warmup_epochs"] == 0
    assert config["early_stopping_metric"] == "val_macro_f1"
    assert config["early_stopping_patience"] == 8
    assert config["early_stopping_min_epochs"] == 40
    assert config["drop_last"] is False and config["test_used"] is False
    assert config["selection_metric"] == "val_macro_f1"
    assert config["preprocessing_profile"] == "dacl_baseline" and config["image_size"] == 224
    assert config["normalization"]["mean"] == [0.5752, 0.4495, 0.4012]
    assert config["normalization"]["std"] == [0.2086, 0.1911, 0.1827]
    assert config["source_commit"] in config["source_url"]
    deviations = " ".join(config["deviations"])
    assert "Batch size 32" in deviations and "F1 macro" in deviations
    assert "test del proyecto permanece reservado" in deviations
    assert json.loads(json.dumps(config))["training_method"] == "dacl_full"


@pytest.mark.parametrize("selection", ["macro_f1", "accuracy"])
def test_training_and_photo_checkpoints_preserve_selection_and_compatible_architecture(
    manifest_data, tmp_path, selection,
):
    args = parse_args("--selection-metric", selection)
    config = make_config(manifest_data, args)
    model, centers = tiny_modules()
    optimizer, scheduler = dacl._configure_optimizer(model, args)
    metrics = {"loss": 1.1, "accuracy": 0.82, "macro_f1": 0.71}
    full = dacl._checkpoint_payload(model, centers, config, 9, metrics, optimizer, scheduler)
    inference = dacl._inference_payload(model, full)
    assert full["architecture"] == full["config"]["architecture"] == "dacl_resnet18"
    assert set(full["model_state_dict"]) == set(model.state_dict())
    assert set(full["center_state_dict"]) == {"centers"}
    assert inference["architecture"] == inference["config"]["architecture"] == "resnet18"
    assert inference["training_architecture"] == "dacl_resnet18"
    assert set(inference["model_state_dict"]) == set(model.backbone.state_dict())
    assert "center_state_dict" not in inference
    assert "optimizer_state_dict" not in inference
    assert "scheduler_state_dict" not in inference
    assert full["config"]["architecture"] == "dacl_resnet18"  # Export must not mutate training config.
    for index, payload in enumerate((full, inference)):
        assert payload["training_method"] == "dacl_full"
        assert payload["class_names"] == list(CLASS_NAMES)
        assert payload["preprocessing_profile"] == "dacl_baseline"
        assert payload["val_macro_f1"] == 0.71 and payload["val_accuracy"] == 0.82
        assert payload["selection_metric"] == f"val_{selection}"
        assert payload["selection_score"] == metrics[selection]
        path = tmp_path / f"metadata-{index}.pt"
        torch.save(payload, path)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        assert restored["config"]["manifest_sha256"] == config["manifest_sha256"]


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step.*:UserWarning")
def test_stubbed_epoch_control_saves_paired_exports_only_on_f1_improvement(
    manifest_data, tmp_path, monkeypatch,
):
    args = parse_args("--epochs", "3", "--step-size", "1")
    model, centers = tiny_modules()
    optimizer, scheduler = dacl._configure_optimizer(model, args)
    calls = []
    train_metrics = {"loss": 1.2, "ce_loss": 1.1, "center_loss": 10.0,
                     "accuracy": 0.7, "macro_f1": 0.6}
    validation = iter([
        {"loss": 1.0, "accuracy": 0.8, "macro_f1": 0.6},
        {"loss": 0.9, "accuracy": 0.9, "macro_f1": 0.5},
        {"loss": 0.8, "accuracy": 0.85, "macro_f1": 0.7},
    ])

    def train_stub(model_arg, loader, criterion, center_arg, device, opt, **kwargs):
        assert model_arg is model and center_arg is centers and opt is optimizer
        assert loader == "train-placeholder" and device.type == "cpu"
        assert criterion.reduction == "mean" and criterion.weight is None
        assert criterion.label_smoothing == 0
        assert kwargs == {"center_lambda": 0.01, "alpha": 0.5}
        calls.append("train-stub")
        return train_metrics

    def validation_stub(model_arg, loader, criterion, device, **kwargs):
        assert model_arg is model and loader == "val-placeholder"
        assert criterion.reduction == "none" and criterion.weight is None
        assert criterion.label_smoothing == 0 and kwargs == {"amp": False}
        calls.append("val-stub")
        return next(validation)

    saved = []
    real_save = torch.save

    def save_record(payload, path):
        saved.append((path.name, payload["epoch"]))
        real_save(payload, path)

    monkeypatch.setattr(dacl, "_run_train_epoch", train_stub)
    monkeypatch.setattr(dacl, "_run_epoch", validation_stub)
    monkeypatch.setattr(torch, "save", save_record)
    score = dacl._fit_epochs(
        args, model, centers, "train-placeholder", "val-placeholder", optimizer, scheduler,
        make_config(manifest_data, args), tmp_path, torch.device("cpu"),
    )
    assert score == 0.7
    assert calls == ["train-stub", "val-stub"] * 3
    assert saved == [("best.pt.tmp", 1), ("inference.pt.tmp", 1),
                     ("best.pt.tmp", 3), ("inference.pt.tmp", 3)]
    with (tmp_path / "history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == 3
    assert [float(row["lr_head"]) for row in history] == pytest.approx([0.01, 0.001, 0.0001])
    assert all(row["train_ce_loss"] == "1.1" and row["train_loss"] == "1.2" for row in history)
    assert all(row["phase"] == "fine_tune_all" and row["lr_layer4"] == "" for row in history)
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == 3
    assert torch.load(tmp_path / "inference.pt", weights_only=True)["epoch"] == 3
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_tensor_guard_rejects_bad_values_without_model_execution(value):
    with pytest.raises(FloatingPointError, match="features"):
        dacl._ensure_finite(features=torch.tensor([0.0, value], device="cpu"))
    dacl._ensure_finite(features=torch.tensor([0.0, 1.0], device="cpu"))


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step.*:UserWarning")
@pytest.mark.parametrize("extra,scores,expected_epochs,best_epoch,stopped", [
    ([], [.8] * 60, 48, 1, True),
    ([], [.8] * 44 + [.81] * 16, 53, 45, True),
    (["--epochs", "12", "--patience", "0", "--min-epochs", "0"],
     [.8] * 12, 12, 1, False),
])
def test_early_stopping_uses_simulated_metrics_and_preserves_best_pair(
    manifest_data, tmp_path, monkeypatch, extra, scores, expected_epochs, best_epoch, stopped,
):
    args = parse_args(*extra)
    model, centers = tiny_modules()
    optimizer, scheduler = dacl._configure_optimizer(model, args)
    training = {"loss": 1.2, "ce_loss": 1.1, "center_loss": 10.0,
                "accuracy": .9, "macro_f1": .9}
    validation = iter(scores)
    calls = {"train": 0, "val": 0}

    def train_stub(*args, **kwargs):
        calls["train"] += 1
        return training

    def val_stub(*args, **kwargs):
        calls["val"] += 1
        return {"loss": .7, "accuracy": .9, "macro_f1": next(validation)}

    monkeypatch.setattr(dacl, "_run_train_epoch", train_stub)
    monkeypatch.setattr(dacl, "_run_epoch", val_stub)
    best = dacl._fit_epochs(
        args, model, centers, "train-placeholder", "val-placeholder", optimizer, scheduler,
        make_config(manifest_data, args), tmp_path, torch.device("cpu"),
    )
    assert best == max(scores[:expected_epochs])
    assert calls == {"train": expected_epochs, "val": expected_epochs}
    with (tmp_path / "history.csv").open(encoding="utf-8", newline="") as f:
        history = list(csv.DictReader(f))
    assert len(history) == expected_epochs
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == best_epoch
    assert torch.load(tmp_path / "inference.pt", weights_only=True)["epoch"] == best_epoch
    summary = json.loads((tmp_path / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["requested_epochs"] == args.epochs
    assert summary["completed_epochs"] == expected_epochs
    assert summary["best_epoch"] == best_epoch
    assert summary["best_score"] == best
    assert summary["stopped_early"] is stopped
    assert summary["stop_reason"] == ("early_stopping" if stopped else "max_epochs_reached")
    assert summary["stopped_epoch"] == (expected_epochs if stopped else None)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_disabling_early_stopping_is_recorded_in_configuration(manifest_data):
    config = make_config(manifest_data, parse_args("--patience", "0"))
    assert config["early_stopping"] is False
    assert config["early_stopping_patience"] == 0
    assert not any("Parada temprana por" in note for note in config["deviations"])


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step.*:UserWarning")
def test_full_dacl_stops_on_loss_and_exports_same_best_epoch(manifest_data, tmp_path, monkeypatch):
    args = parse_args("--epochs", "6", "--selection-metric", "loss",
                      "--patience", "2", "--min-epochs", "0")
    model, centers = tiny_modules()
    optimizer, scheduler = dacl._configure_optimizer(model, args)
    validation = iter([
        {"loss": .5, "accuracy": .7, "macro_f1": .6},
        {"loss": .4, "accuracy": .8, "macro_f1": .7},
        {"loss": .45, "accuracy": .9, "macro_f1": .8},
        {"loss": .46, "accuracy": .95, "macro_f1": .9},
    ])
    monkeypatch.setattr(dacl, "_run_train_epoch", lambda *a, **k: {
        "loss": .1, "ce_loss": .09, "center_loss": 1., "accuracy": .99, "macro_f1": .99})
    monkeypatch.setattr(dacl, "_run_epoch", lambda *a, **k: next(validation))
    score = dacl._fit_epochs(args, model, centers, None, None, optimizer, scheduler,
                             make_config(manifest_data, args), tmp_path, torch.device("cpu"))
    assert score == .4
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["completed_epochs"] == 4 and summary["best_epoch"] == 2
    assert summary["selection_metric"] == "val_loss"
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == 2
    assert torch.load(tmp_path / "inference.pt", weights_only=True)["epoch"] == 2
