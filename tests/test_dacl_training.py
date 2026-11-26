"""DACL baseline configuration and metadata checks; no training or inference."""

import argparse
import csv
import hashlib
import json

import pytest
import torch
from torch import nn

from emotion_recognition import train_dacl_baseline as dacl
from emotion_recognition.constants import CLASS_NAMES
from emotion_recognition.train import _set_training_mode, read_manifest


@pytest.fixture(autouse=True)
def no_execution_or_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Configuration tests must not train, infer, download or use CUDA.")

    monkeypatch.setattr(nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    monkeypatch.setattr(dacl, "_run_epoch", forbidden)
    monkeypatch.setattr(dacl, "build_model", forbidden)


def parse_args(*extra):
    parser = argparse.ArgumentParser()
    dacl.add_parser(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args([
        "train-dacl-baseline", "--manifest", "unused.csv", "--data-root", "unused", "--selection-metric", "macro_f1", "--min-epochs", "40", *extra,
    ])


@pytest.fixture
def manifest_data(tmp_path):
    rows = []
    for split in ("train", "val", "test"):
        for label, name in enumerate(CLASS_NAMES):
            rows.append({
                "image_path": f"{split}/{name}.jpg", "label": label,
                "class_name": name, "split": split,
                "group_id": f"{split}-{label}",
            })
    path = tmp_path / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_rows, val_rows = dacl._split_rows(read_manifest(path))
    return path, train_rows, val_rows


def make_config(manifest_data, args=None, device="cpu"):
    path, train_rows, val_rows = manifest_data
    initialization = {
        "source": "msceleb_dacl", "dataset": "MS-Celeb-1M (según autores de DACL)",
        "sha256": "test-digest", "path": str(path.parent / "weights.pth"),
    }
    return dacl._build_config(
        args or parse_args(), path, path.parent, train_rows, val_rows,
        initialization, torch.device(device),
    )


def test_cli_defaults_match_code_recipe_and_keep_amp_opt_in():
    args = parse_args()
    assert args.func is dacl.train
    assert args.device == "auto"
    assert (args.epochs, args.batch_size, args.num_workers, args.seed) == (60, 64, 0, 42)
    assert (args.lr, args.momentum, args.weight_decay) == (0.01, 0.9, 0.0005)
    assert (args.step_size, args.gamma) == (20, 0.1)
    assert args.selection_metric == "macro_f1"
    assert args.pretrained_weights is None
    assert args.amp is False
    assert args.overwrite is False
    assert args.output_dir == "artifacts/dacl_baseline"
    assert not hasattr(args, "warmup_epochs")
    assert args.patience == 8 and args.min_epochs == 40
    assert not hasattr(args, "split")
    dacl._validate_args(args)


def test_cli_supports_paper_batch_accuracy_selection_and_explicit_weights():
    args = parse_args(
        "--batch-size", "128", "--selection-metric", "accuracy",
        "--pretrained-weights", "base.pth", "--amp", "--device", "cuda", "--overwrite",
    )
    assert args.batch_size == 128
    assert args.selection_metric == "accuracy"
    assert args.pretrained_weights == "base.pth"
    assert args.amp and args.overwrite
    dacl._validate_args(args)
    with pytest.raises(SystemExit):
        parse_args("--selection-metric", "test_accuracy")


@pytest.mark.parametrize("field,value", [
    ("epochs", 0), ("batch_size", 1), ("num_workers", -1),
    ("lr", 0), ("lr", float("nan")), ("lr", float("inf")),
    ("momentum", -0.1), ("momentum", 1), ("momentum", float("nan")),
    ("weight_decay", -1), ("weight_decay", float("inf")),
    ("step_size", 0), ("gamma", 0), ("gamma", 1), ("gamma", float("nan")),
    ("seed", -1), ("seed", 2**32), ("selection_metric", "test_accuracy"),
    ("patience", -1), ("patience", True), ("patience", 1.5),
    ("min_epochs", -1), ("min_epochs", True), ("min_epochs", 2.5),
])
def test_invalid_configuration_is_rejected_before_execution(field, value):
    args = parse_args()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        dacl._validate_args(args)


def test_internal_splits_keep_all_rows_and_allow_final_singleton_batch(manifest_data):
    _, train_rows, val_rows = manifest_data
    assert len(train_rows) == len(val_rows) == len(CLASS_NAMES)
    assert all(row["split"] == "train" for row in train_rows)
    assert all(row["split"] == "val" for row in val_rows)
    args = parse_args("--batch-size", "2")
    assert len(train_rows) % args.batch_size == 1
    # ResNet18 at 224px has BatchNorm2d on 7x7 spatial maps, not BatchNorm1d.
    dacl._validate_args(args)
    for split in ("train", "val"):
        without_class = [row for row in train_rows + val_rows
                         if row["split"] != split or row["label"] != 0]
        with pytest.raises(ValueError, match=split):
            dacl._split_rows(without_class)


def test_shared_manifest_validation_still_rejects_cross_split_groups(manifest_data):
    path, _, _ = manifest_data
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]["group_id"] = rows[0]["group_id"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Fuga de datos"):
        read_manifest(path)


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step.*:UserWarning")
def test_full_sgd_and_step_scheduler_preserve_parameters_without_forward_or_backward():
    model = nn.Sequential(nn.Linear(2, 3, device="cpu"), nn.Linear(3, 7, device="cpu"))
    model.requires_grad_(False)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    optimizer, scheduler = dacl._configure_optimizer(model, parse_args())
    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert {id(parameter) for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters()
    }
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert (group["lr"], group["momentum"], group["weight_decay"]) == (0.01, 0.9, 0.0005)
    used_rates = []
    for _ in range(60):
        used_rates.append(group["lr"])
        scheduler.step()
    assert used_rates[:20] == pytest.approx([0.01] * 20)
    assert used_rates[20:40] == pytest.approx([0.001] * 20)
    assert used_rates[40:] == pytest.approx([0.0001] * 20)
    assert group["lr"] == pytest.approx(0.00001)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_all_training_mode_updates_every_batchnorm_and_validation_restores_eval():
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1, device="cpu"),
        nn.BatchNorm2d(4, device="cpu"),
        nn.Sequential(nn.Conv2d(4, 4, kernel_size=1, device="cpu"),
                      nn.BatchNorm2d(4, device="cpu")),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 7, device="cpu"),
    )
    next(model.parameters()).requires_grad_(False)
    requires_grad_before = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    _set_training_mode(model, training=True, fine_tune=True, fine_tune_from="all")
    assert model.training is True
    assert all(module.training for module in model.modules())
    assert all(module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d))
    assert [parameter.requires_grad for parameter in model.parameters()] == requires_grad_before
    _set_training_mode(model, training=False, fine_tune=True, fine_tune_from="all")
    assert all(not module.training for module in model.modules())
    assert [parameter.requires_grad for parameter in model.parameters()] == requires_grad_before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_configuration_records_recipe_provenance_and_protocol_deviations(manifest_data):
    config = make_config(manifest_data)
    path, train_rows, val_rows = manifest_data
    assert config["architecture"] == "resnet18"
    assert config["pretrained_source"] == "msceleb_dacl"
    assert config["initialization"]["sha256"] == "test-digest"
    assert config["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config["train_samples"] == len(train_rows)
    assert config["val_samples"] == len(val_rows)
    assert config["class_counts_train"] == dict.fromkeys(CLASS_NAMES, 1)
    assert config["class_weights"] is None
    assert config["loss"] == "cross_entropy"
    assert config["optimizer"] == "SGD"
    assert config["scheduler"] == "StepLR"
    assert config["warmup_epochs"] == 0
    assert config["early_stopping"] is True
    assert config["early_stopping_metric"] == "val_macro_f1"
    assert config["early_stopping_patience"] == 8
    assert config["early_stopping_min_epochs"] == 40
    assert config["gradient_clip"] is None
    assert config["test_used"] is False
    assert config["amp_enabled"] is False
    assert config["precision"] == "float32"
    assert config["selection_metric"] == "val_macro_f1"
    assert config["image_size"] == 224
    assert config["preprocessing_profile"] == "dacl_baseline"
    assert config["normalization"] == {
        "name": "dacl_baseline", "mean": [0.5752, 0.4495, 0.4012],
        "std": [0.2086, 0.1911, 0.1827],
    }
    assert config["paper_recipe"]["batch_size"] == 128
    assert config["batch_size"] == config["source_code_default_batch_size"] == 64
    assert config["source_commit"] in config["source_url"]
    deviations = " ".join(config["deviations"])
    assert "Batch size 64" in deviations
    assert "F1 macro" in deviations
    assert "test del proyecto permanece reservado" in deviations
    assert "atención" in deviations
    assert json.loads(json.dumps(config))["initialization"] == config["initialization"]


def test_configuration_records_overrides_and_effective_amp_without_cuda_allocation(manifest_data):
    args = parse_args(
        "--batch-size", "128", "--selection-metric", "accuracy", "--epochs", "2", "--amp",
    )
    config = make_config(manifest_data, args=args, device="cuda")
    assert config["amp_enabled"] is True
    assert config["selection_metric"] == "val_accuracy"
    deviations = " ".join(config["deviations"])
    assert "Batch size" not in deviations
    assert "epochs=2" in deviations
    assert "AMP activado" in deviations
    assert make_config(manifest_data, args=args, device="cpu")["amp_enabled"] is False


@pytest.mark.parametrize("selection", ["macro_f1", "accuracy"])
def test_checkpoint_preserves_f1_and_preprocessing_when_selection_changes(
    manifest_data, tmp_path, selection,
):
    config = make_config(manifest_data, args=parse_args("--selection-metric", selection))
    model = nn.Linear(2, 7, device="cpu")
    optimizer, scheduler = dacl._configure_optimizer(model, parse_args())
    metrics = {"loss": 1.1, "accuracy": 0.82, "macro_f1": 0.71}
    payload = dacl._checkpoint_payload(model, config, 9, metrics, optimizer, scheduler)
    assert payload["epoch"] == 9
    assert payload["architecture"] == "resnet18"
    assert payload["class_names"] == list(CLASS_NAMES)
    assert payload["image_size"] == 224
    assert payload["preprocessing_profile"] == "dacl_baseline"
    assert payload["val_macro_f1"] == 0.71
    assert payload["val_accuracy"] == 0.82
    assert payload["selection_metric"] == f"val_{selection}"
    assert payload["selection_score"] == metrics[selection]
    checkpoint = tmp_path / "metadata.pt"
    torch.save(payload, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert restored["config"]["manifest_sha256"] == config["manifest_sha256"]
    assert restored["config"]["initialization"] == config["initialization"]
    assert restored["scheduler_state_dict"]["step_size"] == 20
    assert set(restored["model_state_dict"]) == set(model.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())
