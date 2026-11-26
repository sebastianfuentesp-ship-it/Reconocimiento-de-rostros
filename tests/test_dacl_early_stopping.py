"""Simulate validation histories without images, network calls or optimizer updates."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from emotion_recognition import train_dacl_baseline as dacl
from emotion_recognition.constants import CLASS_NAMES


def arguments(tmp_path, *extra):
    parser = argparse.ArgumentParser()
    dacl.add_parser(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args([
        "train-dacl-baseline", "--manifest", str(tmp_path / "manifest.csv"),
        "--data-root", str(tmp_path / "raw"), "--output-dir", str(tmp_path / "run"),
        "--device", "cpu", "--selection-metric", "macro_f1", "--min-epochs", "40", *extra,
    ])


@pytest.fixture
def simulated_training(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No training, network execution, downloads or CUDA is allowed in this test.")

    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    (tmp_path / "manifest.csv").write_text("synthetic manifest fixture", encoding="utf-8")
    rows = [{"split": split, "label": i, "class_name": name}
            for split in ("train", "val", "test") for i, name in enumerate(CLASS_NAMES)]
    monkeypatch.setattr(dacl, "read_manifest", lambda path: rows)
    monkeypatch.setattr(dacl, "_seed_everything", lambda seed: None)
    monkeypatch.setattr(dacl, "resolve_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(dacl, "build_transforms", lambda **kwargs: None)
    monkeypatch.setattr(dacl, "ManifestDataset", lambda selected, root, transform: selected)
    monkeypatch.setattr(dacl, "DataLoader", lambda dataset, **kwargs: dataset)
    model = SimpleNamespace(initialization={"source": "synthetic"}, state_dict=dict)
    model.to = lambda device: model
    monkeypatch.setattr(dacl, "build_model", lambda **kwargs: model)
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.01}])
    state = {"epochs": 0, "saves": [], "scores": [], "validation_calls": 0}

    class Scheduler:
        def step(self):
            state["epochs"] += 1
            if state["epochs"] % 20 == 0:
                optimizer.param_groups[0]["lr"] *= 0.1

        def state_dict(self):
            return {"completed_steps": state["epochs"]}

    monkeypatch.setattr(dacl, "_configure_optimizer", lambda model, args: (optimizer, Scheduler()))

    def run_epoch(model, loader, criterion, device, **kwargs):
        split = "train" if kwargs.get("optimizer") is not None else "val"
        assert {row["split"] for row in loader} == {split}
        if split == "train":
            return {"loss": 0.1, "accuracy": 0.99, "macro_f1": 0.98}
        index = state["validation_calls"]
        state["validation_calls"] += 1
        return {"loss": state.get("losses", [0.5] * len(state["scores"]))[index], "accuracy": min(0.99, 0.80 + index * .01),
                "macro_f1": state["scores"][index]}

    def save(payload, path):
        state["saves"].append(payload)
        Path(path).write_text(str(payload["epoch"]), encoding="utf-8")

    monkeypatch.setattr(dacl, "_run_epoch", run_epoch)
    monkeypatch.setattr(dacl.torch, "save", save)
    return state


@pytest.mark.parametrize("variant", [[], ["--label-smoothing", ".05"],
                                     ["--class-weighting", "balanced"]])
def test_baseline_smoothing_and_weighting_stop_on_f1_and_keep_best_checkpoint(
    tmp_path, simulated_training, variant,
):
    state = simulated_training
    state["scores"] = [.70, .71, .71, .70, .69, .72]
    args = arguments(tmp_path, "--epochs", "6", "--patience", "2", "--min-epochs", "1", *variant)
    assert dacl.train(args) == 0
    output = tmp_path / "run"
    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert state["validation_calls"] == state["epochs"] == summary["completed_epochs"] == 4
    assert summary["requested_epochs"] == 6
    assert summary["best_epoch"] == 2 and summary["best_score"] == .71
    assert summary["stopped_early"] is True
    assert summary["stop_reason"] == "early_stopping" and summary["stopped_epoch"] == 4
    assert (summary["patience"], summary["min_epochs"]) == (2, 1)
    assert [payload["epoch"] for payload in state["saves"]] == [1, 2]
    assert (output / "best.pt").read_text() == "2"
    assert config["early_stopping"] is True
    assert config["early_stopping_metric"] == "val_macro_f1"
    with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == 4
    assert float(history[-1]["val_accuracy"]) > float(history[1]["val_accuracy"])
    assert float(history[-1]["val_macro_f1"]) < float(history[1]["val_macro_f1"])


def test_default_grace_reaches_both_scheduler_drops_before_early_stopping(tmp_path, simulated_training):
    state = simulated_training
    state["scores"] = [.80] * 60
    assert dacl.train(arguments(tmp_path)) == 0
    summary = json.loads((tmp_path / "run/training_summary.json").read_text(encoding="utf-8"))
    assert summary["completed_epochs"] == summary["stopped_epoch"] == 48
    assert summary["best_epoch"] == 1 and summary["stopped_early"] is True
    with (tmp_path / "run/history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    assert float(history[19]["lr_head"]) == pytest.approx(.01)
    assert float(history[20]["lr_head"]) == pytest.approx(.001)
    assert float(history[40]["lr_head"]) == pytest.approx(.0001)


def test_patience_zero_keeps_all_requested_epochs(tmp_path, simulated_training):
    state = simulated_training
    state["scores"] = [.80] * 6
    args = arguments(tmp_path, "--epochs", "6", "--patience", "0", "--min-epochs", "0")
    assert dacl.train(args) == 0
    summary = json.loads((tmp_path / "run/training_summary.json").read_text(encoding="utf-8"))
    config = json.loads((tmp_path / "run/config.json").read_text(encoding="utf-8"))
    assert summary["completed_epochs"] == 6
    assert summary["stopped_early"] is False
    assert summary["stop_reason"] == "max_epochs" and summary["stopped_epoch"] is None
    assert config["early_stopping"] is False


def test_existing_run_is_refused_without_replacing_its_config(tmp_path, simulated_training):
    output = tmp_path / "run"
    output.mkdir()
    original = b"original saved experiment"
    (output / "config.json").write_bytes(original)
    with pytest.raises(FileExistsError):
        dacl.train(arguments(tmp_path))
    assert (output / "config.json").read_bytes() == original
    assert simulated_training["validation_calls"] == 0


def test_epoch_limit_before_grace_is_recorded_without_fabricating_a_stop(tmp_path, simulated_training):
    simulated_training["scores"] = [.80] * 2
    assert dacl.train(arguments(tmp_path, "--epochs", "2")) == 0
    summary = json.loads((tmp_path / "run/training_summary.json").read_text(encoding="utf-8"))
    assert summary["completed_epochs"] == 2 and summary["best_epoch"] == 1
    assert summary["stopped_early"] is False and summary["stop_reason"] == "max_epochs"


@pytest.mark.parametrize("variant", [[], ["--label-smoothing", ".05"],
                                     ["--class-weighting", "balanced"]])
def test_loss_minimum_is_saved_even_when_classification_scores_keep_rising(
    tmp_path, simulated_training, variant,
):
    state = simulated_training
    state["scores"] = [.70, .75, .80, .85, .90, .95]
    state["losses"] = [.5, .4, .45, .46, .47, .48]
    args = arguments(tmp_path, "--epochs", "6", "--selection-metric", "loss",
                     "--min-epochs", "0", "--patience", "2", *variant)
    assert dacl.train(args) == 0
    summary = json.loads((tmp_path / "run/training_summary.json").read_text())
    assert summary["completed_epochs"] == 4
    assert summary["best_epoch"] == 2 and summary["best_score"] == .4
    assert summary["selection_metric"] == "val_loss"
    assert (tmp_path / "run/best.pt").read_text() == "2"


def test_actual_cli_defaults_select_minimum_validation_loss():
    from emotion_recognition import train_dacl_full
    for trainer, command in [(dacl, "train-dacl-baseline"), (train_dacl_full, "train-dacl-full")]:
        parser = argparse.ArgumentParser()
        trainer.add_parser(parser.add_subparsers(dest="command", required=True))
        args = parser.parse_args([command, "--manifest", "unused", "--data-root", "unused"])
        assert args.selection_metric == "loss"
        assert args.patience == 8 and args.min_epochs == 0
