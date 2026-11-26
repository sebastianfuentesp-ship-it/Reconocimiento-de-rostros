"""Phase configuration checks without images, forward/backward passes or GPU."""

import argparse
from itertools import product

import pytest
import torch
from torch import nn

from emotion_recognition.train import (
    _configure_phase,
    _make_scheduler,
    _set_training_mode,
    add_parser,
)


class PhaseModel(nn.Module):
    """Small CPU modules with BatchNorm in every stage; never run a forward."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(2, 2, device="cpu")
        self.bn1 = nn.BatchNorm1d(2, device="cpu")
        for stage in range(1, 5):
            setattr(self, f"layer{stage}", nn.Sequential(
                nn.Linear(2, 2, device="cpu"), nn.BatchNorm1d(2, device="cpu"),
            ))
        self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2, 1, device="cpu"))

    def forward(self, inputs):
        raise AssertionError("These configuration tests must never run model inference.")


def expected_stages(fine_tune, fine_tune_from):
    stages = ["fc"]
    if fine_tune:
        stages.append("layer4")
        if fine_tune_from == "layer3":
            stages.append("layer3")
    return stages


def assert_phase(model, optimizer, fine_tune, fine_tune_from):
    stages = expected_stages(fine_tune, fine_tune_from)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == len(stages)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-3, 1e-4, 1e-5][:len(stages)],
    )
    seen = []
    for group, stage in zip(optimizer.param_groups, stages):
        actual = [id(parameter) for parameter in group["params"]]
        expected = {id(parameter) for parameter in getattr(model, stage).parameters()}
        assert set(actual) == expected
        assert len(actual) == len(expected)
        assert group["weight_decay"] == 1e-3
        seen.extend(actual)
    assert len(seen) == len(set(seen))
    trainable_prefixes = tuple(f"{stage}." for stage in stages)
    assert all(
        parameter.requires_grad == name.startswith(trainable_prefixes)
        for name, parameter in model.named_parameters()
    )
    assert set(seen) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    "fine_tune,fine_tune_from", tuple(product((False, True), ("layer4", "layer3"))),
)
def test_parameter_groups_and_freezing_match_each_phase(fine_tune, fine_tune_from):
    model = PhaseModel()
    optimizer = _configure_phase(
        model, fine_tune, lr=1e-3, backbone_lr=1e-4, weight_decay=1e-3,
        fine_tune_from=fine_tune_from, layer3_lr=1e-5,
    )
    assert_phase(model, optimizer, fine_tune, fine_tune_from)


def test_phase_transitions_reset_freezing_and_keep_each_parameter_in_one_group():
    model = PhaseModel()
    for fine_tune, fine_tune_from in (
        (False, "layer3"), (True, "layer4"), (True, "layer3"),
        (False, "layer4"), (True, "layer3"), (True, "layer4"),
    ):
        optimizer = _configure_phase(
            model, fine_tune, lr=1e-3, backbone_lr=1e-4, weight_decay=1e-3,
            fine_tune_from=fine_tune_from, layer3_lr=1e-5,
        )
        assert_phase(model, optimizer, fine_tune, fine_tune_from)


def test_explicit_layer3_learning_rate_is_independent_of_layer4():
    optimizer = _configure_phase(
        PhaseModel(), True, lr=2e-3, backbone_lr=4e-4,
        fine_tune_from="layer3", layer3_lr=2e-5,
    )
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [2e-3, 4e-4, 2e-5],
    )


@pytest.mark.parametrize(
    "training,fine_tune,fine_tune_from",
    tuple(product((False, True), (False, True), ("layer4", "layer3"))),
)
def test_modes_enable_batchnorm_only_in_selected_training_stages(
    training, fine_tune, fine_tune_from,
):
    model = PhaseModel()
    _configure_phase(
        model, fine_tune, lr=1e-3, backbone_lr=1e-4,
        fine_tune_from=fine_tune_from,
    )
    trainable_before = [parameter.requires_grad for parameter in model.parameters()]
    model.train()
    _set_training_mode(model, training, fine_tune, fine_tune_from)

    active_stages = expected_stages(fine_tune, fine_tune_from) if training else []
    prefixes = tuple(f"{stage}." for stage in active_stages)
    assert model.training is False
    for name, module in model.named_modules():
        expected = name in active_stages or (bool(prefixes) and name.startswith(prefixes))
        assert module.training == expected, name or "model"
    assert [parameter.requires_grad for parameter in model.parameters()] == trainable_before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_validation_and_head_warmup_reset_modes_after_layer3_fine_tuning():
    model = PhaseModel()
    _set_training_mode(model, True, True, "layer3")
    assert model.layer3[1].training
    assert model.layer4[1].training

    _set_training_mode(model, False, True, "layer3")
    assert all(not module.training for module in model.modules())

    _set_training_mode(model, True, False, "layer3")
    assert model.fc.training
    assert not model.layer3.training
    assert not model.layer4.training
    assert all(
        not module.training for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )


def parser():
    result = argparse.ArgumentParser()
    add_parser(result.add_subparsers(dest="command", required=True))
    return result


def test_cli_preserves_layer4_defaults_and_accepts_layer3_experiment():
    cli = parser()
    required = ["train", "--manifest", "unused.csv", "--data-root", "unused"]
    default = cli.parse_args(required)
    assert default.fine_tune_from == "layer4"
    assert (default.lr, default.backbone_lr, default.layer3_lr) == (1e-3, 1e-4, 1e-5)
    assert default.weight_decay == 1e-4

    experiment = cli.parse_args(required + [
        "--fine-tune-from", "layer3", "--layer3-lr", "0.00002", "--weight-decay", "0.001",
    ])
    assert experiment.fine_tune_from == "layer3"
    assert experiment.layer3_lr == 2e-5
    assert experiment.weight_decay == 1e-3
    assert experiment.lr == default.lr
    assert experiment.backbone_lr == default.backbone_lr
    with pytest.raises(SystemExit):
        cli.parse_args(required + ["--fine-tune-from", "layer2"])


def test_scheduler_reduces_three_groups_together_and_preserves_learning_rate_ratios():
    model = PhaseModel()
    optimizer = _configure_phase(
        model, True, lr=1e-3, backbone_lr=1e-4,
        fine_tune_from="layer3", layer3_lr=1e-5,
    )
    args = argparse.Namespace(
        scheduler="plateau", scheduler_factor=0.3, scheduler_patience=2,
        scheduler_threshold=0.001, min_lr=1e-7,
    )
    scheduler = _make_scheduler(optimizer, args, fine_tune=True)
    for score in (0.60, 0.59, 0.59):
        scheduler.step(score)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-4, 1e-5]
    scheduler.step(0.59)
    rates = [group["lr"] for group in optimizer.param_groups]
    assert rates == pytest.approx([3e-4, 3e-5, 3e-6])
    assert rates[0] / rates[1] == pytest.approx(10)
    assert rates[1] / rates[2] == pytest.approx(10)
    assert all(parameter.grad is None for parameter in model.parameters())
