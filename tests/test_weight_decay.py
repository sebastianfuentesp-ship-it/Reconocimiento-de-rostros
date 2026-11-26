"""Optimizer-only regressions; no images, forward passes, downloads or GPU."""

import pytest
import torch
from torch import nn

from emotion_recognition.train import _configure_phase


def phase_model():
    model = nn.Module()
    model.layer1 = nn.Linear(2, 2, device="cpu", dtype=torch.float64)
    model.layer4 = nn.Linear(2, 2, device="cpu", dtype=torch.float64)
    model.fc = nn.Linear(2, 1, device="cpu", dtype=torch.float64)
    return model


@pytest.mark.parametrize("fine_tune", [False, True])
@pytest.mark.parametrize("weight_decay", [None, 1e-3], ids=["default", "experiment"])
def test_weight_decay_applies_to_each_phase_without_changing_trainable_layers(
    fine_tune, weight_decay,
):
    model = phase_model()
    # Switching either way must reset which blocks can receive updates.
    _configure_phase(model, not fine_tune, lr=1e-3, backbone_lr=1e-4)
    options = {} if weight_decay is None else {"weight_decay": weight_decay}
    optimizer = _configure_phase(
        model, fine_tune, lr=1e-3, backbone_lr=1e-4, **options,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    expected_decay = 1e-4 if weight_decay is None else weight_decay
    expected_modules = [model.fc, model.layer4] if fine_tune else [model.fc]
    assert len(optimizer.param_groups) == len(expected_modules)
    assert [group["lr"] for group in optimizer.param_groups] == (
        [1e-3, 1e-4] if fine_tune else [1e-3]
    )
    for group, module in zip(optimizer.param_groups, expected_modules):
        assert group["weight_decay"] == expected_decay
        assert {id(parameter) for parameter in group["params"]} == {
            id(parameter) for parameter in module.parameters()
        }
    trainable_prefixes = ("fc.", "layer4.") if fine_tune else ("fc.",)
    assert all(
        parameter.requires_grad == name.startswith(trainable_prefixes)
        for name, parameter in model.named_parameters()
    )


@pytest.mark.parametrize("fine_tune", [False, True])
def test_larger_weight_decay_shrinks_weights_more_with_zero_gradient(fine_tune):
    models = []
    for weight_decay in (1e-4, 1e-3):
        model = phase_model()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(2.0)
        optimizer = _configure_phase(
            model, fine_tune, lr=1e-3, backbone_lr=1e-4,
            weight_decay=weight_decay,
        )
        # Zero gradients isolate AdamW's decoupled decay from loss optimization.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()

        for group in optimizer.param_groups:
            expected = 2.0 * (1.0 - group["lr"] * weight_decay)
            for parameter in group["params"]:
                torch.testing.assert_close(
                    parameter, torch.full_like(parameter, expected), rtol=0, atol=1e-14,
                )
        for parameter in model.parameters():
            if not parameter.requires_grad:
                torch.testing.assert_close(
                    parameter, torch.full_like(parameter, 2.0), rtol=0, atol=0,
                )
        models.append(model)

    assert torch.all(models[1].fc.weight < models[0].fc.weight)
    if fine_tune:
        assert torch.all(models[1].layer4.weight < models[0].layer4.weight)
