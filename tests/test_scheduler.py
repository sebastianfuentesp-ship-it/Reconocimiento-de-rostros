from argparse import Namespace

import pytest
import torch

from emotion_recognition.train import _make_scheduler


def arguments(scheduler="plateau"):
    return Namespace(scheduler=scheduler, scheduler_factor=0.3, scheduler_patience=2,
                     scheduler_threshold=0.001, min_lr=1e-7)


def optimizer():
    return torch.optim.AdamW([
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3},
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-4},
    ])


def test_plateau_reduces_both_groups_after_patience_and_preserves_ratio():
    opt = optimizer()
    scheduler = _make_scheduler(opt, arguments(), fine_tune=True)
    for score in (0.60, 0.59, 0.59):
        scheduler.step(score)
    assert [g["lr"] for g in opt.param_groups] == [1e-3, 1e-4]
    scheduler.step(0.59)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([3e-4, 3e-5])
    # A meaningful improvement resets the patience counter.
    scheduler.step(0.65)
    scheduler.step(0.64)
    scheduler.step(0.64)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([3e-4, 3e-5])


@pytest.mark.parametrize("name,fine_tune", [("none", True), ("plateau", False)])
def test_scheduler_does_not_affect_reference_or_head_warmup(name, fine_tune):
    assert _make_scheduler(optimizer(), arguments(name), fine_tune) is None
