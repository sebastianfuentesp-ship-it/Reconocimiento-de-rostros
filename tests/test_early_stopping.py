"""Exercise plateau decisions without any dataset, neural network or training."""

import pytest

from emotion_recognition.early_stopping import EarlyStopping


def test_default_waits_for_both_lr_changes_and_eight_stale_epochs():
    monitor = EarlyStopping(min_epochs=40)
    for epoch in range(1, 48):
        assert monitor.update(.8 if epoch == 3 else .7, epoch) is False
    assert monitor.best_score == .8
    assert monitor.stale_epochs == 7
    assert monitor.update(.7, 48) is True
    assert monitor.stopped_epoch == 48


def test_new_global_best_restarts_patience_but_equal_score_does_not():
    monitor = EarlyStopping(patience=2, min_epochs=0)
    assert monitor.update(.7, 1) is False
    assert monitor.update(.7, 2) is False
    assert monitor.update(.8, 3) is False
    assert monitor.stale_epochs == 0
    assert monitor.update(.8, 4) is False
    assert monitor.update(.79, 5) is True
    assert monitor.best_score == .8
    assert monitor.stopped_epoch == 5


def test_patience_zero_disables_stopping_and_preserves_score_tracking():
    monitor = EarlyStopping(patience=0, min_epochs=0)
    for epoch in range(1, 80):
        assert monitor.update(.9 if epoch == 2 else .5, epoch) is False
    assert monitor.best_score == .9
    assert monitor.stopped_epoch is None


@pytest.mark.parametrize("field,value", [
    ("patience", -1), ("patience", True), ("patience", 1.5),
    ("min_epochs", -1), ("min_epochs", False), ("min_epochs", 2.5),
])
def test_invalid_controls_are_rejected(field, value):
    with pytest.raises(ValueError):
        EarlyStopping(**{field: value})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), True])
def test_invalid_validation_score_is_rejected(score):
    with pytest.raises(ValueError, match="finita"):
        EarlyStopping().update(score, 1)


@pytest.mark.parametrize("epoch", [0, -1, 1.5, True])
def test_invalid_epoch_is_rejected(epoch):
    with pytest.raises(ValueError, match="positivo"):
        EarlyStopping().update(.7, epoch)
