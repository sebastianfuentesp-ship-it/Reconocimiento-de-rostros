"""A new split must not be compared with an older split of the same size."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "guardar_resultados.py"
SPEC = importlib.util.spec_from_file_location("resultados_protocolos", SCRIPT)
resultados = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resultados)


def test_loss_checkpoint_selection_keeps_early_minimum_despite_later_higher_f1():
    history = [
        {"epoch": 1, "val_loss": .5, "val_macro_f1": .7},
        {"epoch": 2, "val_loss": .4, "val_macro_f1": .75},
        {"epoch": 3, "val_loss": .6, "val_macro_f1": .9},
    ]
    assert resultados.pick_epoch(history, "val_loss")["epoch"] == 2
    assert resultados.pick_epoch(history, "val_macro_f1")["epoch"] == 3


def run(name, manifest, f1, *, smoothing=0.0, samples=2350, split="val"):
    return {
        "name": name, "variant": name, "run_path": name,
        "model_kind": "resnet18", "technical_run": False,
        "duplicate_saved_results_of": None,
        "manifest_sha256": manifest, "config_manifest_sha256": manifest,
        "samples": samples, "split": split,
        "accuracy": f1, "macro_f1": f1, "balanced_accuracy": f1,
        "saved_metrics_match_history_selection": True,
        "configuration": {"preprocessing_profile": "dacl_baseline",
                          "train_label_smoothing": smoothing},
        "selected_checkpoint_epoch_from_history": 1, "best_macro_f1_epoch": 1,
        "checkpoint": {"path": None, "sha256_current_file": None,
                       "matches_evaluation_hash": True},
        "calibration_saved_analysis": None,
    }


def report_for(monkeypatch, tmp_path, runs, analyses=None):
    # No checkpoint files or hashes are read: all inputs are synthetic records.
    monkeypatch.setattr(resultados, "collect_runs", lambda path: (runs, []))
    monkeypatch.setattr(resultados, "collect_saved_analyses", lambda items: analyses or [])
    return resultados.build_report(tmp_path)


def test_different_manifests_with_same_sample_count_are_not_compared(monkeypatch, tmp_path):
    old = run("old_smoothing", "a" * 64, .73, smoothing=.05)
    current = run("new_baseline", "b" * 64, .90)
    report = report_for(monkeypatch, tmp_path, [old, current])
    text = resultados.render_text(report)
    assert report["classification_reference"] == "new_baseline"
    assert report["comparable_cnn_ranking_by_macro_f1"] == ["new_baseline"]
    assert [item["name"] for item in report["runs"]] == ["old_smoothing", "new_baseline"]
    assert "BASELINE DACL FRENTE A SUAVIZADO" not in text
    assert "Cambio suavizado - baseline" not in text
    assert f"Grupo: manifest SHA256={'b' * 64}; partición=val; muestras=2350." in text
    assert "Otros manifiestos, particiones o tamaños de evaluación no se comparan" in text


def test_smoothing_comparison_selects_same_group_not_first_saved_run(monkeypatch, tmp_path):
    runs = [run("old_smoothing", "a" * 64, .73, smoothing=.05),
            run("new_baseline", "b" * 64, .90),
            run("new_smoothing", "b" * 64, .88, smoothing=.05)]
    old_pair = {"data": {"results": [{"run": "old_baseline"}, {"run": "old_smoothing"}],
                         "paired_predictions": {"fixed": 999, "lost": 888}}}
    report = report_for(monkeypatch, tmp_path, runs, [old_pair])
    text = resultados.render_text(report)
    assert report["comparable_cnn_ranking_by_macro_f1"] == ["new_baseline", "new_smoothing"]
    assert "BASELINE DACL FRENTE A SUAVIZADO 0.05" in text
    assert "Suavizado: accuracy=88.0000%; F1=0.880000." in text
    assert "Cambio suavizado - baseline: -2.0000 puntos de accuracy y -0.020000" in text
    assert "Comparación pareada guardada" not in text


@pytest.mark.parametrize("changed", [{"samples": 2351}, {"split": "test"}])
def test_same_manifest_different_evaluation_is_not_compared(monkeypatch, tmp_path, changed):
    baseline = run("baseline", "a" * 64, .80)
    smooth = run("smoothing", "a" * 64, .79, smoothing=.05, **changed)
    report = report_for(monkeypatch, tmp_path, [baseline, smooth])
    assert report["comparable_cnn_ranking_by_macro_f1"] == ["baseline"]
    assert "BASELINE DACL FRENTE A SUAVIZADO" not in resultados.render_text(report)


@pytest.mark.parametrize("invalid_field", ["saved_metrics_match_history_selection", "checkpoint_hash"])
def test_known_invalid_smoothing_result_is_not_compared(monkeypatch, tmp_path, invalid_field):
    baseline = run("baseline", "a" * 64, .80)
    smooth = run("invalid_smoothing", "a" * 64, .90, smoothing=.05)
    if invalid_field == "checkpoint_hash":
        smooth["checkpoint"]["matches_evaluation_hash"] = False
    else:
        smooth[invalid_field] = False
    report = report_for(monkeypatch, tmp_path, [baseline, smooth])
    assert report["comparable_cnn_ranking_by_macro_f1"] == ["baseline"]
    assert "BASELINE DACL FRENTE A SUAVIZADO" not in resultados.render_text(report)


def test_invalid_baseline_cannot_anchor_comparisons(monkeypatch, tmp_path):
    baseline = run("invalid_baseline", "a" * 64, .95)
    baseline["saved_metrics_match_history_selection"] = False
    smooth = run("smoothing", "a" * 64, .79, smoothing=.05)
    report = report_for(monkeypatch, tmp_path, [baseline, smooth])
    assert report["classification_reference"] is None
    assert report["comparable_cnn_ranking_by_macro_f1"] == []
    assert "BASELINE DACL FRENTE A SUAVIZADO" not in resultados.render_text(report)


def test_current_loss_comparison_excludes_older_f1_selected_baseline(monkeypatch, tmp_path):
    old = run("f1_baseline", "a" * 64, .99)
    current = run("loss_baseline", "a" * 64, .75)
    smooth = run("loss_smoothing", "a" * 64, .78, smoothing=.05)
    for item in (current, smooth):
        item["configuration"]["selection_metric"] = "val_loss"
    report = report_for(monkeypatch, tmp_path, [old, current, smooth])
    assert report["classification_reference"] == "loss_baseline"
    assert report["comparable_cnn_ranking_by_macro_f1"] == ["loss_smoothing", "loss_baseline"]
    assert len(report["runs"]) == 3
