"""Resume métricas ya guardadas; no importa modelos ni ejecuta inferencia."""

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_STEM = "Resultados_experimentos_RAFDB"
CONFIG_KEYS = (
    "command", "architecture", "format", "epochs", "warmup_epochs", "batch_size", "seed",
    "image_size", "lr", "backbone_lr", "layer3_lr", "momentum", "weight_decay", "optimizer",
    "scheduler", "step_size", "gamma", "scheduler_factor", "scheduler_patience", "patience",
    "selection_metric", "fine_tune_from", "fine_tune_stages", "freeze_backbone",
    "pretrained_source", "initialization", "no_pretrained", "preprocessing_profile",
    "normalization", "class_counts_train", "class_weights", "class_weight", "no_class_weights",
    "class_weighting", "label_smoothing", "train_label_smoothing", "validation_label_smoothing",
    "train_loss_definition", "val_loss_definition", "precision", "amp", "amp_enabled",
    "gradient_clip", "early_stopping", "min_epochs", "early_stopping_metric", "early_stopping_mode", "max_train_samples", "train_samples", "val_samples",
    "training_samples", "validation_samples", "selected_C", "search", "pca_components",
    "hog_config", "test_used", "test_evaluated", "warning", "paper_url", "source_url",
    "source_commit", "deviations",
    "training_method", "alpha", "center_lambda", "center_update_rule", "center_update",
    "parameter_count", "attention_parameter_count", "train_ce_loss_definition",
)
CALIBRATION_KEYS = (
    "validation_nll", "correct", "errors", "brier_sum", "ece_15_equal_width_bins",
    "mean_confidence", "incorrect_mean_confidence", "incorrect_confidence_at_least_90pct",
)


def read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_row(row):
    result = {}
    for key, value in row.items():
        if value == "":
            result[key] = None
        elif key == "epoch":
            result[key] = int(value)
        elif key == "phase":
            result[key] = value
        elif value in {"True", "False"}:
            result[key] = value == "True"
        else:
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                result[key] = value
    return result


def read_history(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [numeric_row(row) for row in csv.DictReader(stream)]


def pick_epoch(history, key):
    candidates = [row for row in history if isinstance(row.get(key), (int, float))]
    select = min if key == "val_loss" else max
    return select(candidates, key=lambda row: row[key]) if candidates else None


def identify_checkpoint(run, evaluation, hog):
    recorded = evaluation.get("checkpoint")
    if recorded:
        candidate = Path(recorded)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_file():
            return candidate.resolve(), "evaluation.json"
    fallback = run / ("baseline.joblib" if hog else "best.pt")
    if fallback.is_file():
        return fallback.resolve(), "archivo del directorio de ejecución"
    if hog:
        candidates = sorted([*run.glob("*.joblib"), *run.glob("*.pkl")])
        if len(candidates) == 1:
            return candidates[0].resolve(), "único archivo joblib/pkl del directorio"
    return None, None


def collect_runs(artifacts):
    runs, excluded, raw_records = [], [], {}
    for directory in sorted(artifacts.iterdir()):
        if not directory.is_dir() or not (directory / "config.json").is_file():
            continue
        config_path = directory / "config.json"
        config = read_json(config_path)
        hog = config.get("format") == "hog_logistic_v1"
        metrics_path = directory / "validation/metrics.json"
        if not metrics_path.is_file():
            metrics_path = directory / "metrics.json"
        history_path = directory / "history.csv"
        if not metrics_path.is_file() or (not hog and not history_path.is_file()):
            excluded.append({
                "name": directory.name, "run_path": str(directory.resolve()),
                "reason": "Faltan métricas guardadas o historial CNN; no se inventan resultados.",
                "config_present": True, "metrics_present": metrics_path.is_file(),
                "history_present": history_path.is_file(),
            })
            continue
        metrics = read_json(metrics_path)
        history = read_history(history_path) if history_path.is_file() else []
        evaluation_path = metrics_path.parent / "evaluation.json"
        evaluation = read_json(evaluation_path) if evaluation_path.is_file() else {}
        best_f1 = pick_epoch(history, "val_macro_f1")
        selection = config.get("selection_metric", "val_macro_f1")
        selection_key = ("val_loss" if selection in {"loss", "val_loss"} else
                         "val_accuracy" if selection in {"accuracy", "val_accuracy"} else "val_macro_f1")
        selected = pick_epoch(history, selection_key)
        checkpoint, checkpoint_source = identify_checkpoint(directory, evaluation, hog)
        checkpoint_hash = file_sha256(checkpoint) if checkpoint else None
        evaluation_hash = evaluation.get("checkpoint_sha256")
        split = evaluation.get("split", metrics.get("split", config.get("split")))
        if split is None and metrics_path.parent.name == "validation":
            split = "val"
        smoothing = config.get("train_label_smoothing", config.get("label_smoothing"))
        is_dacl = config.get("preprocessing_profile") == "dacl_baseline"
        is_full = (config.get("training_method") == "dacl_full"
                   or config.get("architecture") == "dacl_resnet18")
        technical = directory.name.startswith("smoke") or config.get("max_train_samples") is not None
        variant = "HOG + regresión logística" if hog else "ResNet18"
        if technical:
            variant = "Prueba técnica/reducida; no comparable con resultados completos"
        elif is_full:
            variant = "DACL completo: ResNet18 + atención + sparse center loss"
        elif is_dacl and smoothing not in {None, 0} and config.get("class_weighting") == "balanced":
            variant = f"Baseline DACL + suavizado {smoothing} + pesos de clase"
        elif is_dacl and smoothing not in {None, 0}:
            variant = f"Baseline DACL + suavizado {smoothing}"
        elif is_dacl and config.get("class_weighting") == "balanced":
            variant = "Baseline DACL + pesos de clase; sin suavizado"
        elif is_dacl:
            variant = "Baseline DACL; sin suavizado ni pesos de clase"
        checkpoint_metrics_match = None
        if selected:
            checkpoint_metrics_match = all(
                abs(selected[f"val_{key}"] - metrics[key]) < 1e-12
                for key in ("accuracy", "macro_f1")
                if key in metrics and f"val_{key}" in selected
            )
        record = {
            "name": directory.name, "variant": variant,
            "run_path": str(directory.resolve()),
            "model_kind": "hog_logistic_v1" if hog else "dacl_resnet18" if is_full else "resnet18",
            "technical_run": technical, "duplicate_saved_results_of": None,
            "accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("macro_f1"),
            "balanced_accuracy": metrics.get("balanced_accuracy"), "samples": metrics.get("samples"),
            "split": split, "manifest_sha256": evaluation.get("manifest_sha256", config.get("manifest_sha256")),
            "config_manifest_sha256": config.get("manifest_sha256"),
            "selected_C": config.get("selected_C") if hog else None,
            "epochs_recorded": len(history),
            "best_macro_f1_epoch": best_f1.get("epoch") if best_f1 else None,
            "best_macro_f1_history": best_f1,
            "selected_checkpoint_epoch_from_history": selected.get("epoch") if selected else None,
            "selected_checkpoint_history": selected,
            "maximum_accuracy_history": pick_epoch(history, "val_accuracy"),
            "saved_metrics_match_history_selection": checkpoint_metrics_match,
            "configuration": {key: config[key] for key in CONFIG_KEYS if key in config},
            "per_class": metrics.get("per_class"),
            "checkpoint": {
                "path": str(checkpoint) if checkpoint else None,
                "identification_source": checkpoint_source,
                "sha256_current_file": checkpoint_hash,
                "sha256_recorded_at_evaluation": evaluation_hash,
                "matches_evaluation_hash": checkpoint_hash == evaluation_hash
                if checkpoint_hash and evaluation_hash else None,
            },
            "sources": {
                "config": str(config_path.resolve()), "metrics": str(metrics_path.resolve()),
                "history": str(history_path.resolve()) if history_path.is_file() else None,
                "evaluation": str(evaluation_path.resolve()) if evaluation_path.is_file() else None,
            },
            "calibration_saved_analysis": None,
        }
        raw_records[directory.name] = {"history": history, "metrics": metrics}
        runs.append(record)
    # Match numerical results and the complete stored history, not checkpoint bytes.
    seen = {}
    for record in runs:
        raw = raw_records[record["name"]]
        if record["model_kind"] not in {"resnet18", "dacl_resnet18"} or not raw["history"]:
            continue
        signature = json.dumps(raw, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            record["duplicate_saved_results_of"] = seen[signature]
            record["variant"] += "; réplica exacta de métricas e historial, no independiente"
        else:
            seen[signature] = record["name"]
    return runs, excluded


def collect_saved_analyses(runs):
    analyses = []
    by_name = {run["name"]: run for run in runs}
    for run in runs:
        analysis_dir = Path(run["run_path"]) / "analysis"
        for path in sorted(analysis_dir.glob("*.json")):
            if not any(part in path.name.lower() for part in ("comparison", "calibration", "calibracion")):
                continue
            payload = read_json(path)
            analyses.append({"path": str(path.resolve()), "sha256": file_sha256(path), "data": payload})
            for result in payload.get("results", []):
                target = by_name.get(Path(result.get("run", "")).name)
                if target is None:
                    continue
                if any(abs(target[key] - result[key]) > 1e-12 for key in ("accuracy", "macro_f1")
                       if target.get(key) is not None and result.get(key) is not None):
                    continue
                target["calibration_saved_analysis"] = {
                    "source_path": str(path.resolve()),
                    **{key: result[key] for key in CALIBRATION_KEYS if key in result},
                }
    return analyses


def comparison_group(run):
    """Identify an evaluation protocol only when its identity is recorded."""
    if not run:
        return None
    manifest = run.get("manifest_sha256")
    samples = run.get("samples")
    split = run.get("split")
    if (not isinstance(manifest, str) or not manifest.strip()
            or not isinstance(samples, int) or isinstance(samples, bool) or samples < 1
            or not isinstance(split, str) or not split.strip()):
        return None
    selection = run.get("configuration", {}).get("selection_metric", "val_macro_f1")
    if not selection.startswith("val_"):
        selection = "val_" + selection
    return manifest, samples, split, selection


def valid_comparison_run(run):
    """Exclude incomplete identities, technical replicas and known provenance failures."""
    if not comparison_group(run):
        return False
    scores = [run.get(key) for key in ("accuracy", "macro_f1")]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) and 0 <= value <= 1 for value in scores):
        return False
    return bool(
        run.get("model_kind") in {"resnet18", "dacl_resnet18"}
        and not run.get("technical_run")
        and not run.get("duplicate_saved_results_of")
        and run.get("saved_metrics_match_history_selection") is not False
        and run.get("checkpoint", {}).get("matches_evaluation_hash") is not False
        and run.get("config_manifest_sha256") in {None, run["manifest_sha256"]}
    )


def build_report(artifacts):
    runs, excluded = collect_runs(artifacts)
    analyses = collect_saved_analyses(runs)
    baseline = None
    # Prefer the original unweighted DACL baseline, identified by its recorded settings.
    candidates = [run for run in runs
                  if valid_comparison_run(run)
                  and run["configuration"].get("preprocessing_profile") == "dacl_baseline"
                  and run["model_kind"] == "resnet18"
                  and run["configuration"].get("train_label_smoothing",
                                             run["configuration"].get("label_smoothing", 0)) == 0
                  and run["configuration"].get("class_weighting", "none") == "none"
                  and run["configuration"].get("class_weights") is None]
    if candidates:
        loss_candidates = [run for run in candidates if comparison_group(run)[3] == "val_loss"]
        baseline = max(loss_candidates or candidates, key=lambda run: run["macro_f1"])
    reference_group = comparison_group(baseline)
    for run in runs:
        run["cnn_ranking_eligible"] = bool(
            reference_group and valid_comparison_run(run)
            and comparison_group(run) == reference_group
        )
    ranking = sorted([run for run in runs if run["cnn_ranking_eligible"]],
                     key=lambda run: run["macro_f1"], reverse=True)
    class_weight_runs = [run["name"] for run in runs
                        if run["configuration"].get("preprocessing_profile") == "dacl_baseline"
                        and run["configuration"].get("class_weighting") == "balanced"
                        and run["configuration"].get("train_label_smoothing",
                                                   run["configuration"].get("label_smoothing", 0)) == 0]
    return {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "artifact_root": str(artifacts.resolve()), "completed_runs_count": len(runs),
        "scope": "Lectura de resultados existentes y hash de archivos; sin entrenamiento, "
                 "inferencia ni evaluación nueva. No se mueve ni borra ningún artefacto.",
        "notes": [
            ("Los resultados son los ya guardados; una carpeta con métricas no acredita por sí sola "
             "que alcanzara todas las épocas solicitadas."),
            "Una réplica exacta de métricas e historial con la misma semilla no es evidencia independiente.",
            "Las pruebas técnicas/reducidas y HOG se conservan fuera del ranking CNN comparable.",
            ("El ranking y las comparaciones usan solo el manifiesto, partición y número de muestras "
             "del grupo de referencia; otros protocolos se conservan sin compararlos entre sí."),
            "La época del checkpoint se obtiene del historial y criterio registrado; no se deserializan modelos.",
            "El test no se consulta; los JSON de análisis guardados se transcriben con su procedencia.",
            "La evaluación final B/D ya realizada se conserva por separado en Comparacion_final_test_B_D.json y .txt.",
        ],
        "classification_reference": baseline["name"] if baseline else None,
        "comparable_cnn_ranking_by_macro_f1": [run["name"] for run in ranking],
        "runs": runs, "excluded_incomplete_runs": excluded, "saved_analyses": analyses,
        "class_weight_experiment": {
            "status": "results_available" if class_weight_runs else "not_executed",
            "results": class_weight_runs,
        },
    }


def number(value, digits=6):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "no registrado"


def render_text(report):
    lines = [
        "RESULTADOS GUARDADOS DE EXPERIMENTOS RAF-DB", f"Registro UTC: {report['created_at_utc']}",
        f"Ejecuciones con resultados guardados: {report['completed_runs_count']}", report["scope"], "",
        "RESUMEN DE TODAS LAS EJECUCIONES",
        "Accuracy y balanced accuracy se expresan en porcentaje; F1 macro en escala 0-1.",
    ]
    for run in report["runs"]:
        epoch = run["selected_checkpoint_epoch_from_history"]
        lines.extend([
            "", run["name"], f"Tipo: {run['variant']}",
            (f"Accuracy={number(run['accuracy'] * 100 if run['accuracy'] is not None else None, 4)}%; "
            f"F1 macro={number(run['macro_f1'])}; "
             f"balanced accuracy={number(run['balanced_accuracy'] * 100 if run['balanced_accuracy'] is not None else None, 4)}%."),
            (f"Partición={run['split'] or 'no registrada'}; muestras={run['samples']}; "
            f"época del checkpoint según historial={epoch if epoch is not None else 'no aplica/no registrada'}; "
             f"mejor época F1={run['best_macro_f1_epoch'] if run['best_macro_f1_epoch'] is not None else 'no aplica/no registrada'}."),
            f"Directorio: {run['run_path']}", f"Manifest SHA256: {run['manifest_sha256']}",
            f"Modelo: {run['checkpoint']['path'] or 'no identificado'}",
            f"SHA256 actual del modelo: {run['checkpoint']['sha256_current_file'] or 'no disponible'}",
        ])
        if run["model_kind"] == "hog_logistic_v1":
            lines.append(f"Selección HOG: C={run['selected_C']}; búsqueda y configuración conservadas en JSON.")
        if run["duplicate_saved_results_of"]:
            lines.append(f"Réplica exacta de historial y métricas: {run['duplicate_saved_results_of']}; no independiente.")
        if run["checkpoint"]["matches_evaluation_hash"] is False:
            lines.append("ATENCIÓN: el hash actual del modelo difiere del registrado al evaluar.")
        if run["saved_metrics_match_history_selection"] is False:
            lines.append("ATENCIÓN: métricas guardadas no coinciden con la selección indicada por el historial.")
    by_name = {run["name"]: run for run in report["runs"]}
    baseline = by_name.get(report["classification_reference"])
    reference_group = comparison_group(baseline) if valid_comparison_run(baseline) else None
    lines.extend(["", "RANKING CNN COMPARABLE POR F1 MACRO DEL GRUPO DE REFERENCIA"])
    if reference_group:
        manifest, samples, split, selection = reference_group
        lines.append(f"Grupo: manifest SHA256={manifest}; partición={split}; muestras={samples}.")
        lines.append(f"Selección del checkpoint: {selection}; otros criterios quedan fuera del grupo.")
    else:
        lines.append("No hay una referencia válida con manifiesto, partición y muestras registrados.")
    lines.extend(["Otros manifiestos, particiones o tamaños de evaluación no se comparan con este grupo.",
                  "Pruebas técnicas, HOG, réplicas exactas y registros con fallos conocidos quedan fuera."])
    for position, name in enumerate(report["comparable_cnn_ranking_by_macro_f1"], 1):
        run = by_name[name]
        lines.append(f"{position}. {name}: accuracy={run['accuracy'] * 100:.4f}%; F1={run['macro_f1']:.6f}.")
    smooth = next((run for run in report["runs"]
                   if reference_group and valid_comparison_run(run)
                   and comparison_group(run) == reference_group
                   and run["model_kind"] == "resnet18"
                   and run["configuration"].get("class_weighting", "none") == "none"
                   and run["configuration"].get("class_weights") is None
                   and run["configuration"].get("train_label_smoothing",
                                              run["configuration"].get("label_smoothing")) == 0.05
                   and run["configuration"].get("preprocessing_profile") == "dacl_baseline"), None)
    if baseline and smooth:
        lines.extend([
            "", "BASELINE DACL FRENTE A SUAVIZADO 0.05",
            (f"Mismo grupo: manifest SHA256={baseline['manifest_sha256']}; "
             f"partición={baseline['split']}; muestras={baseline['samples']}."),
            f"Baseline: accuracy={baseline['accuracy'] * 100:.4f}%; F1={baseline['macro_f1']:.6f}.",
            f"Suavizado: accuracy={smooth['accuracy'] * 100:.4f}%; F1={smooth['macro_f1']:.6f}.",
            (f"Cambio suavizado - baseline: {(smooth['accuracy'] - baseline['accuracy']) * 100:+.4f} "
             f"puntos de accuracy y {smooth['macro_f1'] - baseline['macro_f1']:+.6f} de F1 macro."),
            "Se mantiene el baseline sin suavizado como referencia de clasificación.",
            "La CE/NLL de validación es comparable: se guardó sin suavizado ni pesos en ambos ensayos.",
            "La pérdida de train no se compara directamente: el suavizado cambia el objetivo.",
        ])
        base_cal = baseline["calibration_saved_analysis"]
        smooth_cal = smooth["calibration_saved_analysis"]
        if base_cal and smooth_cal:
            for key, label in (("validation_nll", "CE/NLL val"), ("brier_sum", "Brier multiclase (suma)"),
                               ("ece_15_equal_width_bins", "ECE, 15 intervalos"),
                               ("incorrect_mean_confidence", "Probabilidad máxima media en errores"),
                               ("incorrect_confidence_at_least_90pct", "Errores con probabilidad máxima >=0.9")):
                lines.append(f"{label}: baseline={number(base_cal.get(key))}; "
                             f"suavizado={number(smooth_cal.get(key))}.")
            lines.append("Interpretar calibración y clasificación por separado; ECE depende del binning. "
                         "Una semilla y validación reutilizada no prueban mejora universal.")
        pair_names = {baseline["name"], smooth["name"]}
        paired = next((analysis["data"].get("paired_predictions") for analysis in report["saved_analyses"]
                       if {Path(result.get("run", "")).name for result in analysis["data"].get("results", [])}
                       == pair_names and analysis["data"].get("paired_predictions")), None)
        if paired:
            lines.append(f"Comparación pareada guardada: corrige {paired.get('fixed')} errores y pierde "
                         f"{paired.get('lost')} aciertos del baseline.")
    if report["excluded_incomplete_runs"]:
        lines.extend(["", "EJECUCIONES SIN RESULTADOS SUFICIENTES"])
        for run in report["excluded_incomplete_runs"]:
            lines.append(f"{run['name']}: {run['reason']}")
    lines.extend(["", "NOTAS", *report["notes"], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "documentos")
    args = parser.parse_args()
    artifacts = args.artifacts_dir.resolve()
    if not artifacts.is_dir():
        parser.error(f"No existe artifacts-dir: {artifacts}")
    report = build_report(artifacts)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path, text_path = output / f"{OUTPUT_STEM}.json", output / f"{OUTPUT_STEM}.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(render_text(report), encoding="utf-8")
    print(f"Ejecuciones guardadas: {report['completed_runs_count']}")
    print(json_path)
    print(text_path)


if __name__ == "__main__":
    main()
