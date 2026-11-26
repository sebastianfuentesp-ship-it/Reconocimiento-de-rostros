"""Compare saved test predictions; never load models or run inference."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'B': 'resnet18_dacl_suavizado_20261125_213647_482',
    'D': 'resnet18_dacl_suavizado_pesos_20261125_223554_776',
}
records = {}
predictions = {}
for case, run in RUNS.items():
    directory = ROOT / 'artifacts' / run
    metrics = json.loads((directory / 'test/metrics.json').read_text(encoding='utf-8'))
    provenance = json.loads((directory / 'test/evaluation.json').read_text(encoding='utf-8'))
    validation = json.loads((directory / 'validation/metrics.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    with (directory / 'test/predictions.csv').open(encoding='utf-8', newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row['image_path'])
    assert metrics['samples'] == len(rows) == 3067
    assert provenance['split'] == 'test'
    assert provenance['manifest_sha256'] == config['manifest_sha256']
    assert config['selection_metric'] == 'val_loss'
    probabilities = np.array([
        [float(row['probability_' + name]) for name in metrics['class_names']] for row in rows
    ])
    truth = np.array([int(row['true_label']) for row in rows])
    prediction = np.array([int(row['predicted_label']) for row in rows])
    loss = float(-np.log(np.clip(probabilities[np.arange(len(rows)), truth], 1e-15, 1)).mean())
    predictions[case] = rows, truth, prediction
    records[case] = {
        'run': run, 'test': metrics, 'validation': validation, 'test_cross_entropy': loss,
        'provenance': provenance,
        'accuracy_val_to_test_change_pp': (metrics['accuracy'] - validation['accuracy']) * 100,
        'macro_f1_val_to_test_change': metrics['macro_f1'] - validation['macro_f1'],
    }
assert records['B']['provenance']['manifest_sha256'] == records['D']['provenance']['manifest_sha256']
b_rows, truth, b_pred = predictions['B']
d_rows, d_truth, d_pred = predictions['D']
assert [r['image_path'] for r in b_rows] == [r['image_path'] for r in d_rows]
assert np.array_equal(truth, d_truth)
b_correct, d_correct = b_pred == truth, d_pred == truth
paired = {'both_correct': int((b_correct & d_correct).sum()),
          'only_B_correct': int((b_correct & ~d_correct).sum()),
          'only_D_correct': int((~b_correct & d_correct).sum()),
          'both_wrong': int((~b_correct & ~d_correct).sum())}

def macro_f1(cm):
    denominator = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * np.diag(cm), denominator,
                           out=np.zeros(7), where=denominator != 0).mean())

# Paired bootstrap: both predictions are resampled together for each test image.
rng = np.random.default_rng(42)
differences = []
for _ in range(2000):
    indices = rng.integers(0, len(truth), size=len(truth))
    matrices = [np.bincount(truth[indices] * 7 + pred[indices], minlength=49).reshape(7, 7)
                for pred in (b_pred, d_pred)]
    differences.append(macro_f1(matrices[1]) - macro_f1(matrices[0]))
interval = np.quantile(differences, [.025, .975]).tolist()
winner = max(records, key=lambda key: records[key]['test']['macro_f1'])
report = {
    'scope': 'Comparison of two saved checkpoints on the cleaned official test, not training.',
    'test_samples': len(truth), 'models': records, 'paired_predictions': paired,
    'winner_by_test_macro_f1': winner,
    'D_minus_B_macro_f1': records['D']['test']['macro_f1'] - records['B']['test']['macro_f1'],
    'paired_bootstrap_f1_difference_95pct': interval,
    'bootstrap': {'replicates': 2000, 'seed': 42, 'method': 'paired image resampling'},
    'note': 'Test is now consulted. Do not tune further hyperparameters or checkpoints using these scores.',
}
out = ROOT / 'documentos/Comparacion_final_test_B_D'
out.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['COMPARACION FINAL TEST B Y D', '', 'Mismas3067 imagenes del test oficial depurado.',
         'Checkpoints seleccionados previamente por menor val_loss. Sin entrenamiento nuevo.', '']
for case, record in records.items():
    m = record['test']
    lines += [f"{case}: accuracy={m['accuracy']*100:.4f}%; F1macro={m['macro_f1']:.6f}; "
              f"recallmacro={m['balanced_accuracy']*100:.4f}%; CEtest={record['test_cross_entropy']:.6f}.",
              record['run']]
lines += ['', 'PRECISION / RECALL / F1 POR CLASE EN TEST']
for name in records['B']['test']['class_names']:
    lines.append(name)
    for case in RUNS:
        m = records[case]['test']['per_class'][name]
        lines.append(f"  {case}: precision={m['precision']:.6f}, recall={m['recall']:.6f}, F1={m['f1-score']:.6f}, n={m['support']:.0f}")
lines += ['', f'Ganador descriptivo por F1macro:test {winner}.',
          'Comparacion pareada: ' + str(paired),
          'ICbootstrap95% diferencia F1macro D-B: ' + str(interval),
          'Intervalo orientativo de remuestreo de imagenes; no prueba superioridad universal.',
          report['note']]
out.with_suffix('.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({key: report[key] for key in ['winner_by_test_macro_f1', 'paired_predictions',
                 'D_minus_B_macro_f1', 'paired_bootstrap_f1_difference_95pct']}, ensure_ascii=False))
for case, r in records.items():
    print(case, json.dumps({k: r['test'][k] for k in ['accuracy', 'macro_f1', 'balanced_accuracy']}),
          'CE', r['test_cross_entropy'])
