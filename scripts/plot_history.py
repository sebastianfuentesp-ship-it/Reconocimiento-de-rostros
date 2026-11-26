"""Generar curvas reproducibles a partir del historial CSV del entrenamiento."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.history.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("El historial no contiene épocas terminadas.")
    epochs = [int(row["epoch"]) for row in rows]
    first_fine = next((int(row["epoch"]) for row in rows
                       if row["phase"].startswith("fine_tune_")), None)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, metric, title in zip(axes, ("macro_f1", "accuracy"), ("F1 macro", "Exactitud")):
        for split, label in (("train", "Entrenamiento con aumentos"),
                             ("val", "Validación")):
            axis.plot(epochs, [float(row[f"{split}_{metric}"]) for row in rows],
                      marker="o", label=label)
        if first_fine is not None:
            axis.axvline(first_fine - 0.5, color="#777777", linestyle=":",
                         label="Inicio del ajuste fino")
        axis.set(xlabel="Época", ylabel=title, title=title, ylim=(0, 1), xticks=epochs)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("ResNet18: evolución sobre siete expresiones faciales")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"Curvas guardadas en {args.output}")


if __name__ == "__main__":
    main()
