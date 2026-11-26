"""Punto de entrada; importa únicamente el módulo del comando solicitado."""

import argparse
import importlib
import sys

COMMANDS = {
    "download": ("download", "Descargar y extraer el ZIP del profesor sin modificarlo"),
    "pretrained": ("pretrained", "Descargar y verificar los pesos base de reconocimiento facial"),
    "pretrained-dacl": ("dacl_pretrained", "Descargar los pesos base enlazados por DACL"),
    "prepare": ("prepare", "Auditar, limpiar y dividir RAF-DB"),
    "explore": ("explore", "Visualizar distribución de clases y ejemplos de entrenamiento"),
    "train": ("train", "Entrenar ResNet18 con validación independiente"),
    "train-dacl-baseline": ("train_dacl_baseline", "Ajustar ResNet18 completa con la receta baseline de DACL"),
    "train-dacl-full": ("train_dacl_full", "Entrenar DACL: ResNet18, atención y sparse center loss"),
    "baseline": ("baseline", "Entrenar la comparación HOG + regresión logística"),
    "evaluate": ("evaluate", "Evaluar un modelo guardado en una partición"),
    "predict": ("predict", "Clasificar una imagen de rostro ya recortada"),
    "doctor": ("doctor", "Comprobar entorno, GPU y dependencias"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="emociones", description="Reconocimiento de expresiones faciales con RAF-DB"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = argv[0] if argv else None
    for name, (module_name, help_text) in COMMANDS.items():
        if name == command:
            try:
                module = importlib.import_module(f"emotion_recognition.{module_name}")
            except KeyboardInterrupt:
                parser.exit(130, "\nCarga interrumpida por el usuario.\n")
            module.add_parser(subparsers)
        else:
            subparsers.add_parser(name, help=help_text)
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except KeyboardInterrupt:
        parser.exit(130, "\nEjecución interrumpida. Se conservan los checkpoints ya guardados.\n")
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
