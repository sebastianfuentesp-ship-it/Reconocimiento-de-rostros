"""Exportar D a ONNX, comprobar equivalencia y generar el paquete de Vercel.

No entrena ni vuelve a evaluar el dataset. Usa fotos sintéticas para comprobar
que el formato de inferencia mantiene las salidas numéricas del checkpoint.
Requiere: pip install onnx onnxruntime. Ejecutar desde la raíz del proyecto.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from emotion_recognition.inference import load_deployment
from emotion_recognition.inference_onnx import OnnxEmotionService, preprocess
from emotion_recognition.model import build_transforms, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "deploy/vercel"
    models = output / "models"
    models.mkdir(parents=True, exist_ok=True)
    deployment = load_deployment()
    checkpoint = ROOT / deployment["checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == deployment["sha256"]
    model, metadata = load_checkpoint(checkpoint, "cpu")

    # 1. Exportar los mismos pesos en float32, sin cuantizar ni entrenar.
    target = models / "modelo_d.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(target),
                      input_names=["images"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    manifest = {"case": "D", "source_checkpoint_sha256": deployment["sha256"],
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "class_names": metadata["class_names"], "format": "ONNX float32"}
    (models / "modelo_onnx.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 2. Comparar preprocesamiento y puntuaciones sobre tamaños/aspectos distintos.
    service = OnnxEmotionService(models)
    transform = build_transforms(False, 224, "dacl_baseline")
    rng = np.random.default_rng(42)
    errors = []
    for width, height in [(256, 256), (512, 400), (321, 623), (81, 80), (224, 224), (48, 48)]:
        image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        tensor = transform(image).unsqueeze(0)
        np.testing.assert_allclose(tensor.numpy(), preprocess(image), atol=1e-6, rtol=0)
        with torch.inference_mode():
            reference = torch.softmax(model(tensor), 1)[0].numpy()
        prediction = service.predict(image)
        actual = np.array(list(prediction["probabilities"].values()))
        np.testing.assert_allclose(reference, actual, atol=5e-5, rtol=1e-4)
        assert int(reference.argmax()) == prediction["label"]
        errors.append(float(np.max(np.abs(reference - actual))))
    report = {"synthetic_images": len(errors), "max_probability_error": max(errors),
              "same_predictions": True, "source_checkpoint_sha256": deployment["sha256"],
              "onnx_sha256": manifest["sha256"], "training_performed": False}
    (models / "verificacion_exportacion.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    # 3. Empaquetar solo los módulos de inferencia y la interfaz compartida.
    package = output / "emotion_recognition"
    package.mkdir(exist_ok=True)
    for name in ["__init__.py", "constants.py", "face_detection.py", "inference_core.py", "inference_onnx.py"]:
        shutil.copy2(ROOT / "src/emotion_recognition" / name, package / name)
    shutil.copytree(ROOT / "app", output / "app", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    # El servidor público usa un límite inferior al máximo de Vercel (4.5 MB).
    template = output / "app/templates/index.html"
    template.write_text(template.read_text(encoding="utf-8").replace("12 MB", "4 MB")
                        .replace("Procesamiento local", "Procesamiento en servidor"), encoding="utf-8")
    script = output / "app/static/app.js"
    script.write_text(script.read_text(encoding="utf-8-sig").replace("12 * 1024", "4 * 1024")
                      .replace("12 MB", "4 MB"), encoding="utf-8")
    web = output / "app/web.py"
    web.write_text(web.read_text(encoding="utf-8").replace("12 * 1024", "4 * 1024")
                   .replace("12 MB", "4 MB")
                   .replace("from emotion_recognition.inference import EmotionRecognitionService",
                            "from emotion_recognition.inference_onnx import OnnxEmotionService as EmotionRecognitionService"),
                   encoding="utf-8")
    # Vercel sirve los archivos estáticos desde public/; mantener sus URL /static/.
    shutil.copytree(output / "app/static", output / "public/static", dirs_exist_ok=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
