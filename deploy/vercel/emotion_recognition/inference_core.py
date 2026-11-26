"""Detección, recortes y presentación compartidos por PyTorch y ONNX."""
from PIL import ImageOps

from .constants import CLASS_NAMES, SPANISH_NAMES
from .face_detection import FaceBox, crop_face


class NoFaceFound(ValueError):
    """La foto es válida, pero no se encontró un rostro frontal."""

def analyze_faces(image, detector, predictor, lock, already_cropped=False):
    """Aplicar el mismo flujo de fotos independientemente del motor de la CNN."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < 32:
        raise ValueError("La imagen debe medir al menos 32 píxeles por lado.")

    # 3. Distinguir una fotografía completa de un rostro recortado explícitamente.
    with lock:
        boxes = ([FaceBox(0, 0, *image.size)] if already_cropped
                 else detector.detect(image))
        if not boxes:
            raise NoFaceFound(
                "No encontramos un rostro frontal. Prueba con una foto más cercana y bien "
                "iluminada, o marca «La foto ya es un rostro recortado» si corresponde."
            )

        # 4. Clasificar hasta cinco rostros. predict_face reutiliza el perfil guardado:
        # resize256, recorte central224, normalización DACL, eval e inference_mode.
        faces = []
        for index, box in enumerate(boxes[:5], start=1):
            face = image if already_cropped else crop_face(image, box)
            prediction = predictor(face)
            scores = [
                {"class_name": name, "label": SPANISH_NAMES[i].capitalize(),
                 "score": prediction["probabilities"][name]}
                for i, name in enumerate(CLASS_NAMES)
            ]
            faces.append({
                "id": index, "label": SPANISH_NAMES[prediction["label"]].capitalize(),
                "class_name": prediction["class_name"],
                "scores": sorted(scores, key=lambda row: row["score"], reverse=True),
                "box": {"x": box.x, "y": box.y, "width": box.width, "height": box.height},
            })
    return {"faces": faces, "detected_faces": len(boxes),
            "width": image.width, "height": image.height,
            "mode": "cropped" if already_cropped else "automatic"}
