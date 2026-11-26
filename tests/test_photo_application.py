"""Contrato de la aplicación y detección/recorte, sin entrenar ni llamar a la CNN."""

import io
from threading import Lock

import pytest
from PIL import Image

from app.web import create_app
from emotion_recognition import inference
from emotion_recognition.face_detection import FaceBox, FaceDetector, crop_face


def photo(mode="RGB", size=(80, 100)):
    buffer = io.BytesIO()
    Image.new(mode, size, color=0).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class FakeService:
    def __init__(self):
        self.calls = []

    def analyze(self, image, already_cropped=False):
        self.calls.append((image.size, image.mode, already_cropped))
        return {"faces": [{"id": 1, "label": "Neutral", "scores": []}],
                "width": image.width, "height": image.height, "mode": "automatic"}


def test_upload_preserves_explicit_crop_mode_and_reuses_service():
    service = FakeService()
    created = []

    def factory():
        created.append(True)
        return service

    app = create_app(factory)
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/health").json["loaded"] is False
    for _ in range(2):
        response = client.post("/api/predict", data={"photo": (photo("RGBA"), "../../photo.png"),
                                                     "cropped": "true"})
        assert response.status_code == 200
        assert response.json["preview"].startswith("data:image/jpeg;base64,")
        assert response.headers["Cache-Control"] == "no-store"
    assert service.calls == [((80, 100), "RGB", True)] * 2
    assert created == [True]


def test_missing_invalid_or_oversized_upload_does_not_load_model():
    def forbidden():
        pytest.fail("Un archivo inválido no debe cargar el modelo.")

    app = create_app(forbidden)
    client = app.test_client()
    assert client.post("/api/predict").status_code == 400
    assert client.post("/api/predict", data={"photo": (io.BytesIO(b"not an image"), "x.jpg")}).status_code == 400
    app.config["MAX_CONTENT_LENGTH"] = 1024
    response = client.post("/api/predict", data={"photo": (io.BytesIO(b"x" * 2048), "large.png")})
    assert response.status_code == 413 and "12 MB" in response.json["error"]


def test_no_face_returns_an_actionable_error():
    class NoFace:
        def analyze(self, *args, **kwargs):
            raise inference.NoFaceFound("No encontramos un rostro frontal.")

    response = create_app(NoFace).test_client().post("/api/predict", data={"photo": (photo(), "blank.png")})
    assert response.status_code == 422
    assert "rostro" in response.json["error"]


def test_blank_image_is_not_classified_as_a_face():
    assert FaceDetector().detect(Image.new("RGB", (200, 200), "white")) == []


def test_square_crop_stays_inside_image_at_boundary():
    image = Image.new("RGB", (100, 80))
    result = crop_face(image, FaceBox(0, 0, 40, 50))
    assert result.width == result.height == 57


def test_service_keeps_detection_separate_and_only_classifies_crops(monkeypatch):
    service = inference.EmotionRecognitionService.__new__(inference.EmotionRecognitionService)
    service._lock = Lock()
    service.model, service.metadata, service.device = object(), {}, "cpu"

    class Detector:
        def detect(self, image):
            return [FaceBox(10, 10, 40, 40), FaceBox(60, 20, 30, 30)]

    service.detector = Detector()
    crops = []

    def classify(image, *args):
        crops.append(image.size)
        return {"label": 6, "class_name": "neutral", "probabilities": {
            name: (1.0 if name == "neutral" else 0.0) for name in inference.CLASS_NAMES}}

    monkeypatch.setattr(inference, "predict_face", classify)
    result = service.analyze(Image.new("RGB", (100, 100)))
    assert len(result["faces"]) == 2
    assert crops == [(46, 46), (34, 34)]
    result = service.analyze(Image.new("RGB", (100, 100)), already_cropped=True)
    assert len(result["faces"]) == 1 and crops[-1] == (100, 100)
    assert result["faces"][0]["label"] == "Neutral"


def test_deployment_hash_is_verified_before_loading_network(tmp_path, monkeypatch):
    model = tmp_path / "wrong.pt"
    model.write_bytes(b"different checkpoint")
    monkeypatch.setattr(inference, "load_deployment", lambda: {"checkpoint": str(model), "sha256": "0" * 64})
    monkeypatch.setattr(inference, "load_checkpoint", lambda *a: pytest.fail("No cargar un checkpoint distinto."))
    with pytest.raises(ValueError, match="no coincide"):
        inference.EmotionRecognitionService()
