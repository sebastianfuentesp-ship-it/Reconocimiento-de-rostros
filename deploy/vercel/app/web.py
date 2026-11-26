"""Capa HTTP: recibir una imagen, llamar al servicio y presentar el resultado."""

import base64
import io
from threading import Lock

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from emotion_recognition.inference_core import NoFaceFound


def default_service():
    # La versión ONNX no importa PyTorch: el motor local se carga solo si se usa.
    from emotion_recognition.inference import EmotionRecognitionService
    return EmotionRecognitionService()


def create_app(service_factory=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
    factory = service_factory or default_service
    init_lock = Lock()

    def get_service():
        # Carga diferida: abrir la página no necesita inicializar PyTorch/GPU.
        with init_lock:
            if "emotion_service" not in app.extensions:
                app.extensions["emotion_service"] = factory()
        return app.extensions["emotion_service"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", model="D", loaded="emotion_service" in app.extensions)

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("photo")
        if upload is None or not upload.filename:
            return jsonify(error="Selecciona una fotografía para continuar."), 400
        try:
            # 1. Validar el contenido real; el nombre del archivo no se usa como ruta.
            with Image.open(upload.stream) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Usa una imagen JPG, PNG o WEBP.")
                if source.width * source.height > 24_000_000:
                    raise ValueError("La imagen supera 24 megapíxeles. Reduce su tamaño.")
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")

            # 2. Delegar la detección y clasificación; esta capa no conoce la CNN.
            result = get_service().analyze(
                image, already_cropped=request.form.get("cropped") == "true",
            )

            # 3. Entregar una vista orientada igual que las coordenadas de los rostros.
            preview = image.copy()
            preview.thumbnail((1600, 1600))
            buffer = io.BytesIO()
            preview.save(buffer, format="JPEG", quality=88)
            result["preview"] = "data:image/jpeg;base64," + base64.b64encode(
                buffer.getvalue()).decode("ascii")
            return jsonify(result)
        except NoFaceFound as exc:
            return jsonify(error=str(exc)), 422
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify(error="No se pudo leer la imagen. Prueba con otro archivo JPG o PNG."), 400
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except (RuntimeError, FileNotFoundError):
            app.logger.exception("No se pudo cargar o ejecutar el modelo")
            return jsonify(error="No se pudo cargar el modelo. Revisa la terminal de la aplicación."), 503

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="El archivo es demasiado grande. El límite es 4 MB."), 413

    @app.after_request
    def no_cache(response):
        # Las fotografías se procesan en memoria; no se guardan en el proyecto.
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return app
