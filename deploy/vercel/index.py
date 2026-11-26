"""Entrada WSGI que Vercel detecta; no instala ni importa PyTorch."""
from app.web import create_app
from emotion_recognition.inference_onnx import OnnxEmotionService

app = create_app(OnnxEmotionService)
