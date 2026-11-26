"""Abrir la aplicación local de fotografías desde VS Code o PowerShell.

Este archivo solo inicia la interfaz. No prepara datos, entrena ni evalúa test.
La selección del checkpoint final está en config/modelo_aplicacion.json.
"""
import argparse
import threading
import webbrowser

from werkzeug.serving import make_server

from app.web import create_app
from emotion_recognition.inference import EmotionRecognitionService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 1 y 65535.")

    # 1. La interfaz recibe un servicio independiente; el modelo se carga una sola vez.
    application = create_app(lambda: EmotionRecognitionService(device=args.device))
    url = f"http://127.0.0.1:{args.port}"

    # 2. Escuchar solo en este equipo, sin modo de depuración ni recarga automática.
    with make_server("127.0.0.1", args.port, application, threaded=True) as server:
        print(f"Expresiones está lista: {url} — Ctrl+C para cerrar.", flush=True)
        if not args.no_browser:
            timer = threading.Timer(1.0, lambda: webbrowser.open(url))
            timer.daemon = True
            timer.start()
        try:
            # 3. Atender fotografías hasta que el usuario cierre la aplicación.
            server.serve_forever()
        except KeyboardInterrupt:
            print("Aplicación cerrada.")


if __name__ == "__main__":
    main()
