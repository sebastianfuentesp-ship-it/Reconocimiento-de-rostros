"""Descarga y procedencia de los pesos base MSCeleb publicados por DAN."""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

MSCELEB_FILENAME = "resnet18_msceleb.pth"
MSCELEB_SIZE_BYTES = 181_511_212
MSCELEB_SHA256 = "734341508e3ddbcd181e40da0173cf2c77cd0de3bc2455715cc1b7f42a63a6f6"
MSCELEB_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1u2NtY-5DVlTunfN4yxfxys5n8uh7sc3n&export=download&confirm=t"
)
MSCELEB_REFERENCE_URL = (
    "https://github.com/yaoing/DAN/blob/"
    "3872eefef1441231342b1a798719f6e69cdbc1fd/README.md"
)


def verify_msceleb_weights(path: str | Path) -> dict:
    """Valida bytes y hash sin deserializar pesos ni importar PyTorch."""
    path = Path(path).resolve()
    size = path.stat().st_size
    if size != MSCELEB_SIZE_BYTES:
        raise ValueError(
            f"Tamaño de pesos MSCeleb incorrecto: {size} bytes; "
            f"se esperaban {MSCELEB_SIZE_BYTES}. Archivo: {path}"
        )
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != MSCELEB_SHA256:
        raise ValueError(f"SHA256 de pesos MSCeleb no coincide. Archivo: {path}")
    return {
        "source": "msceleb",
        "dataset": "MS-Celeb-1M (según autores de DAN)",
        "path": str(path),
        "sha256": digest,
        "size_bytes": size,
        "source_url": MSCELEB_URL,
        "reference_url": MSCELEB_REFERENCE_URL,
        "attribution_note": (
            "Los autores de DAN identifican este archivo como preentrenamiento MSCeleb, "
            "en un enlace distinto de sus modelos finales de emociones RAF-DB. "
            "No se dispone de una auditoría independiente de sus datos de preentrenamiento."
        ),
        "hash_note": (
            "SHA256 fijado a la descarga observada del enlace oficial; "
            "no es un hash publicado por los autores."
        ),
    }


def _write_receipt(path: Path, provenance: dict) -> None:
    receipt = path.with_suffix(".json")
    temporary = receipt.with_suffix(".json.part")
    try:
        temporary.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(receipt)
    finally:
        temporary.unlink(missing_ok=True)


def download_msceleb(output_dir: str | Path = "data/pretrained") -> Path:
    """Conserva pesos válidos y publica la descarga solo después de verificarla."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / MSCELEB_FILENAME
    if target.exists():
        # Un archivo existente inválido se informa; nunca se reemplaza silenciosamente.
        _write_receipt(target, verify_msceleb_weights(target))
        return target

    partial = target.with_suffix(target.suffix + ".part")
    request = Request(MSCELEB_URL, headers={"User-Agent": "RAFDB-course-project/0.1"})
    try:
        with urlopen(request, timeout=60) as response:
            if response.getcode() != 200:
                raise ValueError("La descarga de pesos no devolvió un archivo completo (HTTP 200).")
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                raise ValueError("Drive devolvió una página HTML en lugar de los pesos MSCeleb.")
            received = 0
            with partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    if received == 0 and chunk.lstrip().lower().startswith((b"<!doctype", b"<html")):
                        raise ValueError("La respuesta es HTML, no un archivo de pesos MSCeleb.")
                    received += len(chunk)
                    if received > MSCELEB_SIZE_BYTES:
                        raise ValueError("La descarga supera el tamaño esperado de los pesos MSCeleb.")
                    stream.write(chunk)
        verify_msceleb_weights(partial)
        if target.exists():
            # Otra descarga pudo terminar durante la petición; conserva ese archivo.
            verify_msceleb_weights(target)
        else:
            partial.replace(target)
        provenance = verify_msceleb_weights(target)
        _write_receipt(target, provenance)
        return target
    finally:
        partial.unlink(missing_ok=True)


def run(args) -> None:
    path = download_msceleb(args.output_dir)
    print(f"Pesos base MSCeleb verificados: {path}")
    print(f"Procedencia: {path.with_suffix('.json')}")
    print("Esta operación descarga pesos; no entrena ni evalúa el modelo.")


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "pretrained", help="Descargar y verificar los pesos base MSCeleb publicados por DAN"
    )
    parser.add_argument("--output-dir", default="data/pretrained")
    parser.set_defaults(func=run)
