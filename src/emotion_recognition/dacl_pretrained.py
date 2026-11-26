"""Traceable download of the base face weights linked by the DACL authors."""

from pathlib import Path
from urllib.request import Request, urlopen

from .pretrained import (
    MSCELEB_SHA256,
    MSCELEB_SIZE_BYTES,
    _write_receipt,
    verify_msceleb_weights,
)

DACL_FILENAME = "resnet18_msceleb_dacl.pth"
DACL_SHA256 = MSCELEB_SHA256
DACL_SIZE_BYTES = MSCELEB_SIZE_BYTES
DACL_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=16uPBWtSfVOJvlxMnb0xg6sOP3-zhHsx3&export=download&confirm=t"
)
DACL_REFERENCE_URL = (
    "https://github.com/amirhfarzaneh/dacl/blob/"
    "c41e9e81de9b70a84cae3d7c19102b5095964d5b/README.md"
)


def verify_dacl_weights(path: str | Path) -> dict:
    """DACL and DAN links were observed to distribute exactly the same bytes."""
    provenance = verify_msceleb_weights(path)
    provenance.update({
        "source": "msceleb_dacl",
        "dataset": "MS-Celeb-1M (según autores de DACL)",
        "source_url": DACL_URL, "reference_url": DACL_REFERENCE_URL,
        "attribution_note": (
            "Archivo base MSCeleb enlazado para inicialización por los autores de DACL. "
            "El SHA256 coincide con el archivo base distribuido por DAN. No se dispone "
            "de una auditoría independiente de los datos originales de preentrenamiento."
        ),
    })
    return provenance


def download_dacl(output_dir: str | Path = "data/pretrained") -> Path:
    """Download only initial face weights, never a model trained on RAF-DB."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / DACL_FILENAME
    if target.exists():
        _write_receipt(target, verify_dacl_weights(target))
        return target
    partial = target.with_suffix(".pth.part")
    request = Request(DACL_URL, headers={"User-Agent": "RAFDB-course-project/0.1"})
    print("Descargando pesos base DACL (182 MB aproximadamente)...", flush=True)
    try:
        with urlopen(request, timeout=60) as response:
            if response.getcode() != 200 or "html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("Drive no devolvió un archivo completo de pesos DACL.")
            received = 0
            with partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > DACL_SIZE_BYTES:
                        raise ValueError("La descarga supera el tamaño esperado.")
                    stream.write(chunk)
        verify_dacl_weights(partial)
        if target.exists():
            verify_dacl_weights(target)
        else:
            partial.replace(target)
        _write_receipt(target, verify_dacl_weights(target))
        return target
    finally:
        partial.unlink(missing_ok=True)


def run(args) -> int:
    path = download_dacl(args.output_dir)
    print(f"Pesos base DACL verificados: {path}")
    print(f"Procedencia: {path.with_suffix('.json')}")
    return 0


def add_parser(subparsers):
    parser = subparsers.add_parser("pretrained-dacl", help="Descargar y verificar pesos base de DACL")
    parser.add_argument("--output-dir", default="data/pretrained")
    parser.set_defaults(func=run)
