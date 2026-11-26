"""Adquisición trazable del archivo proporcionado para la materia."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

DEFAULT_FILE_ID = "1zXTRGRaLIXiF6uSYts0gsW2ctoqqr2zH"


def safe_extract(archive: Path, destination: Path):
    """Valida todas las entradas antes de extraer (sin rutas externas ni enlaces)."""
    destination = destination.resolve()
    with ZipFile(archive) as zipped:
        for item in zipped.infolist():
            name = PurePosixPath(item.filename.replace("\\", "/"))
            target = destination.joinpath(*name.parts).resolve()
            mode = item.external_attr >> 16
            if (
                name.is_absolute() or ".." in name.parts
                or any(":" in part for part in name.parts)
                or not target.is_relative_to(destination)
                or (mode & 0o170000) == 0o120000
            ):
                raise ValueError(f"Entrada insegura en ZIP: {item.filename}")
        zipped.extractall(destination)


def run(args):
    import gdown

    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "RAF-DB_basic_emotions.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        result = gdown.download(id=args.file_id, output=str(partial), quiet=False, resume=True)
        if result is None:
            raise RuntimeError("Drive no permitió descargar. Descarga manualmente el ZIP del profesor.")
        try:
            with ZipFile(partial) as zipped:
                bad = zipped.testzip()
                if bad:
                    raise ValueError(f"Archivo dañado dentro del ZIP: {bad}")
        except BadZipFile as exc:
            raise ValueError("La descarga no es un ZIP válido; revisa acceso o cuota de Drive.") from exc
        partial.replace(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if args.sha256 and digest.lower() != args.sha256.lower():
        raise ValueError("SHA256 del ZIP no coincide con el valor solicitado.")
    # Evita sobreescribir una extracción ya existente o cambios locales.
    extracted = destination / "rafdb"
    receipt = destination / "download.json"
    if extracted.exists() and any(extracted.iterdir()):
        raise ValueError(f"Ya existe contenido en {extracted}; usa prepare sobre esa carpeta.")
    safe_extract(archive, extracted)
    nested_receipts = []
    # El ZIP de la materia contiene otro ZIP con los rostros alineados.
    for nested in sorted(extracted.rglob("aligned.zip")):
        with nested.open("rb") as stream:
            nested_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        safe_extract(nested, nested.parent)
        nested_receipts.append({"path": nested.relative_to(extracted).as_posix(),
                                "sha256": nested_digest})
    receipt.write_text(json.dumps({
        "source_url": f"https://drive.google.com/file/d/{args.file_id}/view",
        "archive": archive.name, "sha256": digest, "size_bytes": archive.stat().st_size,
        "extracted_root": str(extracted),
        "nested_archives": nested_receipts,
    }, indent=2), encoding="utf-8")
    print(f"Dataset extraído en: {extracted}\nSHA256: {digest}")


def add_parser(subparsers):
    parser = subparsers.add_parser("download", help="Descargar el RAF-DB compartido por el profesor")
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--sha256", help="Verificación opcional contra un hash conocido")
    parser.set_defaults(func=run)
