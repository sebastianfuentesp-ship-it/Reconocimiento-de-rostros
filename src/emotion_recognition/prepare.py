"""Preparar RAF-DB sin modificar las fotografías originales.

Pasos: verificar imágenes y etiquetas, identificar duplicados exactos por píxeles
RGB, preservar el test oficial y dividir el train de forma estratificada.
El protocolo activo usa near_policy='exact': conserva fotos distintas similares.
Las políticas perceptuales opcionales se mantienen para reproducir auditorías.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import struct
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .constants import CLASS_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MANIFEST_FIELDS = ["image_path", "label", "class_name", "source_split", "sha256", "pixel_sha256", "group_id", "split"]
ALIASES = {
    "surprise": "surprise", "surprised": "surprise", "fear": "fear", "fearful": "fear",
    "disgust": "disgust", "disgusted": "disgust", "happy": "happiness", "happiness": "happiness",
    "sad": "sadness", "sadness": "sadness", "angry": "anger", "anger": "anger", "neutral": "neutral",
}


class PreparationError(ValueError):
    """The audit cannot produce a valid seven-class experiment."""


@dataclass
class Entry:
    image_path: str
    label: int | None
    source_split: str
    path: Path | None = None
    error: str | None = None
    sha256: str = ""
    pixel_sha256: str = ""
    dhash: int = 0
    width: int = 0
    height: int = 0
    split: str = ""
    group_id: str = ""

    def row(self) -> dict:
        return {"image_path": self.image_path, "label": self.label,
                "class_name": CLASS_NAMES[self.label] if self.label is not None else "",
                "source_split": self.source_split, "sha256": self.sha256,
                "pixel_sha256": self.pixel_sha256, "group_id": self.group_id, "split": self.split}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PreparationError(f"La imagen queda fuera de --data-root: {path}") from exc
    return resolved


def _discover_labels(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        selected = explicit if explicit.is_absolute() else root / explicit
        if not selected.is_file():
            raise PreparationError(f"No existe el archivo de etiquetas: {selected}")
        return selected.resolve()
    candidates = sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower() in {
        "list_patition_label.txt", "list_partition_label.txt"})
    basic = [p for p in candidates if any(part.lower() == "basic" for part in p.relative_to(root).parts)]
    candidates = basic or candidates
    if len(candidates) != 1:
        found = ", ".join(str(p) for p in candidates) or "ninguno"
        raise PreparationError("Indique --labels para elegir el subconjunto básico de RAF-DB. "
                               f"Archivos encontrados: {found}")
    return candidates[0]


def _raf_entries(root: Path, labels_path: Path, images_dir: Path | None) -> list[Entry]:
    search_root = root if images_dir is None else (images_dir if images_dir.is_absolute() else root / images_dir)
    search_root = _inside(root, search_root)
    if not search_root.is_dir():
        raise PreparationError(f"No existe la carpeta de imágenes: {search_root}")
    index: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(search_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.name.lower()].append(path)
    entries = []
    for line_no, line in enumerate(labels_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        name = fields[0]
        entry = Entry(name, None, "")
        match = re.fullmatch(r"(train|test)_\d+(?:_aligned)?\.(?:jpe?g|png|bmp|webp|tiff?)", name, flags=re.IGNORECASE)
        if len(fields) != 2 or not match:
            entry.error = f"invalid_annotation_line:{line_no}"
            entries.append(entry)
            continue
        entry.source_split = match.group(1).lower()
        try:
            original_label = int(fields[1])
            if not 1 <= original_label <= len(CLASS_NAMES):
                raise ValueError
            entry.label = original_label - 1
        except ValueError:
            entry.error = "invalid_label"
            entries.append(entry)
            continue
        stem, suffix = Path(name).stem, Path(name).suffix
        aligned_name = name if stem.lower().endswith("_aligned") else f"{stem}_aligned{suffix}"
        matches = index.get(aligned_name.lower(), []) or index.get(name.lower(), [])
        if len(matches) > 1:
            basic = [p for p in matches if any(part.lower() == "basic" for part in p.relative_to(root).parts)]
            matches = basic or matches
        if len(matches) != 1:
            entry.error = "missing_image" if not matches else "ambiguous_image_use_images_dir"
        else:
            entry.path = _inside(root, matches[0])
            entry.image_path = entry.path.relative_to(root).as_posix()
        entries.append(entry)
    return entries


def _folder_entries(root: Path) -> list[Entry]:
    if any((root / name).exists() for name in ("val", "validation")):
        raise PreparationError("El formato folders espera train/test. Hay una validación existente; "
                               "conserve su protocolo y prepare un manifiesto explícito antes de continuar.")
    entries = []
    for source_split in ("train", "test"):
        split_dir = root / source_split
        if not split_dir.is_dir():
            raise PreparationError(f"Falta la carpeta {split_dir}")
        for path in sorted(split_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(split_dir)
            class_name = ALIASES.get(relative.parts[0].lower()) if len(relative.parts) > 1 else None
            entries.append(Entry(path.relative_to(root).as_posix(),
                                 CLASS_NAMES.index(class_name) if class_name else None,
                                 source_split, _inside(root, path),
                                 None if class_name else "invalid_class_directory"))
    return entries


def _inspect(entry: Entry, min_size: int) -> None:
    assert entry.path is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(entry.path) as image:
            image.verify()
        with Image.open(entry.path) as image:
            image.load()
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            entry.width, entry.height = rgb.size
            if min(rgb.size) < min_size:
                entry.error = "image_too_small"
                return
            if all(low == high for low, high in rgb.getextrema()):
                entry.error = "constant_image"
                return
            entry.sha256 = _sha256(entry.path)
            entry.pixel_sha256 = hashlib.sha256(struct.pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()
            small = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = small.tobytes()
            value = 0
            for row in range(8):
                for col in range(8):
                    value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
            entry.dhash = value


class _BKTree:
    """Hamming-distance index for 64-bit difference hashes."""
    def __init__(self):
        self.root = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value: int, radius: int):
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (node[0] ^ value).bit_count()
            if distance <= radius:
                for index in node[1]:
                    yield index, distance
            stack.extend(child for edge, child in node[2].items()
                         if distance - radius <= edge <= distance + radius)


def _write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _counts(entries: Iterable[Entry], split_attr: str) -> dict:
    counts = defaultdict(Counter)
    for entry in entries:
        if entry.label is not None:
            counts[getattr(entry, split_attr)][CLASS_NAMES[entry.label]] += 1
    return {split: {name: counter[name] for name in CLASS_NAMES} for split, counter in sorted(counts.items())}


def _near_components(entries: list[Entry], distance: int, max_pairs: int):
    """Find bounded dHash candidate pairs and their transitive components."""
    parents = list(range(len(entries)))

    def find(index):
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    tree = _BKTree()
    pairs = []
    truncated = False
    for index, entry in enumerate(entries):
        for other_index, hamming in tree.query(entry.dhash, distance):
            if len(pairs) >= max_pairs:
                truncated = True
                break
            pairs.append((other_index, index, hamming))
            union(other_index, index)
        if truncated:
            break
        tree.add(entry.dhash, index)
    groups = defaultdict(list)
    for index, entry in enumerate(entries):
        groups[find(index)].append(entry)
    for group in groups.values():
        # Include every original component member, including any later quarantine.
        identity = json.dumps(sorted(entry.image_path for entry in group), ensure_ascii=False, separators=(",", ":"))
        group_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        for entry in group:
            entry.group_id = group_id
    return list(groups.values()), pairs, truncated


def _assign_grouped_holdout(groups: list[list[Entry]], fraction: float, seed: int) -> dict[str, int]:
    """Greedily approximate class targets without separating a similarity group.

    Coverage is mandatory in both partitions. Seeded retries resolve ambiguous
    coverage choices; impossible or unsuccessful assignments fail explicitly.
    """
    vectors = [Counter(entry.label for entry in group) for group in groups]
    totals = Counter(entry.label for group in groups for entry in group)
    group_frequency = Counter(label for vector in vectors for label in vector)
    impossible = [CLASS_NAMES[label] for label in range(len(CLASS_NAMES)) if group_frequency[label] < 2]
    if impossible:
        raise PreparationError("No se pueden separar grupos conservando todas las clases en train/val: "
                               + ", ".join(impossible) + " necesita al menos dos grupos de entrenamiento")
    targets = {label: min(totals[label] - 1, max(1, math.ceil(totals[label] * fraction)))
               for label in range(len(CLASS_NAMES))}

    def loss(counts):
        return sum(((counts[label] - targets[label]) / max(1, targets[label])) ** 2
                   for label in range(len(CLASS_NAMES)))

    for attempt in range(32):
        order = list(range(len(groups)))
        random.Random(seed + attempt).shuffle(order)
        rank = {index: position for position, index in enumerate(order)}
        chosen = set()
        counts = Counter()
        missing = set(range(len(CLASS_NAMES)))
        while missing:
            candidates = []
            for index in order:
                vector = vectors[index]
                if index in chosen or not missing.intersection(vector):
                    continue
                if any(totals[label] - counts[label] - amount < 1 for label, amount in vector.items()):
                    continue
                coverage = sum(1 / group_frequency[label] for label in missing.intersection(vector))
                candidates.append((coverage, -loss(counts + vector), -rank[index], index))
            if not candidates:
                break
            selected = max(candidates)[-1]
            chosen.add(selected)
            counts.update(vectors[selected])
            missing.difference_update(vectors[selected])
        if missing:
            continue
        # Larger components are handled first; seeded order breaks ties.
        for index in sorted(order, key=lambda item: (-len(groups[item]), rank[item])):
            if index in chosen:
                continue
            vector = vectors[index]
            if any(totals[label] - counts[label] - amount < 1 for label, amount in vector.items()):
                continue
            if loss(counts + vector) < loss(counts):
                chosen.add(index)
                counts.update(vector)
        for index, group in enumerate(groups):
            for entry in group:
                entry.split = "val" if index in chosen else "train"
        return {CLASS_NAMES[label]: count for label, count in targets.items()}
    raise PreparationError("No se encontró una partición por grupos con todas las clases en train/val. "
                           "Revise componentes muy grandes o use --near-policy report con revisión manual.")


def _write_near_pairs(path: Path, entries: list[Entry], pairs, retained: list[Entry]) -> int:
    retained_paths = {entry.image_path for entry in retained}
    rows = []
    cross_count = 0
    for first, second, distance in pairs:
        left, right = entries[first], entries[second]
        keep_left, keep_right = left.image_path in retained_paths, right.image_path in retained_paths
        cross = keep_left and keep_right and left.split != right.split
        cross_count += int(cross)
        rows.append({"image_path_a": left.image_path, "image_path_b": right.image_path,
                     "split_a": left.split or left.source_split, "split_b": right.split or right.source_split,
                     "label_a": left.label, "label_b": right.label, "hamming_distance": distance,
                     "cross_split": cross, "label_conflict": left.label != right.label,
                     "retained_a": keep_left, "retained_b": keep_right,
                     "group_id": left.group_id if left.group_id == right.group_id else ""})
    _write_csv(path, rows, ["image_path_a", "image_path_b", "split_a", "split_b", "label_a", "label_b",
                            "hamming_distance", "cross_split", "label_conflict", "retained_a", "retained_b", "group_id"])
    return cross_count


def prepare_dataset(data_root: str | Path, output_dir: str | Path, *, dataset_format: str = "rafdb",
                    labels_path: str | Path | None = None, images_dir: str | Path | None = None,
                    val_fraction: float = 0.2, seed: int = 42, min_size: int = 32,
                    near_distance: int = 4, max_near_pairs: int = 50000,
                    near_policy: str = "exact") -> dict:
    """Crear manifiesto e informes; detenerse si no se logra una división válida."""
    # PASO 1. Validar rutas y leer etiquetas. La salida queda fuera de los originales.
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    if not root.is_dir():
        raise PreparationError(f"No existe --data-root: {root}")
    if (not 0 < val_fraction < 1 or min_size < 1 or not 0 <= near_distance <= 16 or max_near_pairs < 1
            or near_policy not in {"exact", "group", "report"}):
        raise PreparationError("Parámetros inválidos: 0 < val-fraction < 1, min-size >= 1, "
                               "0 <= near-distance <= 16, max-near-pairs >= 1 y near-policy exact/group/report.")
    # Keep output outside the source tree to prevent a future scan reading generated files.
    if output == root or root in output.parents:
        raise PreparationError("--output-dir debe estar fuera de --data-root para conservar los datos originales.")
    selected_labels = None
    if dataset_format == "rafdb":
        selected_labels = _discover_labels(root, Path(labels_path) if labels_path else None)
        entries = _raf_entries(root, selected_labels, Path(images_dir) if images_dir else None)
    elif dataset_format == "folders":
        entries = _folder_entries(root)
    else:
        raise PreparationError(f"Formato desconocido: {dataset_format}")
    output.mkdir(parents=True, exist_ok=True)
    exclusions = []
    valid = []

    def exclude(entry: Entry, reason: str, kept: str = ""):
        exclusions.append({"image_path": entry.image_path, "label": entry.label,
                           "source_split": entry.source_split, "reason": reason, "kept_image_path": kept})

    # PASO 2. Comprobar lectura, tamaño, etiqueta y hashes de cada imagen.
    print(f"[prepare] Inspeccionando {len(entries)} registros de imágenes...", flush=True)
    for inspected, entry in enumerate(entries, 1):
        if entry.error is None:
            try:
                _inspect(entry, min_size)
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                entry.error = "corrupt_or_unsafe_image"
        if entry.error:
            exclude(entry, entry.error)
        else:
            valid.append(entry)
        if inspected % 1000 == 0 or inspected == len(entries):
            print(f"[prepare] Inspeccionadas {inspected}/{len(entries)}; válidas {len(valid)}; "
                  f"excluidas {len(exclusions)}", flush=True)
    # PASO 3. Agrupar solo identidades exactas de píxeles RGB y conservar una copia.
    # Si una copia pertenece al test oficial, se conserva esa para evitar filtraciones.
    groups = defaultdict(list)
    for entry in valid:
        groups[entry.pixel_sha256].append(entry)
    kept = []
    duplicate_groups = 0
    conflicting_groups = 0
    for group in groups.values():
        if len({entry.label for entry in group}) > 1:
            conflicting_groups += 1
            for entry in group:
                exclude(entry, "conflicting_labels_exact_duplicate")
            continue
        ordered = sorted(group, key=lambda entry: (entry.source_split != "test", entry.image_path))
        keeper = ordered[0]
        kept.append(keeper)
        if len(group) > 1:
            duplicate_groups += 1
        for entry in ordered[1:]:
            reason = ("exact_duplicate_of_official_test" if keeper.source_split == "test" and entry.source_split == "train"
                      else "exact_duplicate_same_source_split")
            exclude(entry, reason, keeper.image_path)

    kept.sort(key=lambda entry: entry.image_path)
    exact_cleaned = list(kept)
    # PASO 4. Con 'exact' se omite la similitud perceptual: no elimina fotos parecidas.
    scan_performed = near_policy != "exact"
    if scan_performed:
        print(f"[prepare] Buscando similitud perceptual en {len(kept)} imágenes; política {near_policy}...", flush=True)
        components, near_pairs, truncated = _near_components(kept, near_distance, max_near_pairs)
        pre_isolation_cross_source = sum(exact_cleaned[a].source_split != exact_cleaned[b].source_split
                                        for a, b, _ in near_pairs)
    else:
        print("[prepare] Solo duplicados exactos: no se excluyen ni agrupan imágenes por similitud.", flush=True)
        components, near_pairs, truncated = [], [], False
        pre_isolation_cross_source = None
    quarantine = []
    if near_policy == "group" and not truncated:
        for component in components:
            test_entries = [entry for entry in component if entry.source_split == "test"]
            if not test_entries:
                continue
            representative = min(entry.image_path for entry in test_entries)
            for entry in component:
                if entry.source_split == "train":
                    quarantine.append(entry)
                    entry.split = "quarantined"
                    exclude(entry, "perceptual_similarity_to_official_test", representative)
        quarantined_paths = {entry.image_path for entry in quarantine}
        kept = [entry for entry in kept if entry.image_path not in quarantined_paths]

    report = {
        "status": "audited", "data_root": str(root), "dataset_format": dataset_format,
        "class_to_index": {name: i for i, name in enumerate(CLASS_NAMES)},
        "parameters": {"val_fraction": val_fraction, "seed": seed, "min_size": min_size,
                       "near_distance": near_distance, "max_near_pairs": max_near_pairs, "near_policy": near_policy},
        "source_labels": str(selected_labels) if selected_labels else None,
        "source_labels_sha256": _sha256(selected_labels) if selected_labels else None,
        "input_records": len(entries), "valid_decoded_images": len(valid), "retained_records": len(kept),
        "before_by_source_split": _counts(entries, "source_split"),
        "after_exact_cleaning_by_source_split": _counts(exact_cleaned, "source_split"),
        "retained_by_source_split": _counts(kept, "source_split"),
        "excluded_records": len(exclusions), "exclusions_by_reason": dict(Counter(row["reason"] for row in exclusions)),
        "exact_duplicate_groups": duplicate_groups, "conflicting_label_groups": conflicting_groups,
        "official_test_exclusions": sum(row["source_split"] == "test" for row in exclusions),
        "test_protocol": "Official train/test membership is preserved. Corrupt, invalid, conflicting-label, "
                         "or redundant exact-duplicate images can be excluded. Report metrics as CLEANED official "
                         "test whenever official_test_exclusions > 0; do not equate them with published benchmark scores.",
        "source_files_modified": False,
        "near_duplicate_policy": (
            "Exact pixel duplicates only. Perceptual similarity is not scanned, used to exclude images, "
            "or used to group train/validation. Similar-looking distinct images are retained. "
            "No identity-disjoint split is guaranteed." if near_policy == "exact" else
            "dHash does not prove photographs are duplicates or identify the same person. In group mode, "
            "connected similarity components are conservatively isolated: training images in components touching "
            "official test are quarantined, and other training components stay wholly in train or validation. "
            "Conservative perceptual similarity isolation may quarantine genuine distinct photographs. "
            "In report mode similarity is only reported for manual review. No identity-disjoint split is guaranteed."),
        "split_policy": (
            "Approximately stratified grouped holdout from cleaned official training data. "
            "No oversampling before split." if near_policy == "group" else
            "Stratified image holdout from exact-deduplicated official training data. "
            "Official test membership is preserved. No perceptual grouping or oversampling before split."),
        "perceptual_similarity_groups": len(components) if scan_performed else None,
        "perceptual_similarity_components_with_multiple_images": (
            sum(len(group) > 1 for group in components) if scan_performed else None),
        "quarantined_similar_to_test": len(quarantine),
        "quarantined_by_class": {name: sum(entry.label == label for entry in quarantine)
                                 for label, name in enumerate(CLASS_NAMES)},
        "near_duplicate_pairs_reported": len(near_pairs),
        "near_duplicate_scan_performed": scan_performed,
        "near_duplicate_cross_source_pairs_before_isolation": pre_isolation_cross_source,
        "near_duplicate_scan_truncated": truncated,
    }
    _write_csv(output / "exclusions.csv", exclusions,
               ["image_path", "label", "source_split", "reason", "kept_image_path"])

    def save_report():
        (output / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def fail(problems):
        report["status"] = "invalid_experiment"
        report["errors"] = problems
        (output / "manifest.csv").unlink(missing_ok=True)
        _write_near_pairs(output / "near_duplicates.csv", exact_cleaned, near_pairs, kept)
        save_report()
        raise PreparationError("; ".join(problems) + f". Consulte {output / 'cleaning_report.json'}")

    if truncated and near_policy == "group":
        fail([("El escaneo perceptual alcanzó --max-near-pairs. Aumente el límite: no se puede garantizar "
               "el aislamiento de grupos con un escaneo incompleto")])
    print(f"[prepare] Similitud: {len(near_pairs)} pares candidatos; {len(quarantine)} imágenes de train "
          "en cuarentena. Creando particiones...", flush=True)
    # PASO 5. Reservar el test y dividir exclusivamente el train por cada clase.
    problems = []
    train_by_label = defaultdict(list)
    for entry in kept:
        if entry.source_split == "train":
            train_by_label[entry.label].append(entry)
        else:
            entry.split = "test"
    for label, class_name in enumerate(CLASS_NAMES):
        if len(train_by_label[label]) < 2:
            problems.append(f"{class_name}: se requieren al menos 2 imágenes distintas de entrenamiento")
    if not any(entry.source_split == "test" for entry in kept):
        problems.append("El conjunto de prueba quedó vacío")
    if problems:
        fail(problems)
    if near_policy == "group":
        train_groups = [group for group in components if all(entry.source_split == "train" for entry in group)]
        try:
            targets = _assign_grouped_holdout(train_groups, val_fraction, seed)
        except PreparationError as exc:
            fail([str(exc)])
        report["validation_target_by_class"] = targets
    else:
        rng = random.Random(seed)
        for label in range(len(CLASS_NAMES)):
            pool = sorted(train_by_label[label], key=lambda entry: entry.image_path)
            rng.shuffle(pool)
            count = min(len(pool) - 1, max(1, math.ceil(len(pool) * val_fraction)))
            for index, entry in enumerate(pool):
                entry.split = "val" if index < count else "train"
        # Exact and report modes do not enforce perceptual groups. Each retained
        # canonical pixel identity is its own manifest group.
        for entry in kept:
            entry.group_id = entry.pixel_sha256
    kept.sort(key=lambda entry: (entry.split, entry.image_path))
    # Every retained canonical pixel identity must occur exactly once, across all partitions.
    if len({entry.pixel_sha256 for entry in kept}) != len(kept) or len({entry.sha256 for entry in kept}) != len(kept):
        raise AssertionError("Internal exact-duplicate audit failed")
    cross_count = _write_near_pairs(output / "near_duplicates.csv", exact_cleaned, near_pairs, kept)
    if near_policy == "group" and cross_count:
        fail(["Falló la verificación interna de aislamiento de similitud entre particiones"])
    grouped_splits = defaultdict(set)
    for entry in kept:
        grouped_splits[entry.group_id].add(entry.split)
    if any(len(splits) > 1 for splits in grouped_splits.values()):
        fail(["Un mismo group_id aparece en distintas particiones"])
    for label, name in enumerate(CLASS_NAMES):
        if not all(any(entry.label == label and entry.split == split for entry in kept) for split in ("train", "val")):
            fail([f"La clase {name} no está representada en train y val"])
    # PASO 6. Tras verificar separación y clases, guardar rutas, etiquetas y auditoría.
    # El aumento de datos ocurre después, al entrenar; aquí no se generan imágenes.
    _write_csv(output / "manifest.csv", (entry.row() for entry in kept), MANIFEST_FIELDS)
    report.update({"status": "ready_for_review" if cross_count or truncated else "ready",
                   "after_by_split": _counts(kept, "split"),
                   "manifest_sha256": _sha256(output / "manifest.csv"),
                   "exclusions_sha256": _sha256(output / "exclusions.csv"),
                   "near_duplicate_cross_split_pairs_reported": cross_count,
                   "requires_manual_near_duplicate_review": bool(cross_count or truncated),
                   "exact_duplicates_between_splits": 0,
                   "perceptual_similarity_cross_split_pairs_after_isolation": cross_count if scan_performed else None,
                   "perceptual_groups_isolated": near_policy == "group" and not truncated,
                   "manifest_group_policy": "transitive_dhash_components" if near_policy == "group" else "exact_pixel_identity"})
    save_report()
    return report


def run(args: argparse.Namespace) -> int:
    report = prepare_dataset(args.data_root, args.output_dir, dataset_format=args.dataset_format,
                             labels_path=args.labels, images_dir=args.images_dir,
                             val_fraction=args.val_fraction, seed=args.seed, min_size=args.min_size,
                             near_distance=args.near_distance, max_near_pairs=args.max_near_pairs,
                             near_policy=args.near_policy)
    print(json.dumps({"status": report["status"], "retained_records": report["retained_records"],
                      "excluded_records": report["excluded_records"],
                      "requires_manual_near_duplicate_review": report["requires_manual_near_duplicate_review"],
                      "output_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False, indent=2))
    return 0


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("prepare", help="Auditar datos y crear particiones sin duplicados exactos")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--format", dest="dataset_format", choices=("rafdb", "folders"), default="rafdb")
    parser.add_argument("--labels", type=Path, help="Archivo de etiquetas RAF-DB, relativo a data-root o absoluto")
    parser.add_argument("--images-dir", type=Path, help="Carpeta de imágenes para resolver archivos ambiguos")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=32)
    parser.add_argument("--near-distance", type=int, default=4)
    parser.add_argument("--max-near-pairs", type=int, default=50000)
    parser.add_argument("--near-policy", choices=("exact", "group", "report"), default="exact",
                        help="exact elimina solo duplicados exactos (predeterminado); group aísla componentes "
                             "perceptuales; report solo señala candidatos para revisión")
    parser.set_defaults(func=run)
