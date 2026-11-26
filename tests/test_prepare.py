"""Data-quality and split-integrity checks on small, generated image corpora."""
import argparse
import csv
import json
import random
import shutil

import pytest
from PIL import Image

from emotion_recognition.constants import CLASS_NAMES
from emotion_recognition.prepare import PreparationError, add_parser, prepare_dataset


def make_image(path, seed, size=(40, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    image = Image.new("RGB", size)
    image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                   for _ in range(size[0] * size[1])])
    image.save(path)


def corpus(tmp_path):
    root = tmp_path / "raw"
    for label, name in enumerate(CLASS_NAMES):
        for index in range(4):
            make_image(root / "train" / name / f"{index}.png", label * 100 + index)
        make_image(root / "test" / name / "test.png", 5000 + label)
    return root


def manifest(output):
    with (output / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def exclusions(output):
    with (output / "exclusions.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_corrupt_small_constant_and_invalid_labels_are_audited(tmp_path):
    root = corpus(tmp_path)
    bad = root / "train" / "neutral" / "broken.jpg"
    bad.write_bytes(b"this is not a jpeg")
    make_image(root / "train" / "neutral" / "tiny.png", 125, size=(12, 12))
    Image.new("RGB", (40, 40), color=(30, 40, 50)).save(root / "train" / "neutral" / "constant.png")
    make_image(root / "train" / "unknown" / "unknown.png", 15)
    output = tmp_path / "processed"
    report = prepare_dataset(root, output, dataset_format="folders")
    reasons = {row["reason"] for row in exclusions(output)}
    assert {"corrupt_or_unsafe_image", "image_too_small", "constant_image", "invalid_class_directory"} <= reasons
    assert report["retained_records"] == 35
    assert bad.read_bytes() == b"this is not a jpeg"
    assert report["source_files_modified"] is False


def test_exact_train_test_duplicate_keeps_test_and_removes_train(tmp_path):
    root = corpus(tmp_path)
    shutil.copyfile(root / "test" / "neutral" / "test.png", root / "train" / "neutral" / "copy.png")
    # Different encoding but identical decoded pixels must also be recognized.
    with Image.open(root / "train" / "fear" / "0.png") as image:
        image.save(root / "train" / "fear" / "same_pixels.bmp")
    output = tmp_path / "processed"
    report = prepare_dataset(root, output, dataset_format="folders")
    rows = manifest(output)
    assert any(row["image_path"] == "test/neutral/test.png" and row["split"] == "test" for row in rows)
    assert not any(row["image_path"] == "train/neutral/copy.png" for row in rows)
    assert len({row["pixel_sha256"] for row in rows}) == len(rows)
    assert "exact_duplicate_of_official_test" in {row["reason"] for row in exclusions(output)}
    assert report["exact_duplicates_between_splits"] == 0
    assert report["official_test_exclusions"] == 0


def test_conflicting_labels_exclude_entire_exact_group(tmp_path):
    root = corpus(tmp_path)
    shutil.copyfile(root / "train" / "fear" / "0.png", root / "train" / "anger" / "conflict.png")
    output = tmp_path / "processed"
    report = prepare_dataset(root, output, dataset_format="folders")
    conflicts = [row for row in exclusions(output) if row["reason"] == "conflicting_labels_exact_duplicate"]
    assert len(conflicts) == 2
    assert report["conflicting_label_groups"] == 1
    assert not any(row["image_path"] in {"train/fear/0.png", "train/anger/conflict.png"} for row in manifest(output))


def test_split_is_deterministic_stratified_and_has_no_exact_leakage(tmp_path):
    root = corpus(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    prepare_dataset(root, first, dataset_format="folders", seed=89)
    prepare_dataset(root, second, dataset_format="folders", seed=89)
    assert (first / "manifest.csv").read_bytes() == (second / "manifest.csv").read_bytes()
    rows = manifest(first)
    for label in range(len(CLASS_NAMES)):
        assert {row["split"] for row in rows if int(row["label"]) == label} == {"train", "val", "test"}
    by_split = {split: {row["pixel_sha256"] for row in rows if row["split"] == split}
                for split in ("train", "val", "test")}
    assert not by_split["train"] & by_split["val"]
    assert not by_split["train"] & by_split["test"]
    assert not by_split["val"] & by_split["test"]


def test_raf_prefers_aligned_and_invalid_annotations_are_excluded(tmp_path):
    root = tmp_path / "raw"
    label_file = root / "basic" / "EmoLabel" / "list_patition_label.txt"
    label_file.parent.mkdir(parents=True)
    lines = []
    for label in range(1, 8):
        for index in range(3):
            name = f"train_{label * 10 + index:05d}.jpg"
            lines.append(f"{name} {label}")
            make_image(root / "basic" / "Image" / "aligned" / name.replace(".jpg", "_aligned.jpg"), label * 100 + index)
        name = f"test_{label:04d}.jpg"
        lines.append(f"{name} {label}")
        make_image(root / "basic" / "Image" / "aligned" / name.replace(".jpg", "_aligned.jpg"), 1000 + label)
    lines.extend(["train_99999.jpg 99", "train_99998.jpg 1", "../outside.jpg 1"])
    label_file.write_text("\n".join(lines), encoding="utf-8")
    output = tmp_path / "processed"
    report = prepare_dataset(root, output)
    assert report["retained_records"] == 28
    assert all("_aligned.jpg" in row["image_path"] for row in manifest(output))
    reasons = {row["reason"] for row in exclusions(output)}
    assert "invalid_label" in reasons
    assert "missing_image" in reasons
    assert any(reason.startswith("invalid_annotation_line:") for reason in reasons)
    assert report["source_labels_sha256"]


def test_insufficient_clean_data_fails_and_writes_diagnostic(tmp_path):
    root = corpus(tmp_path)
    for index in range(1, 4):
        (root / "train" / "fear" / f"{index}.png").unlink()
    output = tmp_path / "processed"
    output.mkdir()
    (output / "manifest.csv").write_text("stale manifest")
    with pytest.raises(PreparationError, match="fear"):
        prepare_dataset(root, output, dataset_format="folders")
    assert not (output / "manifest.csv").exists()
    report = json.loads((output / "cleaning_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "invalid_experiment"


def test_near_duplicate_similarity_is_only_a_review_signal(tmp_path):
    root = corpus(tmp_path)
    # A one-pixel perturbation has different exact content but the same coarse dHash.
    with Image.open(root / "test" / "neutral" / "test.png") as image:
        altered = image.copy()
    pixel = altered.getpixel((0, 0))
    altered.putpixel((0, 0), ((pixel[0] + 1) % 256, pixel[1], pixel[2]))
    altered.save(root / "train" / "neutral" / "near.png")
    output = tmp_path / "processed"
    report = prepare_dataset(root, output, dataset_format="folders", near_policy="report")
    assert any(row["image_path"] == "train/neutral/near.png" for row in manifest(output))
    assert report["requires_manual_near_duplicate_review"] is True
    assert report["near_duplicate_cross_split_pairs_reported"] >= 1
    assert report["retained_records"] == 36


def test_default_exact_policy_keeps_similar_distinct_images_without_scanning(tmp_path, monkeypatch):
    root = corpus(tmp_path)
    # Similarity to an official test image must not quarantine a distinct train image.
    perturb_image(root / "test" / "neutral" / "test.png", root / "train" / "neutral" / "near.png", 1)
    # Similar train images must also have their own exact identities, not a dHash group.
    perturb_image(root / "train" / "fear" / "0.png", root / "train" / "fear" / "near.png", 1)

    def unexpected_perceptual_scan(*args, **kwargs):
        pytest.fail("The exact-only policy must not perform a perceptual scan")

    monkeypatch.setattr("emotion_recognition.prepare._near_components", unexpected_perceptual_scan)
    output = tmp_path / "processed"
    report = prepare_dataset(root, output, dataset_format="folders")
    rows = manifest(output)
    paths = {row["image_path"] for row in rows}
    assert {"test/neutral/test.png", "train/neutral/near.png", "train/fear/0.png", "train/fear/near.png"} <= paths
    assert report["retained_records"] == 37
    assert report["excluded_records"] == 0
    assert report["parameters"]["near_policy"] == "exact"
    assert report["status"] == "ready"
    assert report["quarantined_similar_to_test"] == 0
    assert report["requires_manual_near_duplicate_review"] is False
    assert report["near_duplicate_scan_performed"] is False
    assert report["perceptual_groups_isolated"] is False
    assert report["perceptual_similarity_cross_split_pairs_after_isolation"] is None
    assert report["perceptual_similarity_groups"] is None
    assert report["manifest_group_policy"] == "exact_pixel_identity"
    assert all(row["group_id"] == row["pixel_sha256"] for row in rows)
    assert len({row["group_id"] for row in rows}) == len(rows)
    with (output / "near_duplicates.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == []


def test_prepare_cli_defaults_to_exact_deduplication():
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["prepare", "--data-root", "raw", "--output-dir", "processed"])
    assert args.near_policy == "exact"
    assert args.seed == 42
    assert args.val_fraction == 0.2


def test_ambiguous_raf_annotation_files_require_an_explicit_choice(tmp_path):
    root = tmp_path / "raw"
    for name in ("copy_a", "copy_b"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "list_patition_label.txt").write_text("train_00001.jpg 1\n")
    with pytest.raises(PreparationError, match="--labels"):
        prepare_dataset(root, tmp_path / "processed")


def test_duplicate_official_test_is_audited_as_a_changed_test_protocol(tmp_path):
    root = corpus(tmp_path)
    shutil.copyfile(root / "test" / "fear" / "test.png", root / "test" / "fear" / "z_copy.png")
    output = tmp_path / "processed"
    report = prepare_dataset(root, output, dataset_format="folders")
    assert report["official_test_exclusions"] == 1
    assert report["retained_records"] == 35
    assert "CLEANED" in report["test_protocol"]


def test_near_duplicate_report_limit_is_explicit(tmp_path):
    root = corpus(tmp_path)
    for index in range(3):
        with Image.open(root / "test" / "neutral" / "test.png") as source:
            image = source.copy()
        pixel = image.getpixel((0, 0))
        image.putpixel((0, 0), ((pixel[0] + index + 1) % 256, pixel[1], pixel[2]))
        image.save(root / "train" / "neutral" / f"similar_{index}.png")
    output = tmp_path / "processed"
    report = prepare_dataset(root, output, dataset_format="folders", max_near_pairs=1, near_policy="report")
    assert report["near_duplicate_scan_truncated"] is True
    assert report["requires_manual_near_duplicate_review"] is True
    assert report["near_duplicate_pairs_reported"] == 1
    assert report["retained_records"] == 38


def perturb_image(source_path, destination_path, offset):
    with Image.open(source_path) as source:
        image = source.copy()
    pixel = image.getpixel((0, 0))
    image.putpixel((0, 0), ((pixel[0] + offset) % 256, pixel[1], pixel[2]))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination_path)


def test_group_policy_quarantines_train_components_touching_official_test(tmp_path):
    root = corpus(tmp_path)
    source = root / "test" / "neutral" / "test.png"
    perturb_image(source, root / "train" / "neutral" / "near.png", 1)
    perturb_image(source, root / "train" / "fear" / "near_other_label.png", 2)
    output = tmp_path / "processed"
    report = prepare_dataset(root, output, dataset_format="folders", near_policy="group")
    rows = manifest(output)
    assert report["quarantined_similar_to_test"] == 2
    assert report["perceptual_groups_isolated"] is True
    assert report["near_duplicate_cross_split_pairs_reported"] == 0
    assert report["requires_manual_near_duplicate_review"] is False
    assert report["official_test_exclusions"] == 0
    assert any(row["image_path"] == "test/neutral/test.png" for row in rows)
    assert not any("near" in row["image_path"] for row in rows)
    reasons = [row for row in exclusions(output) if row["reason"] == "perceptual_similarity_to_official_test"]
    assert len(reasons) == 2
    assert report["conflicting_label_groups"] == 0


def test_group_policy_keeps_similar_train_images_in_one_partition(tmp_path):
    root = corpus(tmp_path)
    source = root / "train" / "neutral" / "0.png"
    perturb_image(source, root / "train" / "neutral" / "near.png", 1)
    perturb_image(source, root / "train" / "fear" / "near_other_label.png", 2)
    first, second = tmp_path / "first", tmp_path / "second"
    report = prepare_dataset(root, first, dataset_format="folders", seed=123, near_policy="group")
    prepare_dataset(root, second, dataset_format="folders", seed=123, near_policy="group")
    rows = manifest(first)
    component_paths = {"train/neutral/0.png", "train/neutral/near.png", "train/fear/near_other_label.png"}
    component_rows = [row for row in rows if row["image_path"] in component_paths]
    assert len(component_rows) == 3
    assert len({row["group_id"] for row in component_rows}) == 1
    assert len({row["split"] for row in component_rows}) == 1
    assert report["quarantined_similar_to_test"] == 0
    assert (first / "manifest.csv").read_bytes() == (second / "manifest.csv").read_bytes()
    for name in CLASS_NAMES:
        assert {row["split"] for row in rows if row["class_name"] == name} >= {"train", "val"}
    assert report["near_duplicate_cross_split_pairs_reported"] == 0


def test_group_policy_fails_when_similarity_prevents_class_coverage(tmp_path):
    root = corpus(tmp_path)
    source = root / "train" / "fear" / "0.png"
    for index in range(1, 4):
        perturb_image(source, root / "train" / "fear" / f"{index}.png", index)
    output = tmp_path / "processed"
    with pytest.raises(PreparationError, match="fear"):
        prepare_dataset(root, output, dataset_format="folders", near_policy="group")
    assert not (output / "manifest.csv").exists()
    report = json.loads((output / "cleaning_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "invalid_experiment"


def test_group_policy_rejects_incomplete_similarity_scan(tmp_path):
    root = corpus(tmp_path)
    source = root / "test" / "neutral" / "test.png"
    for index in range(3):
        perturb_image(source, root / "train" / "neutral" / f"near_{index}.png", index + 1)
    output = tmp_path / "processed"
    with pytest.raises(PreparationError, match="max-near-pairs"):
        prepare_dataset(root, output, dataset_format="folders", max_near_pairs=1, near_policy="group")
    assert not (output / "manifest.csv").exists()
    report = json.loads((output / "cleaning_report.json").read_text(encoding="utf-8"))
    assert report["near_duplicate_scan_truncated"] is True
    assert report["status"] == "invalid_experiment"
