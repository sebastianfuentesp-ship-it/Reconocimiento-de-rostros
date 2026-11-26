import argparse
import hashlib
import io
import json
from pathlib import Path

import pytest

from emotion_recognition import pretrained


@pytest.fixture
def payload(monkeypatch):
    data = b"\x80\x02fixture-pesos-prueba"
    monkeypatch.setattr(pretrained, "MSCELEB_SIZE_BYTES", len(data))
    monkeypatch.setattr(pretrained, "MSCELEB_SHA256", hashlib.sha256(data).hexdigest())
    return data


class FakeResponse(io.BytesIO):
    def __init__(self, data, content_type="application/octet-stream", status=200):
        super().__init__(data)
        self.headers = {"Content-Type": content_type}
        self.status = status

    def getcode(self):
        return self.status


def mock_download(monkeypatch, data, **response_args):
    calls = []

    def open_url(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse(data, **response_args)

    monkeypatch.setattr(pretrained, "urlopen", open_url)
    return calls


def test_verify_reports_provenance_without_deserializing(tmp_path, payload):
    path = tmp_path / "weights.pth"
    path.write_bytes(payload)
    result = pretrained.verify_msceleb_weights(path)
    assert result["path"] == str(path.resolve())
    assert result["source"] == "msceleb"
    assert result["dataset"] == "MS-Celeb-1M (según autores de DAN)"
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["size_bytes"] == len(payload)
    assert result["source_url"] == pretrained.MSCELEB_URL
    assert result["reference_url"] == pretrained.MSCELEB_REFERENCE_URL
    assert "no es un hash publicado" in result["hash_note"]
    assert "auditoría independiente" in result["attribution_note"]


@pytest.mark.parametrize("bad_data", [b"short", b"x" * 23])
def test_verify_rejects_wrong_size(tmp_path, payload, bad_data):
    path = tmp_path / "weights.pth"
    path.write_bytes(bad_data)
    with pytest.raises(ValueError, match="Tamaño"):
        pretrained.verify_msceleb_weights(path)


def test_verify_rejects_wrong_hash(tmp_path, payload):
    path = tmp_path / "weights.pth"
    path.write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="SHA256"):
        pretrained.verify_msceleb_weights(path)


def test_download_publishes_verified_file_and_receipt(tmp_path, monkeypatch, payload):
    calls = mock_download(monkeypatch, payload)
    replacements = []
    original_replace = Path.replace

    def checked_replace(source, destination):
        if source.name.endswith(".pth.part"):
            assert source.read_bytes() == payload
            assert not Path(destination).exists()
            replacements.append((source, destination))
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", checked_replace)
    target = pretrained.download_msceleb(tmp_path / "nested")
    assert target.is_absolute()
    assert target.read_bytes() == payload
    assert calls == [(pretrained.MSCELEB_URL, 60)]
    assert len(replacements) == 1
    receipt = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    assert receipt == pretrained.verify_msceleb_weights(target)
    assert not list(target.parent.glob("*.part"))


def test_cached_valid_weights_are_preserved_without_network(tmp_path, monkeypatch, payload):
    target = tmp_path / pretrained.MSCELEB_FILENAME
    target.write_bytes(payload)
    original_mtime = target.stat().st_mtime_ns

    def no_network(*args, **kwargs):
        pytest.fail("Los pesos válidos existentes no deben descargarse otra vez.")

    monkeypatch.setattr(pretrained, "urlopen", no_network)
    assert pretrained.download_msceleb(tmp_path) == target.resolve()
    assert target.stat().st_mtime_ns == original_mtime
    assert target.with_suffix(".json").exists()


def test_invalid_cached_weights_are_not_overwritten(tmp_path, monkeypatch, payload):
    target = tmp_path / pretrained.MSCELEB_FILENAME
    corrupted = b"x" * len(payload)
    target.write_bytes(corrupted)
    calls = mock_download(monkeypatch, payload)
    with pytest.raises(ValueError, match="SHA256"):
        pretrained.download_msceleb(tmp_path)
    assert target.read_bytes() == corrupted
    assert calls == []
    assert not target.with_suffix(".json").exists()


@pytest.mark.parametrize("content_type", ["text/html; charset=utf-8", "application/octet-stream"])
def test_html_response_is_rejected(tmp_path, monkeypatch, payload, content_type):
    mock_download(monkeypatch, b"<!DOCTYPE html><html>Drive warning</html>",
                  content_type=content_type)
    with pytest.raises(ValueError, match="HTML"):
        pretrained.download_msceleb(tmp_path)
    assert not (tmp_path / pretrained.MSCELEB_FILENAME).exists()
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("kind", ["short", "oversized", "wrong_hash"])
def test_invalid_download_never_becomes_final_weights(tmp_path, monkeypatch, payload, kind):
    data = {"short": payload[:-1], "oversized": payload + b"x",
            "wrong_hash": b"x" * len(payload)}[kind]
    mock_download(monkeypatch, data)
    with pytest.raises(ValueError):
        pretrained.download_msceleb(tmp_path)
    assert not (tmp_path / pretrained.MSCELEB_FILENAME).exists()
    assert not list(tmp_path.iterdir())


def test_partial_http_response_is_rejected(tmp_path, monkeypatch, payload):
    mock_download(monkeypatch, payload, status=206)
    with pytest.raises(ValueError, match="HTTP 200"):
        pretrained.download_msceleb(tmp_path)
    assert not list(tmp_path.iterdir())


def test_interrupted_download_cleans_partial_file(tmp_path, monkeypatch, payload):
    class InterruptedResponse(FakeResponse):
        def read(self, size):
            if self.tell():
                raise TimeoutError("Interrupción simulada")
            return super().read(1)

    monkeypatch.setattr(pretrained, "urlopen", lambda *a, **kw: InterruptedResponse(payload))
    with pytest.raises(TimeoutError):
        pretrained.download_msceleb(tmp_path)
    assert not list(tmp_path.iterdir())


def test_cli_default_only_registers_download_command(tmp_path, monkeypatch, payload, capsys):
    parser = argparse.ArgumentParser()
    pretrained.add_parser(parser.add_subparsers(required=True))
    args = parser.parse_args(["pretrained"])
    assert args.output_dir == "data/pretrained"
    assert args.func is pretrained.run
    args = parser.parse_args(["pretrained", "--output-dir", str(tmp_path)])
    mock_download(monkeypatch, payload)
    args.func(args)
    assert "no entrena ni evalúa" in capsys.readouterr().out
