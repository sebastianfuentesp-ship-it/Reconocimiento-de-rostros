"""DACL weight provenance and atomic downloads using tiny offline fixtures."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from emotion_recognition import dacl_pretrained, pretrained


@pytest.fixture
def payload(monkeypatch):
    data = b"\x80\x02dacl-fixture-pretrained"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(pretrained, "MSCELEB_SHA256", digest)
    monkeypatch.setattr(pretrained, "MSCELEB_SIZE_BYTES", len(data))
    monkeypatch.setattr(dacl_pretrained, "DACL_SHA256", digest)
    monkeypatch.setattr(dacl_pretrained, "DACL_SIZE_BYTES", len(data))
    return data


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200, content_type="application/octet-stream"):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Type": content_type}

    def getcode(self):
        return self.status


def mock_download(monkeypatch, data, **kwargs):
    calls = []

    def open_url(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse(data, **kwargs)

    monkeypatch.setattr(dacl_pretrained, "urlopen", open_url)
    return calls


def test_official_dacl_weights_have_same_observed_hash_as_dan():
    assert dacl_pretrained.DACL_SHA256 == (
        "734341508e3ddbcd181e40da0173cf2c77cd0de3bc2455715cc1b7f42a63a6f6"
    )
    assert dacl_pretrained.DACL_SHA256 == pretrained.MSCELEB_SHA256
    assert dacl_pretrained.DACL_SIZE_BYTES == 181_511_212
    assert "16uPBWtSfVOJvlxMnb0xg6sOP3-zhHsx3" in dacl_pretrained.DACL_URL
    assert "c41e9e81de9b70a84cae3d7c19102b5095964d5b" in dacl_pretrained.DACL_REFERENCE_URL


def test_verifier_uses_dacl_provenance_with_verified_shared_bytes(tmp_path, payload):
    path = tmp_path / "weights.pth"
    path.write_bytes(payload)
    result = dacl_pretrained.verify_dacl_weights(path)
    assert result["source"] == "msceleb_dacl"
    assert result["dataset"] == "MS-Celeb-1M (según autores de DACL)"
    assert result["path"] == str(path.resolve())
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["size_bytes"] == len(payload)
    assert result["source_url"] == dacl_pretrained.DACL_URL
    assert result["reference_url"] == dacl_pretrained.DACL_REFERENCE_URL
    assert "auditoría independiente" in result["attribution_note"]
    assert "no es un hash publicado" in result["hash_note"]


@pytest.mark.parametrize("kind", ["wrong_hash", "wrong_size"])
def test_verifier_rejects_invalid_bytes(tmp_path, payload, kind):
    path = tmp_path / "weights.pth"
    path.write_bytes(b"x" * len(payload) if kind == "wrong_hash" else payload[:-1])
    with pytest.raises(ValueError, match="SHA256" if kind == "wrong_hash" else "Tamaño"):
        dacl_pretrained.verify_dacl_weights(path)


def test_download_is_verified_before_atomic_publication(tmp_path, monkeypatch, payload):
    calls = mock_download(monkeypatch, payload)
    original_replace = Path.replace
    published = []

    def verify_before_replace(source, destination):
        if source.name.endswith(".pth.part"):
            assert source.read_bytes() == payload
            assert not Path(destination).exists()
            published.append(source.name)
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", verify_before_replace)
    target = dacl_pretrained.download_dacl(tmp_path / "nested")
    assert target.name == dacl_pretrained.DACL_FILENAME
    assert target.is_absolute()
    assert target.read_bytes() == payload
    assert len(published) == 1
    assert calls == [(dacl_pretrained.DACL_URL, 60)]
    receipt = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    assert receipt == dacl_pretrained.verify_dacl_weights(target)
    assert not list(target.parent.glob("*.part"))


def test_valid_cache_avoids_network_and_preserves_weights(tmp_path, monkeypatch, payload):
    target = tmp_path / dacl_pretrained.DACL_FILENAME
    target.write_bytes(payload)
    original_mtime = target.stat().st_mtime_ns

    def no_network(*args, **kwargs):
        pytest.fail("No se deben descargar otra vez los pesos válidos.")

    monkeypatch.setattr(dacl_pretrained, "urlopen", no_network)
    assert dacl_pretrained.download_dacl(tmp_path) == target.resolve()
    assert target.stat().st_mtime_ns == original_mtime
    assert target.read_bytes() == payload
    receipt = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    assert receipt["source"] == "msceleb_dacl"


def test_invalid_cache_is_never_overwritten(tmp_path, monkeypatch, payload):
    target = tmp_path / dacl_pretrained.DACL_FILENAME
    bad = b"x" * len(payload)
    target.write_bytes(bad)
    calls = mock_download(monkeypatch, payload)
    with pytest.raises(ValueError, match="SHA256"):
        dacl_pretrained.download_dacl(tmp_path)
    assert target.read_bytes() == bad
    assert calls == []


@pytest.mark.parametrize("kind", ["short", "oversized", "wrong_hash"])
def test_bad_download_does_not_publish_weights_or_receipt(tmp_path, monkeypatch, payload, kind):
    data = {"short": payload[:-1], "oversized": payload + b"x",
            "wrong_hash": b"x" * len(payload)}[kind]
    mock_download(monkeypatch, data)
    with pytest.raises(ValueError):
        dacl_pretrained.download_dacl(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status,content_type", [(206, "application/octet-stream"),
                                                (200, "text/html; charset=utf-8")])
def test_incomplete_http_or_html_response_is_rejected(
    tmp_path, monkeypatch, payload, status, content_type
):
    mock_download(monkeypatch, payload, status=status, content_type=content_type)
    with pytest.raises(ValueError, match="Drive"):
        dacl_pretrained.download_dacl(tmp_path)
    assert not list(tmp_path.iterdir())


def test_failed_request_does_not_leave_partial_weights(tmp_path, monkeypatch, payload):
    def fail_request(*args, **kwargs):
        raise URLError("Error de conexión simulado")

    monkeypatch.setattr(dacl_pretrained, "urlopen", fail_request)
    with pytest.raises(URLError):
        dacl_pretrained.download_dacl(tmp_path)
    assert not list(tmp_path.iterdir())


def test_interrupted_stream_is_cleaned_without_publishing(tmp_path, monkeypatch, payload):
    class BrokenResponse(FakeResponse):
        def read(self, size):
            if self.tell():
                raise TimeoutError("Interrupción simulada")
            return super().read(1)

    monkeypatch.setattr(dacl_pretrained, "urlopen", lambda *a, **kw: BrokenResponse(payload))
    with pytest.raises(TimeoutError):
        dacl_pretrained.download_dacl(tmp_path)
    assert not list(tmp_path.iterdir())


def test_dacl_cli_registers_separate_traceable_source(tmp_path, monkeypatch, payload, capsys):
    parser = argparse.ArgumentParser()
    dacl_pretrained.add_parser(parser.add_subparsers(required=True))
    defaults = parser.parse_args(["pretrained-dacl"])
    assert defaults.output_dir == "data/pretrained"
    args = parser.parse_args(["pretrained-dacl", "--output-dir", str(tmp_path)])
    mock_download(monkeypatch, payload)
    assert args.func(args) == 0
    assert "Pesos base DACL verificados" in capsys.readouterr().out
