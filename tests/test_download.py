from zipfile import ZipFile

import pytest

from emotion_recognition.download import safe_extract


@pytest.mark.parametrize('name', ['../escape.txt', '/absolute.txt', 'C:/escape.txt', 'a/../../x'])
def test_zip_rejects_paths_outside_destination(tmp_path, name):
    archive = tmp_path / 'bad.zip'
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr('safe.txt', 'okay')
        zipped.writestr(name, 'bad')
    destination = tmp_path / 'extracted'
    with pytest.raises(ValueError, match='insegura'):
        safe_extract(archive, destination)
    assert not destination.exists()  # Todas las rutas se validan antes de escribir.


def test_zip_extracts_expected_file(tmp_path):
    archive = tmp_path / 'good.zip'
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr('basic/labels.txt', 'train_00001.jpg 1')
    destination = tmp_path / 'extracted'
    safe_extract(archive, destination)
    assert (destination / 'basic/labels.txt').read_text() == 'train_00001.jpg 1'
