"""Verificar el preprocesamiento equivalente y el paquete sin importar PyTorch."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("onnxruntime")
from emotion_recognition.inference_onnx import preprocess
from emotion_recognition.model import build_transforms

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size", [(256, 256), (512, 400), (321, 623), (81, 80), (224, 224), (48, 48)])
def test_preprocessing_matches_training_profile(size):
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    reference = build_transforms(False, 224, "dacl_baseline")(image).unsqueeze(0).numpy()
    np.testing.assert_allclose(preprocess(image), reference, atol=1e-6, rtol=0)


def test_standalone_vercel_package_upload_without_torch_import():
    for name in ("app.js", "styles.css"):
        assert (ROOT / "deploy/vercel/public/static" / name).read_bytes() == (
            ROOT / "deploy/vercel/app/static" / name).read_bytes()
    script = '''
import io, sys
sys.path.insert(0, sys.argv[1])
from PIL import Image
from index import app
assert 'torch' not in sys.modules
client = app.test_client()
assert client.get('/').status_code == 200
assert '4 MB' in client.get('/').text
assert 'Procesamiento en servidor' in client.get('/').text
buffer = io.BytesIO()
Image.new('RGB', (224, 224), 'gray').save(buffer, format='PNG')
buffer.seek(0)
response = client.post('/api/predict', data={'photo': (buffer, 'face.png'), 'cropped': 'true'})
assert response.status_code == 200, response.json
scores = response.json['faces'][0]['scores']
assert len(scores) == 7
assert abs(sum(s['score'] for s in scores) - 1) < 1e-6
assert 'torch' not in sys.modules
assert client.post('/api/predict').status_code == 400
print('Paquete Vercel: carga, inferencia ONNX y ausencia de PyTorch comprobadas.')
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT / "deploy/vercel")],
                            check=False, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
