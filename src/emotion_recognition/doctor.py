import importlib.metadata
import json
import platform


def run(args):
    import torch

    packages = ("torch", "torchvision", "Pillow", "numpy", "pandas", "scikit-learn",
                "scikit-image", "matplotlib")
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if result["cuda_available"]:
        tensor = torch.ones((8, 8), device="cuda")
        result["cuda_tensor_check"] = float((tensor @ tensor).sum().item()) == 512.0
    print(json.dumps(result, indent=2, ensure_ascii=False))


def add_parser(subparsers):
    parser = subparsers.add_parser("doctor", help="Comprobar dependencias y GPU")
    parser.set_defaults(func=run)
