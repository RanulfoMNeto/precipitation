"""Reproducible local execution and artifact fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint() -> dict[str, str]:
    """Bind an inference receipt to the exact runnable source and lock files."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src/worcap_forecast").rglob("*.py"))
    paths += sorted((root / "src/worcap_forecast/config").glob("*.json"))
    paths += [
        root / name
        for name in (
            "pyproject.toml",
            "SETTINGS.json",
            "requirements.txt",
            "requirements-acquisition.lock",
        )
    ]
    if not all(path.is_file() for path in paths):
        raise ValueError("The installed package does not include its complete source and lock files")
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def environment_receipt() -> dict:
    import lightgbm
    import torch

    device = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "numpy": np.__version__,
        "lightgbm": lightgbm.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_memory_bytes": device.total_memory,
    }


def atomic_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    print(f"{timestamp} {message}", flush=True)


def configure_torch(seed: int, threads: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for neural production runs")
    return torch
