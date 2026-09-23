"""Independent, resumable construction of the separate forecast data package."""

from __future__ import annotations

from pathlib import Path

from . import acquire_atmosphere, acquire_competition, acquire_gefs, acquire_seasonal
from .data import OBSERVATIONS, verify
from .io import checked, read
from .raw import MEMBERS, dates
from .runtime import atomic_json, log, sha256


def _expected(output: Path) -> list[Path]:
    names = [
        "competition_originals.json", "sample_submission.csv", "training.json", "source_terms.json",
        "raw/coordinates.npz", "raw/test_atmosphere.npy", "audit/recipes.json",
        "audit/seasonal.npz",
    ]
    names += [f"raw/observations/{name}.npy" for name in OBSERVATIONS]
    for target in dates():
        for centre in ("ecmwf", "dwd", "meteo_france"):
            names += [f"raw/seasonal/{centre}/{target}{suffix}" for suffix in (".npz", ".json")]
        names += [f"raw/seasonal_atmosphere/{target}{suffix}" for suffix in (".npz", ".json")]
        if target >= "2007-01":
            for member in MEMBERS:
                names += [f"raw/gefs/{target}_{member}{suffix}" for suffix in (".npz", ".json")]
            names += [f"weather/gefs/{target}{suffix}" for suffix in (".npz", ".json")]
        if target >= "2023-01":
            names += [f"forecast/seasonal_precipitation/{target}{suffix}" for suffix in (".npz", ".json")]
    inventory = read(Path(__file__).parent / "config/acquisition_inventory.json")
    for centre, systems in inventory["calibration_initializations"].items():
        for system, initializations in systems.items():
            for init in initializations:
                names += [f"audit/calibration/{centre}/{system}/{init}{suffix}" for suffix in (".npz", ".json")]
    return [output / name for name in names]


def verify_sources(data: Path, cache: Path) -> int:
    """Rehash source GRIBs and NOAA byte ranges retained outside Git."""
    cache = Path(cache).resolve()
    receipts = read(Path(data) / "source_receipts.json")["files"]
    for name, digest in receipts.items():
        path = (cache / name).resolve()
        if not path.is_relative_to(cache) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or changed source bytes: {name}")
    return len(receipts)


def finalize(output: Path, cache: Path) -> dict:
    output, cache = Path(output), Path(cache)
    if (output / "manifest.json").exists():
        return verify(output)
    recipes = read(Path(__file__).parent / "config/recipes.json")
    training = read(Path(__file__).parent / "config/training.json")
    terms = read(Path(__file__).parent / "config/source_terms.json")
    atomic_json(output / "audit/recipes.json", recipes)
    atomic_json(output / "training.json", training)
    atomic_json(output / "source_terms.json", terms)
    expected = _expected(output)
    missing = [str(path.relative_to(output)) for path in expected if not path.is_file()]
    if missing:
        raise ValueError(f"Independent acquisition incomplete: {len(missing)} files; first: {missing[:8]}")
    for path in expected:
        if path.suffix == ".npz" and path.with_suffix(".json").exists():
            checked(path)
    audit_files = [p for p in expected if p.is_relative_to(output / "audit")]
    atomic_json(output / "audit/manifest.json", {"files": {str(p.relative_to(output / "audit")): sha256(p) for p in audit_files}})
    source_files = {}
    for marker in sorted(cache.rglob("*.json")):
        record = read(marker)
        if "sha256" not in record:
            continue
        raw = marker.with_suffix("")
        if not raw.is_file() or sha256(raw) != record["sha256"]:
            raise ValueError(f"Source cache changed: {raw}")
        source_files[str(raw.relative_to(cache))] = record["sha256"]
    if not source_files:
        raise ValueError("No independently downloaded source bytes were retained")
    atomic_json(output / "source_receipts.json", {"files": source_files})
    atomic_json(output / "provenance.json", {
        "acquisition": "Independent CDS and NOAA retrieval; participant-supplied official Kaggle files",
        "observation_end": "2022-12",
        "original_download_reproduced": False,
        "note": "Kaggle original files were imported and hashed; CDS/NOAA were retrieved here. Retrospective forecast publication at historical origin remains unverified.",
    })
    names = [str(p.relative_to(output)) for p in expected]
    names += ["audit/manifest.json", "source_receipts.json", "provenance.json"]
    atomic_json(output / "manifest.json", {
        "format": 1, "observed_precipitation_end": "2022-12",
        "files": {name: sha256(output / name) for name in names},
    })
    log(f"Independent forecast data package complete: {len(names)} files")
    return verify(output)


def run(output: Path, cache: Path, originals: Path | None, stage: str) -> dict:
    output, cache = Path(output), Path(cache)
    if stage in ("all", "competition"):
        if originals is None:
            raise ValueError("--competition-dir is required for original Kaggle files")
        acquire_competition.convert(originals, output)
    if stage in ("all", "seasonal"):
        acquire_seasonal.acquire(output, cache)
        acquire_seasonal.aggregates(output)
    if stage in ("all", "atmosphere"):
        acquire_atmosphere.acquire(output, cache)
    if stage in ("all", "gefs"):
        acquire_gefs.acquire(output, cache)
    if stage in ("all", "finalize"):
        return finalize(output, cache)
    return {"stage": stage, "complete": True, "manifest": None}
