"""Verify the immutable, independently assembled input package."""

from __future__ import annotations

from pathlib import Path

from .io import read
from .runtime import sha256

OBSERVATIONS = (
    "precipitation",
    "air_temperature_2m",
    "cloud_cover",
    "surface_pressure",
    "specific_humidity_850hpa",
    "relative_humidity_850hpa",
    "air_temperature_850hpa",
    "geopotential_850hpa",
    "eastward_wind_850hpa",
    "northward_wind_850hpa",
)


def verify(root):
    root = Path(root).resolve()
    record = read(root / "manifest.json")
    if record["observed_precipitation_end"] != "2022-12":
        raise ValueError("Forbidden precipitation cutoff")
    for name, digest in record["files"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha256(p) != digest:
            raise ValueError(f"Changed or missing input: {name}")
    if record.get("format") == 1:
        from .acquire import _expected

        expected = {
            str(path.relative_to(root)) for path in _expected(root)
        } | {"audit/manifest.json", "source_receipts.json", "provenance.json"}
        if set(record["files"]) != expected:
            raise ValueError("Input package does not match the registered source inventory")
    return record


def verify_originals(root, originals_dir):
    """Check the separately held competition NetCDF files against their receipt."""
    receipt = read(Path(root) / "competition_originals.json")["files"]
    originals_dir = Path(originals_dir)
    for name, record in receipt.items():
        path = originals_dir / name
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Changed competition original: {name}")
    return len(receipt)


def availability_report(root):
    """A declaration of available_by is not proof of historical publication."""
    root = Path(root)
    counts = {
        "publication_evidence": 0,
        "missing_publication_evidence": 0,
        "retrospective_or_unverified": 0,
        "checked": 0,
    }
    examples = []
    for relative in read(root / "manifest.json")["files"]:
        if not relative.endswith(".json") or not relative.startswith(
            (
                "raw/seasonal/",
                "raw/seasonal_atmosphere/",
                "raw/gefs/",
                "weather/gefs/",
                "audit/calibration/",
                "forecast/seasonal_precipitation/",
            )
        ):
            continue
        meta = read(root / relative)
        counts["checked"] += 1
        evidence = meta.get("publication_evidence") or meta.get("publication_record")
        if evidence:
            counts["publication_evidence"] += 1
        else:
            counts["missing_publication_evidence"] += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "file": relative,
                        "target": meta.get("target"),
                        "available_by_claim": meta.get("available_by"),
                    }
                )
        target = meta.get("target", "")
        if (
            (meta.get("kind", "").startswith("retrospective"))
            or (relative.startswith("raw/seasonal/") and target < "2018-01")
            or not evidence
        ):
            counts["retrospective_or_unverified"] += 1
    return {
        "counts": counts,
        "examples": examples,
        "strict_origin_available": counts["retrospective_or_unverified"] == 0,
        "interpretation": "Historical hindcasts and reforecasts need actual publication evidence.",
    }
