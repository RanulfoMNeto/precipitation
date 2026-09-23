"""Acquire exactly the CDS seasonal precipitation fields consumed by forecast."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import checked, read
from .raw import dates
from .runtime import atomic_json, log, sha256
from .weather import route

DATASET = "seasonal-monthly-single-levels"
AREA = [16, -91, -61, -24]
CENTRES = {"ecmwf": "ecmf", "dwd": "edzw", "meteo_france": "lfpw"}


def previous(target: str) -> str:
    return str(np.datetime64(target, "M") - np.timedelta64(1, "M"))


def next_month(initialization: str) -> str:
    return str(np.datetime64(initialization, "M") + np.timedelta64(1, "M"))


def vintage(centre: str, system: str, initialization: str) -> tuple[str, int]:
    if centre == "ecmwf":
        return ("hindcast", 25) if initialization <= "2016-12" else ("forecast", 51)
    if centre == "dwd":
        cutoff = "2017-12" if system == "2" else "2019-12"
        return ("hindcast", 30) if initialization <= cutoff else ("forecast", 50)
    last_hindcast = {"6": "2016-12", "7": "2016-12", "8": "2018-12"}[system]
    return ("hindcast", 25) if initialization <= last_hindcast else ("forecast", 51)


def inventory() -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = {}
    for target in dates():
        init = previous(target)
        for centre in CENTRES:
            result.setdefault((centre, route(centre, init)), set()).add(init)
    register = read(Path(__file__).parent / "config/acquisition_inventory.json")
    for centre, systems in register["calibration_initializations"].items():
        for system, initializations in systems.items():
            result.setdefault((centre, system), set()).update(initializations)
    return result


def tasks() -> list[dict]:
    """Group years with identical requested months and vintage, at most four years."""
    result = []
    for (centre, system), initializations in inventory().items():
        calendar: dict[tuple[str, tuple[str, ...]], list[str]] = {}
        for year in sorted({d[:4] for d in initializations}):
            for kind in ("hindcast", "forecast"):
                months = tuple(sorted(d[5:] for d in initializations if d.startswith(year) and vintage(centre, system, d)[0] == kind))
                if months:
                    calendar.setdefault((kind, months), []).append(year)
        for (kind, months), years in calendar.items():
            for start in range(0, len(years), 4):
                block = years[start : start + 4]
                selected = [f"{year}-{month}" for year in block for month in months]
                request = {
                    "originating_centre": centre,
                    "system": system,
                    "variable": ["total_precipitation"],
                    "product_type": ["monthly_mean"],
                    "year": block,
                    "month": list(months),
                    "leadtime_month": ["2"],
                    "data_format": "grib",
                    "area": AREA,
                }
                result.append({"centre": centre, "system": system, "kind": kind, "initializations": selected, "request": request})
    return result


def _grid(ec, message):
    latitude = ec.codes_get_array(message, "latitudes")
    longitude = (ec.codes_get_array(message, "longitudes") + 180) % 360 - 180
    y, x = np.unique(latitude), np.unique(longitude)
    order = np.lexsort((longitude, latitude))
    if (
        len(y) != 77 or len(x) != 67 or len(y) * len(x) != len(latitude)
        or not np.array_equal(latitude[order], np.repeat(y, len(x)))
        or not np.array_equal(longitude[order], np.tile(x, len(y)))
    ):
        raise ValueError("Unexpected or incomplete CDS seasonal grid")
    values = np.asarray(ec.codes_get_values(message), dtype=np.float64)[order].reshape(77, 67)
    if ec.codes_get(message, "numberOfMissing") or not np.isfinite(values).all():
        raise ValueError("Missing seasonal precipitation cells")
    rates = values * 86_400_000  # m/s to mm/day
    if rates.min() < -0.01 or rates.max() > 1000:
        raise ValueError("Unphysical CDS seasonal precipitation rate")
    return y, x, np.maximum(rates, 0)


def _save(path: Path, metadata: dict, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".partial").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".partial").replace(path)
    atomic_json(path.with_suffix(".json"), metadata | {"sha256": sha256(path)})


def ingest(raw: Path, task: dict, output: Path) -> None:
    import eccodes as ec

    groups = {}
    with raw.open("rb") as stream:
        while (message := ec.codes_grib_new_from_file(stream)) is not None:
            try:
                header = {key: ec.codes_get(message, key) for key in (
                    "dataDate", "forecastMonth", "verifyingMonth", "centre",
                    "systemNumber", "shortName", "units", "number",
                )}
                stamp = str(header["dataDate"])
                init = f"{stamp[:4]}-{stamp[4:6]}"
                target = next_month(init)
                if (
                    init not in task["initializations"]
                    or stamp[6:] != "01"
                    or header["forecastMonth"] != 2
                    or header["verifyingMonth"] != int(target.replace("-", ""))
                    or header["centre"] != CENTRES[task["centre"]]
                    or str(header["systemNumber"]) != task["system"]
                    or header["shortName"] not in ("tprate", "tp")
                    or header["units"] not in ("m s**-1", "m s-1")
                ):
                    raise ValueError(f"Unexpected CDS seasonal GRIB header: {header}")
                y, x, rates = _grid(ec, message)
                group = groups.setdefault(init, {"latitude": y, "longitude": x, "members": {}})
                member = int(header["number"])
                if member in group["members"] or not np.array_equal(y, group["latitude"]) or not np.array_equal(x, group["longitude"]):
                    raise ValueError("Duplicated member or changed CDS grid")
                group["members"][member] = rates
            finally:
                ec.codes_release(message)
    if set(groups) != set(task["initializations"]):
        raise ValueError("CDS response omitted a requested initialization")
    raw_hash = sha256(raw)
    for init, group in groups.items():
        kind, count = vintage(task["centre"], task["system"], init)
        ids = sorted(group["members"])
        if kind != task["kind"] or len(ids) != count or ids not in (list(range(count)), list(range(1, count + 1))):
            raise ValueError(f"Incomplete CDS ensemble: {task['centre']} {task['system']} {init}")
        values = np.stack([group["members"][member] for member in ids])
        target = next_month(init)
        meta = {
            "centre": task["centre"], "system": task["system"],
            "initialization": init, "target": target, "leadtime_month": 2,
            "kind": kind, "members": count, "units": "mm/day",
            "dataset": DATASET, "request": task["request"], "raw_sha256": raw_hash,
            "publication_evidence": None,
            "availability_note": "GRIB initialization is not evidence of historical publication",
        }
        arrays = {"latitude": group["latitude"], "longitude": group["longitude"]}
        if task["centre"] == "meteo_france":
            arrays.update(mean=values.mean(axis=0, dtype=np.float64).astype(np.float32), spread=values.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32))
        else:
            arrays["members"] = values.astype(np.float32)
        if init in read(Path(__file__).parent / "config/acquisition_inventory.json")["calibration_initializations"].get(task["centre"], {}).get(task["system"], []):
            _save(output / "audit/calibration" / task["centre"] / task["system"] / f"{init}.npz", meta, **arrays)
        if route(task["centre"], init) == task["system"]:
            _save(output / "raw/seasonal" / task["centre"] / f"{target}.npz", meta, **arrays)


def _acquire_task(task: dict, output: Path, cache: Path) -> None:
    """A task owns disjoint cache and output paths, including calibration files."""
    import cdsapi

    key = sha256_request(task)
    raw = cache / "cds/seasonal" / task["centre"] / task["system"] / f"{key}.grib"
    record = raw.with_suffix(raw.suffix + ".json")
    raw.parent.mkdir(parents=True, exist_ok=True)
    if record.exists():
        saved = read(record)
        if saved["task"] != task or sha256(raw) != saved["sha256"]:
            raise ValueError(f"Changed CDS response: {raw}")
    else:
        log(f"CDS seasonal {task['centre']} system {task['system']} {task['initializations']}")
        temporary = raw.with_suffix(".partial")
        temporary.unlink(missing_ok=True)
        client = cdsapi.Client(quiet=True, debug=False, timeout=120, retry_max=3)
        client.retrieve(DATASET, task["request"], str(temporary))
        if temporary.stat().st_size == 0:
            raise ValueError("Empty CDS seasonal response")
        temporary.replace(raw)
        atomic_json(record, {"task": task, "dataset": DATASET, "sha256": sha256(raw)})
    products = [output / "raw/seasonal" / task["centre"] / f"{next_month(d)}.npz" for d in task["initializations"] if route(task["centre"], d) == task["system"]]
    calibration = read(Path(__file__).parent / "config/acquisition_inventory.json")["calibration_initializations"].get(task["centre"], {}).get(task["system"], [])
    products += [output / "audit/calibration" / task["centre"] / task["system"] / f"{d}.npz" for d in task["initializations"] if d in calibration]
    raw_digest = sha256(raw)
    if not all(
        p.exists() and p.with_suffix(".json").exists() and checked(p)["raw_sha256"] == raw_digest
        for p in products
    ):
        ingest(raw, task, output)
    log(
        f"CDS seasonal complete {task['centre']} system {task['system']} "
        f"{task['initializations'][0]}..{task['initializations'][-1]}"
    )


def acquire(output: Path, cache: Path, limit_tasks: int | None = None, start_task: int = 0, workers: int = 2) -> tuple[int, int]:
    """Resumable source download; retain GRIBs separately for byte-level audit."""
    from concurrent.futures import ThreadPoolExecutor

    output, cache = Path(output), Path(cache)
    if not Path.home().joinpath(".cdsapirc").is_file():
        raise FileNotFoundError("CDS credentials missing at ~/.cdsapirc")
    plan = tasks()
    selected = plan[start_task:limit_tasks]
    if workers < 1 or workers > 2:
        raise ValueError("CDS workers must be between one and two")
    if workers == 1:
        for task in selected:
            _acquire_task(task, output, cache)
    else:
        def run_one(task):
            try:
                _acquire_task(task, output, cache)
            except Exception as error:
                log(
                    f"CDS seasonal failed {task['centre']} system {task['system']} "
                    f"{task['initializations'][0]}: {error}"
                )
                raise

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_one, selected))
    return len(selected), len(plan)


def sha256_request(task: dict) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:20]


def aggregates(output: Path) -> None:
    """Derive the historical U-Net archive and test-month SEAS5 stats from members."""
    folder = Path(output) / "raw/seasonal/ecmwf"
    means, spreads = [], []
    for target in dates("1994-01", "2023-01"):
        path = folder / f"{target}.npz"
        checked(path)
        with np.load(path, allow_pickle=False) as z:
            values = z["members"]
            means.append(values.mean(axis=0, dtype=np.float64).astype(np.float32))
            spreads.append(values.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32))
            latitude, longitude = z["latitude"], z["longitude"]
    path = Path(output) / "audit/seasonal.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=np.stack(means), spread=np.stack(spreads), latitude=latitude, longitude=longitude)
    for target in dates("2023-01", "2025-01"):
        source = folder / f"{target}.npz"
        meta = checked(source)
        with np.load(source, allow_pickle=False) as z:
            values = z["members"]
            arrays = {"mean": values.mean(axis=0, dtype=np.float64).astype(np.float32), "spread": values.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32), "latitude": z["latitude"], "longitude": z["longitude"]}
        _save(Path(output) / "forecast/seasonal_precipitation" / f"{target}.npz", meta | {"derived_from": str(source.relative_to(output)), "spread_ddof": 1}, **arrays)
