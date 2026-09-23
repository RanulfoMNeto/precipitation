"""CDS SEAS5 atmospheric member means and spreads for forecast only."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import checked, read
from .raw import dates
from .runtime import atomic_json, log, sha256

AREA = [16, -91, -61, -24]
SURFACE = ("total_cloud_cover", "total_column_water_vapour")
PRESSURE = ("specific_humidity", "u_component_of_wind", "v_component_of_wind")
NAMES = ("cloud", "water", "q850", "u850", "v850")
UNITS = ("1", "kg/m2", "kg/kg", "m/s", "m/s")


def previous(target):
    return str(np.datetime64(target, "M") - np.timedelta64(1, "M"))


def request(year: str | list[str], months: list[str], surface: bool):
    result = {
        "originating_centre": "ecmwf", "system": "51",
        "variable": list(SURFACE if surface else PRESSURE),
        "product_type": ["monthly_mean"], "year": [year] if isinstance(year, str) else year, "month": months,
        "leadtime_month": ["2"], "data_format": "grib", "area": AREA,
    }
    if not surface:
        result["pressure_level"] = ["850"]
    return result


def year_blocks(years: list[str]) -> list[list[str]]:
    """Batch only full years of the same SEAS5 ensemble vintage."""
    covered = set(years)
    blocks = [[year] for year in years if year <= "2006" or year == "2024"]
    for start, end in ((2007, 2016), (2017, 2023)):
        matching = [str(year) for year in range(start, end + 1) if str(year) in covered]
        blocks += [matching[i : i + 4] for i in range(0, len(matching), 4)]
    if sorted(year for block in blocks for year in block) != sorted(years):
        raise ValueError("Unregistered SEAS5 atmospheric year")
    return sorted(blocks, key=lambda block: block[0])


def _reduce(ds, target: str, surface: bool):
    if "forecastMonth" not in ds.coords or not np.array_equal(np.asarray(ds.forecastMonth).reshape(-1), [2]):
        raise ValueError("SEAS5 atmospheric lead time differs")
    instant = np.datetime64(previous(target) + "-01", "ns")
    if instant not in np.asarray(ds.time).reshape(-1).astype("datetime64[ns]"):
        raise ValueError(f"SEAS5 atmospheric initialization missing: {target}")
    ds = ds.sel(time=instant) if "time" in ds.dims else ds
    if np.asarray(ds.time).astype("datetime64[ns]") != instant:
        raise ValueError("SEAS5 atmospheric month differs")
    if not surface and ("isobaricInhPa" not in ds.coords or float(ds.isobaricInhPa) != 850):
        raise ValueError("SEAS5 atmospheric pressure level differs")
    short = ("tcc", "tcwv") if surface else ("q", "u", "v")
    means, spreads = [], []
    for name in short:
        if name not in ds:
            raise ValueError(f"Missing SEAS5 atmospheric variable: {name}")
        field = ds[name]
        for dimension in tuple(field.dims):
            if dimension not in {"number", "latitude", "longitude"}:
                if field.sizes[dimension] != 1:
                    raise ValueError("Unexpected SEAS5 atmospheric dimension")
                field = field.isel({dimension: 0}, drop=True)
        unit = field.attrs.get("units", "").replace(" ", "").replace("**", "^")
        valid_units = {
            "tcc": {"(0-1)", "1", "0-1", "Proportion"},
            "tcwv": {"kgm^-2", "kgm-2", "kg/m^2"},
            "q": {"kgkg^-1", "kgkg-1", "kg/kg"},
            "u": {"ms^-1", "ms-1", "m/s"},
            "v": {"ms^-1", "ms-1", "m/s"},
        }
        if unit not in valid_units[name] or int(field.attrs.get("GRIB_systemNumber", -1)) != 51:
            raise ValueError(f"Unexpected SEAS5 atmospheric unit/system: {name}, {unit}")
        members = np.asarray(field.number).astype(int)
        expected = 25 if previous(target) <= "2016-12" else 51
        if len(members) != expected or not np.array_equal(np.sort(members), np.arange(expected)):
            raise ValueError("Incomplete SEAS5 atmospheric ensemble")
        values = np.asarray(field.transpose("number", "latitude", "longitude"), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Missing SEAS5 atmospheric cells")
        means.append(values.mean(0))
        spreads.append(values.std(0, ddof=1))
    latitude = np.asarray(ds.latitude, np.float64)
    longitude = (np.asarray(ds.longitude, np.float64) + 180) % 360 - 180
    iy, ix = np.argsort(latitude), np.argsort(longitude)
    if len(latitude) != 77 or len(longitude) != 67 or not np.all(np.diff(latitude[iy]) > 0) or not np.all(np.diff(longitude[ix]) > 0):
        raise ValueError("Unexpected SEAS5 atmospheric grid")
    return {
        "mean": np.stack(means)[:, iy][:, :, ix].astype(np.float32),
        "spread": np.stack(spreads)[:, iy][:, :, ix].astype(np.float32),
        "latitude": latitude[iy], "longitude": longitude[ix],
    }, expected


def acquire(output: Path, cache: Path, limit_years: int | None = None, start_year: str | None = None) -> tuple[int, int]:
    import cdsapi
    import xarray as xr

    output, cache = Path(output), Path(cache)
    starts = [previous(target) for target in dates()]
    years = sorted({start[:4] for start in starts})
    if start_year is not None:
        years = [year for year in years if year >= start_year]
    if not Path.home().joinpath(".cdsapirc").is_file():
        raise FileNotFoundError("CDS credentials missing at ~/.cdsapirc")
    client = cdsapi.Client(quiet=True, debug=False, timeout=120, retry_max=3)
    for block in year_blocks(years[:limit_years]):
        targets = [target for target in dates() if previous(target)[:4] in block]
        paths = [output / "raw/seasonal_atmosphere" / f"{target}.npz" for target in targets]
        if all(
            path.is_file()
            and path.with_suffix(".json").is_file()
            and checked(path)["initialization"] == previous(target)
            for path, target in zip(paths, targets, strict=True)
        ):
            continue
        raw = []
        datasets = []
        try:
            for surface in (True, False):
                dataset = "seasonal-monthly-single-levels" if surface else "seasonal-monthly-pressure-levels"
                selection = request(block, sorted({previous(t)[5:] for t in targets}), surface)
                identifier = block[0] if len(block) == 1 else f"{block[0]}-{block[-1]}"
                path = cache / "cds/atmosphere" / f"{identifier}-{'surface' if surface else 'pressure'}.grib"
                path.parent.mkdir(parents=True, exist_ok=True)
                marker = path.with_suffix(path.suffix + ".json")
                if marker.exists():
                    record = read(marker)
                    if record["dataset"] != dataset or record["request"] != selection or sha256(path) != record["sha256"]:
                        raise ValueError(f"Changed CDS atmospheric response: {path}")
                else:
                    log(f"CDS atmosphere {dataset} {identifier}")
                    temporary = path.with_suffix(".partial")
                    temporary.unlink(missing_ok=True)
                    client.retrieve(dataset, selection, str(temporary))
                    if not temporary.is_file() or not temporary.stat().st_size:
                        raise ValueError("Empty CDS atmospheric response")
                    temporary.replace(path)
                    atomic_json(marker, {"dataset": dataset, "request": selection, "sha256": sha256(path)})
                raw.append(path)
                datasets.append(xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": "", "time_dims": ("forecastMonth", "time"), "read_keys": ["systemNumber", "numberOfForecastsInEnsemble"]}))
            for target, path in zip(targets, paths, strict=True):
                left, lm = _reduce(datasets[0], target, True)
                right, rm = _reduce(datasets[1], target, False)
                if lm != rm or not np.array_equal(left["latitude"], right["latitude"]) or not np.array_equal(left["longitude"], right["longitude"]):
                    raise ValueError("SEAS5 atmospheric grids or ensembles disagree")
                arrays = left | {"mean": np.concatenate([left["mean"], right["mean"]]), "spread": np.concatenate([left["spread"], right["spread"]])}
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix(".partial").open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                path.with_suffix(".partial").replace(path)
                atomic_json(path.with_suffix(".json"), {
                    "source": "seas5", "target": target, "initialization": previous(target),
                    "system": "51", "leadtime_month": 2, "members": lm,
                    "field_names": list(NAMES), "units": list(UNITS),
                    "datasets": ["seasonal-monthly-single-levels", "seasonal-monthly-pressure-levels"],
                    "requests": [read(p.with_suffix(p.suffix + ".json"))["request"] for p in raw],
                    "raw_sha256": [sha256(p) for p in raw], "sha256": sha256(path),
                    "publication_evidence": None,
                    "availability_note": "Historical CDS hindcast publication remains unverified",
                })
        finally:
            for ds in datasets:
                ds.close()
        log(f"CDS atmosphere complete {block[0]}..{block[-1]}")
    return len(years[:limit_years]), len(years)
