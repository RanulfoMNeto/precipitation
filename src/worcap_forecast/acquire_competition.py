"""Convert the thirteen official competition files without reading test labels."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .data import OBSERVATIONS
from .io import read
from .runtime import atomic_json, log, sha256

FIELDS = {
    "air_temperature_2m": "t2",
    "cloud_cover": "cloud_cover",
    "surface_pressure": "surface_pressure",
    "specific_humidity_850hpa": "shum_850",
    "relative_humidity_850hpa": "rel_hum_850",
    "air_temperature_850hpa": "temperature_850",
    "geopotential_850hpa": "geopotential_850",
    "eastward_wind_850hpa": "u_850",
    "northward_wind_850hpa": "v_850",
}


def convert(originals: Path, output: Path) -> None:
    """Import official Kaggle files and verify recorded sizes and SHA-256 hashes."""
    import xarray as xr

    originals, output = Path(originals), Path(output)
    expected = read(Path(__file__).parent / "config/competition_expected.json")
    receipt = {}
    for name, spec in expected["files"].items():
        path = originals / name
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            raise ValueError(f"Missing or changed official original: {name}")
        digest = sha256(path)
        if digest != spec["sha256"]:
            raise ValueError(f"Official original hash mismatch: {name}")
        receipt[name] = {"bytes": spec["bytes"], "sha256": digest}
    atomic_json(
        output / "competition_originals.json",
        {
            "files": receipt,
            "verified_against_converted_arrays": True,
            "retrieval": "Local import of official Kaggle files verified by size and SHA-256",
        },
    )
    with xr.open_dataset(originals / "treino_tp.nc") as ds:
        lat = ds.lat.values
        lon = ds.lon.values
        train_time = ds.time.values.astype("datetime64[M]")
        if (
            not np.array_equal(train_time, np.arange("1940-01", "2023-01", dtype="datetime64[M]"))
            or ds.tp.attrs.get("units") != "mm/day"
            or len(lat) != 301
            or len(lon) != 261
            or not np.all(np.diff(lat) > 0)
            or not np.all(np.diff(lon) > 0)
        ):
            raise ValueError("Unexpected training precipitation grid, dates or units")
        rain = output / "raw/observations/precipitation.npy"
        rain.parent.mkdir(parents=True, exist_ok=True)
        array = np.lib.format.open_memmap(rain, mode="w+", dtype="float32", shape=(996, 301, 261))
        for begin in range(0, 996, 12):
            values = ds.tp.isel(time=slice(begin, begin + 12)).values.astype("float32")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid observed precipitation")
            array[begin : begin + len(values)] = values
        array.flush()
        del array
    with xr.open_dataset(originals / "treino_tp_alvo.nc") as ds:
        if not np.array_equal(ds.time.values.astype("datetime64[M]"), train_time):
            raise ValueError("Target training dates changed")
        saved = np.load(rain, mmap_mode="r")
        for begin in range(0, 995, 12):
            end = min(begin + 12, 995)
            if not np.allclose(ds.tp_alvo.isel(time=slice(begin, end)).values, saved[begin + 1 : end + 1], atol=1e-6, rtol=1e-6):
                raise ValueError("Training target differs from the next precipitation month")
        if not np.isnan(ds.tp_alvo.isel(time=-1).values).all():
            raise ValueError("Last training target must be absent")
    with xr.open_dataset(originals / "teste_features.nc") as ds:
        targets = np.arange("2023-01", "2025-01", dtype="datetime64[M]")
        origins = targets - np.timedelta64(1, "M")
        if (
            not np.array_equal(ds.time.values.astype("datetime64[M]"), targets)
            or not np.array_equal(ds.time_origem.values.astype("datetime64[M]"), origins)
            or not np.array_equal(ds.lat.values, lat)
            or not np.array_equal(ds.lon.values, lon)
            or not np.array_equal(ds.lag_meses.values, np.arange(1, 25))
            or not np.isnan(ds.tp_alvo.values).all()
        ):
            raise ValueError("Unexpected test chronology, grid or targets")
        if not np.allclose(ds.tp_ultima_obs.values, np.load(rain, mmap_mode="r")[-1], atol=1e-6, rtol=1e-6):
            raise ValueError("Test precipitation is not frozen at December 2022")
        test = np.lib.format.open_memmap(output / "raw/test_atmosphere.npy", mode="w+", dtype="float32", shape=(9, 35, 301, 261))
        for i, (name, field) in enumerate(FIELDS.items()):
            path = output / "raw/observations" / f"{name}.npy"
            with xr.open_dataset(originals / f"treino_{field}.nc") as train:
                if (
                    not np.array_equal(train.time.values.astype("datetime64[M]"), train_time)
                    or not np.array_equal(train.lat.values, lat)
                    or not np.array_equal(train.lon.values, lon)
                ):
                    raise ValueError(f"Training atmosphere chronology or grid differs: {field}")
                observed = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(996, 301, 261))
                for begin in range(0, 996, 12):
                    values = train[field].isel(time=slice(begin, begin + 12)).values.astype("float32")
                    if not np.isfinite(values).all():
                        raise ValueError(f"Missing historical atmosphere: {field}")
                    observed[begin : begin + len(values)] = values
                observed.flush()
                for j in range(11):
                    test[i, j] = observed[984 + j]
                for j in range(24):
                    values = ds[field].isel(time=j).values.astype("float32")
                    if not np.isfinite(values).all():
                        raise ValueError(f"Missing test-origin atmosphere: {field}")
                    if j == 0 and not np.allclose(values, observed[-1], atol=1e-5, rtol=1e-5):
                        raise ValueError(f"Test origin December 2022 differs: {field}")
                    test[i, j + 11] = values
                del observed
            log(f"Competition atmosphere converted: {field}")
        test.flush()
        del test
    np.savez(output / "raw/coordinates.npz", latitude=lat, longitude=lon, observation_time=train_time.astype("datetime64[ns]"), target_time=targets.astype("datetime64[ns]"), forecast_origin=origins.astype("datetime64[ns]"))
    shutil.copy2(originals / "sample_submission.csv", output / "sample_submission.csv")
    if set(OBSERVATIONS) != {"precipitation", *FIELDS}:
        raise ValueError("Competition variable registry differs")
