"""Seasonal members and GEFS coverage for the chosen forecast recipe."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .io import bounds, checked


def read_source(root, group, target):
    if group != "gefs":
        raise ValueError("Only the GEFS correction is part of this recipe")
    path = root / group / f"{target}.npz"
    record = checked(path)
    if record["target"] != target or record["available_by"] >= target + "-01":
        raise ValueError("GEFS predictor crossed its row origin")
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def route(centre, initialization):
    if centre == "ecmwf":
        return "51"
    if centre == "dwd":
        return "2" if initialization <= "2020-10" else "21"
    if centre == "meteo_france":
        return (
            "6"
            if initialization <= "2019-09"
            else ("7" if initialization <= "2021-06" else "8")
        )
    raise ValueError("Unregistered seasonal centre")


def interpolate(values, latitude, longitude, lat, lon):
    """Member axis stays separate; interpolation changes only spatial resolution."""
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    array = np.moveaxis(values, (-2, -1), (0, 1))
    out = RegularGridInterpolator(
        (latitude, longitude),
        array,
        bounds_error=True,
    )(np.column_stack([yy.ravel(), xx.ravel()]))
    out = out.reshape(len(lat), len(lon), *values.shape[:-2])
    return np.moveaxis(out, (0, 1), (-2, -1)).astype(np.float32)


def extra_columns(target, source, latitude, longitude, baseline):
    """Only predictors from the row's own origin; no target or neighbouring rows."""
    yy, xx = np.meshgrid(latitude, longitude, indexing="ij")
    points = np.column_stack([yy.ravel(), xx.ravel()])
    grid = RegularGridInterpolator(
        (source["latitude"], source["longitude"]),
        source["rates"].transpose(1, 2, 0),
        bounds_error=True,
    )(points)
    start, end = bounds(target)
    lengths = np.array([7, 7, 7, (end - start).days - 21], dtype=np.float32)
    coverage = source["duration"] / lengths
    if np.any(coverage < 0) or np.any(coverage > 1):
        raise ValueError("Invalid forecast coverage")
    filled = (grid * coverage + baseline[:, None] * (1 - coverage)).astype(np.float32)
    return np.column_stack(
        [filled, np.broadcast_to(coverage, filled.shape), filled - baseline[:, None]]
    )
