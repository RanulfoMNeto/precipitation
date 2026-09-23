"""Read only the observations and forecasts used by forecast."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import checked
from .preprocessing import ATMOSPHERE
from .weather import interpolate, route

FIRST = (1994 - 1940) * 12
END = (2023 - 1940) * 12
MEMBERS = ("c00", "p01", "p02", "p03", "p04")


def dates(first="1994-01", last="2025-01"):
    return np.arange(first, last, dtype="datetime64[M]").astype(str)


def index(month):
    return int((np.datetime64(month, "M") - np.datetime64("1940-01", "M")).astype(int))


def origin(target, meta, centre=None, require_publication=False):
    previous = str(np.datetime64(target, "M") - np.timedelta64(1, "M"))
    if meta["target"] != target or meta["initialization"][:7] != previous:
        raise ValueError(f"Forecast initialisation crosses origin: {target}")
    if meta.get("available_by") and meta["available_by"] >= target + "-01":
        raise ValueError(f"Forecast publication crosses origin: {target}")
    if require_publication and not meta.get("publication_evidence"):
        raise ValueError(f"No publication evidence: {target}")
    if centre and (
        meta["centre"] != centre
        or str(meta["system"]) != route(centre, previous)
        or meta["leadtime_month"] != 2
        or meta["units"] != "mm/day"
    ):
        raise ValueError(f"Seasonal system or units mismatch: {target}")


class RawData:
    def __init__(self, root, stride=4):
        self.root, self.stride = Path(root), stride
        with np.load(self.root / "coordinates.npz", allow_pickle=False) as z:
            self.lat, self.lon = z["latitude"], z["longitude"]
        self.shape = len(self.lat), len(self.lon)
        self.atmosphere = [self.map(n) for n in ATMOSPHERE]
        self.precipitation = self.map("precipitation")
        if len(self.precipitation) != END or any(
            len(a) != END for a in self.atmosphere
        ):
            raise ValueError(
                "Observed precipitation and atmosphere must end December 2022"
            )
        self.test = np.load(self.root / "test_atmosphere.npy", mmap_mode="r")
        if self.test.shape != (9, 35, *self.shape):
            raise ValueError("Test atmosphere must span January 2022–November 2024")
        self.cache = {}

    def map(self, name):
        a = np.load(self.root / "observations" / f"{name}.npy", mmap_mode="r")
        return a if a.ndim == 3 else a.T.reshape(-1, *self.shape)

    def atmospheric_month(self, month):
        if not 0 <= month < index("2024-12"):
            raise ValueError("Atmospheric month unavailable")
        if month < END:
            return np.stack([a[month] for a in self.atmosphere])
        # Each target uses only months strictly before itself; no shifted test row.
        return np.asarray(self.test[:, month - index("2022-01")])

    def truth(self, target, cutoff):
        t = index(target)
        if t >= min(index(cutoff), END):
            raise ValueError("Future or test precipitation is forbidden")
        return np.array(self.precipitation[t], dtype=np.float32, copy=True)

    def weather(self, target):
        if target in self.cache:
            return self.cache[target]
        result = {"members": {}}
        for centre in ("ecmwf", "dwd"):
            p = self.root / "seasonal" / centre / f"{target}.npz"
            meta = checked(p)
            origin(target, meta, centre)
            with np.load(p, allow_pickle=False) as z:
                if len(z["members"]) != meta["members"]:
                    raise ValueError("Incomplete seasonal ensemble")
                result["members"][centre] = interpolate(
                    z["members"],
                    z["latitude"],
                    z["longitude"],
                    self.lat[:: self.stride],
                    self.lon[:: self.stride],
                )
        trajectories = []
        for member in MEMBERS:
            p = self.root / "gefs" / f"{target}_{member}.npz"
            meta = checked(p)
            origin(target, meta)
            if meta["member"] != member or meta["units"] != "mm/day":
                raise ValueError("GEFS member or units mismatch")
            if target >= "2023-01" and not meta.get("test_publication_certified"):
                raise ValueError("Operational GEFS publication evidence missing")
            with np.load(p, allow_pickle=False) as z:
                if not np.array_equal(z["duration"], [7, 7, 1, 0]):
                    raise ValueError("GEFS coverage mismatch")
                trajectories.append(
                    interpolate(
                        z["rates"],
                        z["latitude"],
                        z["longitude"],
                        self.lat[:: self.stride],
                        self.lon[:: self.stride],
                    )
                )
        result["members"]["gefs"] = np.stack(trajectories)
        result["coverage"] = np.array([1, 1, 1 / 7, 0], dtype=np.float32)
        self.cache[target] = result
        return result
