"""NOAA GEFS five trajectories and operational mean used by forecast."""

from __future__ import annotations

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from .io import bounds, checked, read
from .raw import MEMBERS, dates
from .runtime import atomic_json, log, sha256

RETRO = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
OPER = "https://noaa-gefs-pds.s3.amazonaws.com"
LATITUDE = np.arange(-61, 17, dtype=np.float32)
LONGITUDE = np.arange(-91, -23, dtype=np.float32)


def _get(url, **kwargs):
    import requests

    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(15, 180), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(min(30, 2**attempt))


def _save(path, metadata, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".partial").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".partial").replace(path)
    atomic_json(path.with_suffix(".json"), metadata | {"sha256": sha256(path)})


def _interval(target, init, start_hour, end_hour):
    start, end = bounds(target)
    offset = (start - init).days * 24
    a, b = start_hour - offset, end_hour - offset
    if b - a != 6:
        raise ValueError("GEFS interval is not six hours")
    if a < 0 or b > (end - start).days * 24:
        return None
    edges = [0, 168, 336, 504, (end - start).days * 24]
    for i, (left, right) in enumerate(pairwise(edges)):
        if left <= a and b <= right:
            return i
    raise ValueError("GEFS accumulation straddles a target week")


def _crop(ec, message):
    ni, nj = ec.codes_get(message, "Ni"), ec.codes_get(message, "Nj")
    if ec.codes_get(message, "jPointsAreConsecutive") or ec.codes_get(message, "alternativeRowScanning"):
        raise ValueError("Unsupported GEFS scanning")
    lat0 = ec.codes_get(message, "latitudeOfFirstGridPointInDegrees")
    lon0 = ec.codes_get(message, "longitudeOfFirstGridPointInDegrees")
    dy = ec.codes_get(message, "jDirectionIncrementInDegrees") * (1 if ec.codes_get(message, "jScansPositively") else -1)
    dx = ec.codes_get(message, "iDirectionIncrementInDegrees") * (-1 if ec.codes_get(message, "iScansNegatively") else 1)
    iy = np.rint((LATITUDE - lat0) / dy).astype(int)
    ix = np.rint(((LONGITUDE % 360 - lon0) % 360) / dx).astype(int)
    if not np.allclose(lat0 + iy * dy, LATITUDE) or not np.allclose((lon0 + ix * dx) % 360, LONGITUDE % 360):
        raise ValueError("GEFS grid differs from one-degree extraction")
    values = ec.codes_get_values(message).reshape(nj, ni)[np.ix_(iy, ix)]
    if not np.isfinite(values).all() or np.any(values < -0.001) or np.any(values > 1000):
        raise ValueError("Invalid GEFS precipitation")
    return np.maximum(values, 0)


def _decode(path, target, init, totals, hours, seen, max_hour=384, member=None):
    import eccodes as ec

    with Path(path).open("rb") as stream:
        while (message := ec.codes_grib_new_from_file(stream)) is not None:
            try:
                if ec.codes_get(message, "shortName") not in ("tp", "unknown") or ec.codes_get(message, "units") not in ("kg m**-2", "kg m-2"):
                    raise ValueError("Expected GEFS total precipitation in kg/m2")
                if str(ec.codes_get(message, "dataDate")) != init.strftime("%Y%m%d") or ec.codes_get(message, "dataTime") != 0:
                    raise ValueError("GEFS initialization differs from origin")
                if member is not None and ec.codes_get(message, "perturbationNumber") != int(member[1:]):
                    raise ValueError("Decoded GEFS member differs")
                a, b = int(ec.codes_get(message, "startStep")), int(ec.codes_get(message, "endStep"))
                if a >= max_hour:
                    continue
                if b > max_hour:
                    raise ValueError("GEFS accumulation crosses forecast horizon")
                if b - a == 3:
                    continue
                week = _interval(target, init, a, b)
                if week is None:
                    continue
                if (a, b) in seen:
                    raise ValueError("GEFS interval counted twice")
                seen.add((a, b))
                totals[week] += _crop(ec, message)
                hours[week] += 6
            finally:
                ec.codes_release(message)


def _retrospective(target, member, cache):
    init = bounds(target)[0] - timedelta(days=1)
    stamp = init.strftime("%Y%m%d00")
    totals = np.zeros((4, 78, 68), np.float64)
    hours, seen, sources = np.zeros(4), set(), []
    for segment in ("1-10", "10-35" if init.weekday() == 2 else "10-16"):
        url = f"{RETRO}/{init.year}/{stamp}/{member}/Days:{segment}/apcp_sfc_{stamp}_{member}.grib2"
        path = cache / "noaa/retrospective" / target / member / f"{segment}.grib2"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            response = _get(url, stream=True)
            with path.with_suffix(".partial").open("wb") as stream:
                for chunk in response.iter_content(2**20):
                    stream.write(chunk)
            path.with_suffix(".partial").replace(path)
        record = {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size}
        marker = path.with_suffix(path.suffix + ".json")
        if marker.exists() and read(marker) != record:
            raise ValueError(f"Changed GEFS reforecast source: {path}")
        atomic_json(marker, record)
        sources.append(record)
        _decode(path, target, init, totals, hours, seen, member=member)
    if not np.array_equal(hours / 24, [7, 7, 1, 0]):
        raise ValueError("Incomplete 15-day GEFS retrospective coverage")
    rates = np.divide(totals, hours[:, None, None] / 24, out=np.zeros_like(totals), where=hours[:, None, None] > 0).astype(np.float32)
    return rates, (hours / 24).astype(np.float32), {
        "target": target, "member": member, "initialization": str(init),
        "units": "mm/day", "kind": "retrospective_v12", "sources": sources,
        "temporal_intervals": sorted(seen), "publication_evidence": None,
        "availability_note": "Reforecast was generated retrospectively; nominal date is not publication date",
    }


def _url(init, member, hour):
    stamp = init.strftime("%Y%m%d")
    if init >= date(2020, 9, 23):
        return f"{OPER}/gefs.{stamp}/00/atmos/pgrb2ap5/ge{member}.t00z.pgrb2a.0p50.f{hour:03d}"
    return f"{OPER}/gefs.{stamp}/00/pgrb2a/ge{member}.t00z.pgrb2af{hour:02d}"


def _step(url, path, hour):
    marker = path.with_suffix(path.suffix + ".json")
    if marker.exists():
        record = read(marker)
        if record["url"] != url or sha256(path) != record["sha256"]:
            raise ValueError(f"Changed GEFS byte-range product: {path}")
        return record
    rows_response = _get(url + ".idx")
    rows = rows_response.text.strip().splitlines()
    matches = [i for i, row in enumerate(rows) if ":APCP:surface:" in row]
    if len(matches) != 1:
        raise ValueError("Ambiguous GEFS precipitation index")
    i = matches[0]
    match = re.search(r":(\d+)-(\d+) hour acc fcst:", rows[i])
    if not match or tuple(map(int, match.groups())) != (hour - 6, hour):
        raise ValueError("GEFS accumulation interval changed")
    start = int(rows[i].split(":")[1])
    end = str(int(rows[i + 1].split(":")[1]) - 1) if i + 1 < len(rows) else ""
    response = _get(url, headers={"Range": f"bytes={start}-{end}"})
    if response.status_code != 206 or not response.content.startswith(b"GRIB"):
        raise ValueError("GEFS response did not honor byte range")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".partial").write_bytes(response.content)
    path.with_suffix(".partial").replace(path)
    record = {"url": url, "last_modified": response.headers.get("Last-Modified"),
              "index_sha256": hashlib.sha256(rows_response.content).hexdigest(),
              "byte_range": [start, end], "sha256": sha256(path)}
    atomic_json(marker, record)
    return record


def _operational(target, member, cache):
    init = bounds(target)[0] - timedelta(days=1)
    totals = np.zeros((4, 78, 68), np.float64)
    hours, seen, sources = np.zeros(4), set(), []
    def step(hour):
        url = _url(init, member, hour)
        path = cache / "noaa/operational" / target / member / f"{hour:03d}.grib2"
        return hour, path, _step(url, path, hour)

    with ThreadPoolExecutor(max_workers=4) as pool:
        steps = pool.map(step, range(30, 385, 6))
        for _, path, record in steps:
            _decode(path, target, init, totals, hours, seen, member=None if member == "avg" else member)
            sources.append(record)
    duration = (hours / 24).astype(np.float32)
    if not np.array_equal(duration, [7, 7, 1, 0]):
        raise ValueError("Incomplete operational GEFS coverage")
    modified = [parsedate_to_datetime(r["last_modified"]) for r in sources if r.get("last_modified")]
    cutoff = datetime.combine(bounds(target)[0], datetime.min.time(), UTC)
    publication = None
    if len(modified) == len(sources) and all(t.tzinfo is not None for t in modified) and max(modified) < cutoff:
        publication = max(modified).isoformat()
    elif target >= "2023-01":
        raise ValueError("GEFS operational object timestamps cross target start")
    rates = np.divide(totals, duration[:, None, None], out=np.zeros_like(totals), where=duration[:, None, None] > 0).astype(np.float32)
    return rates, duration, {
        "target": target, "initialization": str(init), "units": "mm/day",
        "kind": "operational_v12" if init >= date(2020, 9, 23) else "operational_v11",
        "sources": sources, "temporal_intervals": sorted(seen),
        "publication_evidence": {"source": "S3 Last-Modified", "latest": publication} if publication else None,
        "available_by": publication,
        "availability_note": None if publication else "Object publication before target has not been demonstrated",
    }


def acquire(output: Path, cache: Path, limit_months: int | None = None) -> tuple[int, int]:
    output, cache = Path(output), Path(cache)
    targets = list(dates("2007-01", "2025-01"))
    for target in targets[:limit_months]:
        def one(member, target=target):
            path = output / "raw/gefs" / f"{target}_{member}.npz"
            if path.exists():
                checked(path)
                with np.load(path, allow_pickle=False) as z:
                    rates, duration = z["rates"], z["duration"]
            else:
                if target <= "2019-12":
                    rates, duration, meta = _retrospective(target, member, cache)
                else:
                    rates, duration, meta = _operational(target, member, cache)
                meta["member"] = member
                meta["test_publication_certified"] = bool(target >= "2023-01" and meta.get("publication_evidence"))
                _save(path, meta, rates=rates, duration=duration, latitude=LATITUDE, longitude=LONGITUDE)
            return rates, duration

        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(one, MEMBERS))
        member_rates = [rates for rates, _ in results]
        member_durations = [duration for _, duration in results]
        if any(not np.array_equal(member_durations[0], other) for other in member_durations[1:]):
            raise ValueError(f"GEFS members have different target coverage: {target}")
        duration = member_durations[0]
        weather = output / "weather/gefs" / f"{target}.npz"
        if weather.exists():
            checked(weather)
            continue
        if target <= "2019-12" or bounds(target)[0] - timedelta(days=1) < date(2020, 9, 23):
            rates = np.mean(member_rates, axis=0, dtype=np.float64).astype(np.float32)
            meta = {"target": target, "initialization": str(bounds(target)[0] - timedelta(days=1)),
                    "kind": "retrospective_v12" if target <= "2019-12" else "operational_v11",
                    "units": "mm/day", "sources": {member: checked(output / "raw/gefs" / f"{target}_{member}.npz")["sha256"] for member in MEMBERS},
                    "available_by": str(bounds(target)[0] - timedelta(days=1)), "publication_evidence": None}
        else:
            rates, duration, meta = _operational(target, "avg", cache)
            meta["ensemble_statistic"] = "operational_mean"
        _save(weather, meta | {"coverage_days": duration.tolist()}, rates=rates, duration=duration, latitude=LATITUDE, longitude=LONGITUDE)
        log(f"GEFS complete: {target}")
    return len(targets[:limit_months]), len(targets)
