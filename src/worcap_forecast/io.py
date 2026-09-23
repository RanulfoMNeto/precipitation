"""Hashed inputs, calendar boundaries and the official submission contract."""

from __future__ import annotations

import csv
import json
from datetime import date
from itertools import zip_longest
from pathlib import Path

import numpy as np

from .runtime import sha256


def read(path):
    return json.loads(Path(path).read_text())


def bounds(target):
    start = date.fromisoformat(target + "-01")
    end = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
    return start, end


def checked(path):
    path = Path(path)
    metadata = read(path.with_suffix(".json"))
    if sha256(path) != metadata["sha256"]:
        raise ValueError(f"Changed input: {path}")
    return metadata


def write_submission(
    sample: Path,
    output: Path,
    predictions: np.ndarray,
    dates: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> int:
    expected_shape = (len(dates), len(lat), len(lon))
    if predictions.shape != expected_shape or not np.isfinite(predictions).all():
        raise ValueError("Grade de inferência inválida")
    if np.any(predictions < 0):
        raise ValueError("Precipitação negativa na submissão")
    date_map = {str(date): i for i, date in enumerate(dates.astype("datetime64[M]"))}
    lat_map = {float(value): i for i, value in enumerate(lat)}
    lon_map = {float(value): i for i, value in enumerate(lon)}
    seen = np.zeros(expected_shape, dtype=bool)
    temporary = output.with_suffix(".csv.tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with sample.open(newline="") as source, temporary.open("w", newline="") as target:
        reader, writer = csv.DictReader(source), csv.writer(target)
        if reader.fieldnames != ["id", "tp_mm_day"]:
            raise ValueError("Cabeçalho do sample_submission inválido")
        writer.writerow(reader.fieldnames)
        for row in reader:
            year, month, latitude, longitude = row["id"].split("_")
            location = (
                date_map[f"{year}-{month}"],
                lat_map[float(latitude)],
                lon_map[float(longitude)],
            )
            if seen[location]:
                raise ValueError(f"ID duplicado no sample_submission: {row['id']}")
            seen[location] = True
            writer.writerow([row["id"], f"{float(predictions[location]):.7f}"])
            rows += 1
    if not seen.all():
        raise ValueError("Sample submission não cobre toda a grade e todos os meses")
    temporary.replace(output)
    validate_submission(sample, output, rows)
    return rows


def validate_submission(sample: Path, submission: Path, expected_rows: int) -> None:
    with sample.open(newline="") as first, submission.open(newline="") as second:
        reference, candidate = csv.DictReader(first), csv.DictReader(second)
        if candidate.fieldnames != ["id", "tp_mm_day"]:
            raise ValueError("Cabeçalho da submissão inválido")
        rows = 0
        for expected, actual in zip_longest(reference, candidate):
            if expected is None or actual is None or actual["id"] != expected["id"]:
                raise ValueError(
                    "IDs ou ordem da submissão diferem do sample_submission"
                )
            value = float(actual["tp_mm_day"])
            if not np.isfinite(value) or value < 0:
                raise ValueError("Submissão contém valor inválido")
            rows += 1
        if rows != expected_rows:
            raise ValueError(
                f"Número de linhas incorreto: {rows}, esperado {expected_rows}"
            )
