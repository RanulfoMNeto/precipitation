"""Checks on the actual prepared package, enabled only with explicit paths."""

import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
def test_chronological_oof_and_no_cached_test_contexts():
    data = os.environ.get("WORCAP_FORECAST_DATA")
    work = os.environ.get("WORCAP_FORECAST_WORK")
    if not data or not work:
        pytest.skip("Set WORCAP_FORECAST_DATA and WORCAP_FORECAST_WORK to the local artifact directories")
    data, work = Path(data), Path(work)
    prepared = json.loads((work / "prepared.json").read_text())
    assert prepared["months"] == 192
    assert len(prepared["context_files"]) == 192
    assert prepared["historical_tabular_oof"]["2015-2020"]["rmse"] > 0
    assert not list((work / "context").glob("202[34]-*.npy"))
    assert not list((work / "ancestors/2023").glob("202[34]-*.npy"))
    assert not (work / "ancestors/2023/unet.npy").exists()
    for year in range(2007, 2024, 2):
        receipt = json.loads(
            (work / "ancestors" / str(year) / "result.json").read_text()
        )
        assert receipt["training_end"] == f"{year - 1}-12"
        assert max(receipt["targets"])[:4] == str(year + 1)
    labels = np.load(work / "ancestors/tabular_samples/dates.npy")
    assert labels.min() == np.datetime64("1999-01", "M")
    assert labels.max() == np.datetime64("2022-12", "M")
    observed = np.load(data / "raw/observations/precipitation.npy", mmap_mode="r")
    assert len(observed) == 996
