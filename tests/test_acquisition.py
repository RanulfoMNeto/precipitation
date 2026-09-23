"""Guard the independent input inventory and forecast-origin boundaries."""

import numpy as np
import pytest

from worcap_forecast.acquire import _expected, finalize
from worcap_forecast.acquire_atmosphere import request, year_blocks
from worcap_forecast.acquire_gefs import _interval
from worcap_forecast.acquire_seasonal import inventory, next_month, tasks
from worcap_forecast.raw import dates
from worcap_forecast.weather import route


def test_cds_plan_covers_only_registered_initializations():
    planned = {}
    for task in tasks():
        key = task["centre"], task["system"]
        selected = planned.setdefault(key, set())
        assert not selected.intersection(task["initializations"])
        selected.update(task["initializations"])
        assert {d[:4] for d in task["initializations"]} == set(task["request"]["year"])
        assert {d[5:] for d in task["initializations"]} == set(task["request"]["month"])
        assert task["request"]["leadtime_month"] == ["2"]
    assert planned == inventory()
    for target in dates():
        for centre in ("ecmwf", "dwd", "meteo_france"):
            init = str(np.datetime64(target, "M") - np.timedelta64(1, "M"))
            assert init in planned[centre, route(centre, init)]
            assert next_month(init) == target


def test_independent_package_is_complete_or_has_no_manifest(tmp_path):
    expected = _expected(tmp_path)
    assert len(expected) == len(set(expected)) == 8812
    assert not any("2023-01.npy" in str(path) and "observations" in str(path) for path in expected)
    with pytest.raises(ValueError, match="Independent acquisition incomplete"):
        finalize(tmp_path, tmp_path / "source-cache")
    assert not (tmp_path / "manifest.json").exists()


def test_external_requests_stay_on_previous_month():
    for surface in (True, False):
        choice = request("2022", ["12"], surface)
        assert choice["leadtime_month"] == ["2"]
        assert choice["month"] == ["12"]
        assert ("pressure_level" in choice) == (not surface)
    from datetime import date

    origin = date(2022, 12, 31)
    assert _interval("2023-01", origin, 24, 30) == 0
    assert _interval("2023-01", origin, 768, 774) is None


def test_atmospheric_cds_batches_share_one_ensemble_vintage():
    years = [str(year) for year in range(1993, 2025)]
    blocks = year_blocks(years)
    assert sorted(year for block in blocks for year in block) == years
    assert all(len(block) <= 4 for block in blocks)
    assert all(not (block[0] <= "2016" < block[-1]) for block in blocks)
    assert blocks[0] == ["1993"] and blocks[-1] == ["2024"]
    assert request(["2007", "2008"], ["01", "12"], True)["year"] == ["2007", "2008"]
