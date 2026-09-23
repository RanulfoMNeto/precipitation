import json

import numpy as np
import pytest

from worcap_forecast.data import availability_report, verify
from worcap_forecast.io import validate_submission, write_submission
from worcap_forecast.pipeline import TEMPORAL_WEIGHT
from worcap_forecast.raw import END, RawData, origin
from worcap_forecast.runtime import sha256


def test_origin_rejects_future_and_wrong_system():
    meta = {
        "target": "2023-01",
        "initialization": "2022-12",
        "centre": "ecmwf",
        "system": "51",
        "leadtime_month": 2,
        "units": "mm/day",
        "available_by": "2023-01-02",
    }
    with pytest.raises(ValueError, match="publication"):
        origin("2023-01", meta, "ecmwf")
    meta["available_by"] = "2022-12-06"
    origin("2023-01", meta, "ecmwf")
    meta["system"] = "5"
    with pytest.raises(ValueError, match="system"):
        origin("2023-01", meta, "ecmwf")


def test_truth_refuses_test_and_future_fold():
    raw = object.__new__(RawData)
    raw.precipitation = np.zeros((END, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="Future"):
        raw.truth("2023-01", "2023-01")
    with pytest.raises(ValueError, match="Future"):
        raw.truth("2015-01", "2015-01")


def test_manifest_tampering_and_unknown_publication(tmp_path):
    (tmp_path / "raw/seasonal/ecmwf").mkdir(parents=True)
    meta = tmp_path / "raw/seasonal/ecmwf/1994-01.json"
    meta.write_text(json.dumps({"target": "1994-01", "initialization": "1993-12"}))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "observed_precipitation_end": "2022-12",
                "files": {"raw/seasonal/ecmwf/1994-01.json": sha256(meta)},
            }
        )
    )
    verify(tmp_path)
    assert not availability_report(tmp_path)["strict_origin_available"]
    meta.write_text("modified")
    with pytest.raises(ValueError, match="Changed"):
        verify(tmp_path)


def test_submission_contract_and_fixed_weight(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("id,tp_mm_day\n2023_01_0.0_1.0,0\n")
    output = tmp_path / "submission.csv"
    temporal, gefs = np.float32(2), np.float32(4)
    value = np.float32(np.float64(gefs) + TEMPORAL_WEIGHT * (np.float64(temporal) - gefs))
    write_submission(
        sample,
        output,
        np.array([[[value]]]),
        np.array(["2023-01"], dtype="datetime64[M]"),
        np.array([0.0]),
        np.array([1.0]),
    )
    validate_submission(sample, output, 1)
    assert float(output.read_text().splitlines()[1].split(",")[1]) == pytest.approx(
        value, abs=1e-6
    )
    output.write_text("id,tp_mm_day\n2023_01_0.0_1.0,-1\n")
    with pytest.raises(ValueError, match="inválido"):
        validate_submission(sample, output, 1)


def test_all_base_features_ignore_next_test_row():
    from types import SimpleNamespace

    from worcap_forecast.features import ResidualExamples, TreeExamples
    from worcap_forecast.preprocessing import Examples
    from worcap_forecast.raw import index

    test = np.zeros((9, 35, 2, 2), np.float32)
    test[0, 12] = 7.0  # January 2023, the origin for February 2023.
    test[0, 13] = 99.0  # February 2023 must not enter its own forecast.
    observed = [np.zeros((END, 2, 2), np.float32) for _ in range(9)]
    data = SimpleNamespace(
        height=2,
        width=2,
        n_locations=4,
        observed_months=END,
        coordinates={
            "latitude": np.array([-2.0, -1.0]),
            "longitude": np.array([1.0, 2.0]),
        },
        atmosphere=observed,
        forecast_atmosphere=[a[11:] for a in test],
    )
    stats = {
        "atmosphere_mean": np.zeros((9, 12), np.float32),
        "atmosphere_std": np.ones((9, 12), np.float32),
        "climatology": np.zeros((12, 2, 2), np.float32),
        "climatology_scale": np.float32(1),
        "seas_climatology": np.zeros((12, 2, 2), np.float32),
        "seas_scale": np.ones((12, 2, 2), np.float32),
        "spread_scale": np.float32(1),
        "training_end": np.int64(index("2023-01") - 1),
    }
    archive = SimpleNamespace(
        fields=lambda *_: (np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))
    )
    target = index("2023-02")
    unet = Examples(data, archive, stats, True)
    tree = TreeExamples(data, archive, stats, True)
    residual = ResidualExamples(data, archive, stats, True)
    before_unet = unet.example(target)["features"]
    before_tree, _ = tree.features(target)
    before_residual = residual.features(target, np.zeros((2, 2), np.float32))
    assert np.array_equal(before_unet[0], np.full((2, 2), 7.0))

    # February's atmosphere is only in March's row: change it adversarially.
    test[:, 13] = -999.0
    np.testing.assert_array_equal(unet.example(target)["features"], before_unet)
    np.testing.assert_array_equal(tree.features(target)[0], before_tree)
    np.testing.assert_array_equal(
        residual.features(target, np.zeros((2, 2), np.float32)), before_residual
    )
