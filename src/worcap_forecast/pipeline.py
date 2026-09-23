"""Five final fits, chronological ancestors and fresh inference for forecast."""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

from .base import unet_model
from .data import verify as verify_data
from .io import read, write_submission
from .oof import Builder
from .preprocessing import Examples
from .raw import RawData, dates, index
from .runtime import (
    atomic_json,
    configure_torch,
    environment_receipt,
    log,
    sha256,
    source_fingerprint,
)
from .temporal import Corrector, encode_members, runtime
from .weather import extra_columns, read_source

TEMPORAL_WEIGHT = 0.6337257586287943
SEEDS = (42, 2026)
COLUMNS = 98
BASELINE = 86


def model_receipt(folder, names, **extra):
    atomic_json(
        folder / "result.json",
        {**extra, "files": {n: sha256(folder / n) for n in names}},
    )


def done(folder):
    p = folder / "result.json"
    if not p.is_file():
        return False
    for name, digest in read(p)["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"Changed model artifact: {folder / name}")
    return True


def fit_tree(folder, x, y, options):
    folder.mkdir(parents=True, exist_ok=True)
    if done(folder):
        return lgb.Booster(model_file=str(folder / "model.txt"))
    params, rounds = options["parameters"], options["rounds"]
    train = lgb.Dataset(x, label=y, params=params, free_raw_data=False)
    train.construct()
    model = None
    for start in range(0, rounds, 100):
        model = lgb.train(
            params,
            train,
            num_boost_round=min(100, rounds - start),
            init_model=str(folder / "model.txt") if start else None,
            keep_training_booster=True,
        )
        model.save_model(str(folder / "model.txt"))
        atomic_json(
            folder / "progress.json",
            {"rounds": min(start + 100, rounds), "total": rounds},
        )
        log(f"{folder.name}: {min(start + 100, rounds)}/{rounds} trees")
    del train
    gc.collect()
    model_receipt(folder, ["model.txt"], rounds=rounds)
    return model


def context(folder, target):
    value = np.load(folder / "context" / f"{target}.npy", mmap_mode="r")
    if value.shape[0] != COLUMNS or not np.isfinite(value).all():
        raise ValueError(f"Invalid context schema: {target}")
    return np.asarray(value)


def members(raw, target, reference, stride):
    old = raw.stride
    raw.stride = stride
    raw.cache.pop(target, None)
    weather = raw.weather(target)
    raw.stride = old
    raw.cache.pop(target, None)
    return encode_members(
        {
            "seas5": weather["members"]["ecmwf"],
            "dwd": weather["members"]["dwd"],
            "gefs": weather["members"]["gefs"],
            "coverage": weather["coverage"],
        },
        reference,
        stride,
    )


def prepare(data, work):
    data, work = Path(data), Path(work)
    verify_data(data)
    work.mkdir(parents=True, exist_ok=True)
    binding = work / "binding.json"
    identity = {
        "data_manifest": sha256(data / "manifest.json"),
        "recipe": "monthly_precipitation_five_models",
        "observation_end": "2022-12",
        "source_files": source_fingerprint(),
    }
    if binding.exists() and read(binding) != identity:
        raise ValueError("Work directory belongs to a different input or mode")
    atomic_json(binding, identity)
    builder = Builder(data, work / "ancestors")
    for year in range(1999, 2024, 2):
        builder.block(year)
    if not (work / "ancestors/tabular_samples/x.npy").is_file():
        builder.samples()
    samples = work / "ancestors/tabular_samples"
    sample_dates = np.load(samples / "dates.npy")
    x = np.load(samples / "x.npy", mmap_mode="r")
    y = np.load(samples / "y.npy", mmap_mode="r")
    config = read(data / "training.json")
    for year in range(2007, 2024, 2):
        fit = work / "models" / f"tabular_{year}"
        take = sample_dates < np.datetime64(f"{year}-01")
        if not take.any() or sample_dates[take].max() >= np.datetime64(f"{year}-01"):
            raise ValueError("Tabular training crossed its OOF fold")
        booster = fit_tree(fit, np.asarray(x[take]), np.asarray(y[take]), config["tabular"])
        target_folder = work / "context"
        target_folder.mkdir(exist_ok=True)
        if year == 2023:
            del booster
            continue
        for target in dates(f"{year}-01", f"{year + 2}-01"):
            output = target_folder / f"{target}.npy"
            if output.exists():
                continue
            base = np.load(work / "ancestors" / str(year) / f"{target}.npy")
            correction = booster.predict(base, num_threads=8).astype(np.float32)
            reference = np.maximum(0, base[:, 44] + np.float32(0.35) * correction)
            centres = np.load(work / "ancestors" / str(year) / f"{target}_centres.npy")
            full = np.column_stack([base, reference, centres.reshape(11, -1).T])
            grid = full.T.reshape(COLUMNS, *builder.raw.shape).astype(np.float32)
            if not np.isfinite(grid).all():
                raise ValueError("OOF context contains non-finite values")
            np.save(output, grid)
        del booster
        gc.collect()
    for p in (work / "context").glob("202[34]-*.npy"):
        p.unlink()
    scores = {}
    for period, targets in (
        ("2007-2022", dates("2007-01", "2023-01")),
        ("2015-2020", dates("2015-01", "2021-01")),
    ):
        squared, count = 0.0, 0
        for target in targets:
            forecast = np.load(work / "context" / f"{target}.npy", mmap_mode="r")[
                BASELINE
            ]
            truth = builder.raw.truth(target, "2023-01")
            difference = forecast.astype(np.float64) - truth
            squared += float(np.square(difference).sum())
            count += difference.size
        scores[period] = {"rmse": float(np.sqrt(squared / count)), "cells": count}
    atomic_json(
        work / "prepared.json",
        {
            **identity,
            "months": 192,
            "context_files": {
                p.name: sha256(p) for p in sorted((work / "context").glob("*.npy"))
            },
            "unet_fit": "new_chronological_prefixes",
            "tabular_fit": "new_chronological_prefixes",
            "historical_tabular_oof": scores,
        },
    )
    log("OOF and final contexts prepared")


def _tensor(value, encoded, mean, scale):
    normalized = np.clip((value - mean[:, None, None]) / scale[:, None, None], -15, 15)
    return (
        torch.from_numpy(np.ascontiguousarray(normalized[None])).cuda(),
        {k: torch.from_numpy(v).cuda() for k, v in encoded.items()},
        torch.from_numpy(value[BASELINE].copy()).cuda(),
    )


def fit_temporal(data, work, seed):
    folder = work / "models" / f"temporal_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if done(folder):
        return
    config = read(data / "training.json")["temporal"]
    options = {
        **config,
        "input_channels": COLUMNS,
        "width": 32,
        "coarse_stride": 4,
        "threads": 8,
        "max_epochs": 24,
        "epochs": 11,
    }
    dates_train = dates("2007-01", "2023-01")
    total, square = np.zeros(COLUMNS), np.zeros(COLUMNS)
    raw = RawData(data / "raw")
    for target in dates_train:
        value = context(work, target).reshape(COLUMNS, -1).astype(np.float64)
        total += value.sum(1)
        square += np.square(value).sum(1)
    count = len(dates_train) * np.prod(raw.shape)
    mean = (total / count).astype(np.float32)
    scale = np.sqrt(np.maximum(square / count - (total / count) ** 2, 1e-4)).astype(
        np.float32
    )
    np.savez(folder / "statistics.npz", mean=mean, scale=scale)
    runtime(options, seed)
    torch.backends.cudnn.allow_tf32 = True
    model = Corrector(options).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=options["weight_decay"],
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=24, eta_min=options["learning_rate"] / 10
    )
    for epoch in range(11):
        model.train()
        loss_sum = 0.0
        for target in np.random.default_rng(seed + epoch * 1009).permutation(
            dates_train
        ):
            value = context(work, target)
            encoded = members(raw, target, value[BASELINE], 4)
            x, inputs, baseline = _tensor(value, encoded, mean, scale)
            truth = torch.from_numpy(raw.truth(target, "2023-01")).cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                delta = model(x, inputs).float()
                loss = (baseline + delta - truth).square().mean() + options[
                    "residual_penalty"
                ] * delta.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite temporal corrector objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            loss_sum += float(loss.detach())
        schedule.step()
        atomic_json(
            folder / "progress.json",
            {"epoch": epoch + 1, "total": 11, "objective": loss_sum / len(dates_train)},
        )
        log(f"temporal corrector {seed}: epoch {epoch + 1}/11")
    torch.save(model.state_dict(), folder / "model.pt")
    model_receipt(
        folder,
        ["model.pt", "statistics.npz"],
        seed=seed,
        options=options,
        training_end="2022-12",
    )
    del model, optimizer, schedule
    torch.cuda.empty_cache()


def fit_gefs(data, work):
    folder = work / "models/gefs"
    folder.mkdir(parents=True, exist_ok=True)
    if done(folder):
        return
    raw = RawData(data / "raw")
    train_dates = dates("2007-01", "2023-01")
    cells = np.prod(raw.shape)
    first = context(work, train_dates[0])
    extra = extra_columns(
        train_dates[0],
        read_source(data / "weather", "gefs", train_dates[0]),
        raw.lat,
        raw.lon,
        first[BASELINE].ravel(),
    )
    ncols = COLUMNS + extra.shape[1]
    with tempfile.TemporaryDirectory(prefix="worcap-gefs-", dir="/dev/shm") as scratch:
        scratch = Path(scratch)
        x = np.lib.format.open_memmap(
            scratch / "features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(train_dates) * cells, ncols),
        )
        y = np.lib.format.open_memmap(
            scratch / "labels.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(train_dates) * cells,),
        )
        for i, target in enumerate(train_dates):
            value = context(work, target).reshape(COLUMNS, -1).T
            baseline = value[:, BASELINE]
            extra = extra_columns(
                target,
                read_source(data / "weather", "gefs", target),
                raw.lat,
                raw.lon,
                baseline,
            )
            rows = slice(i * cells, (i + 1) * cells)
            x[rows] = np.column_stack([value, extra]).astype(np.float32)
            y[rows] = raw.truth(target, "2023-01").ravel() - baseline
            if i % 24 == 23:
                log(f"GEFS training matrix: {i + 1}/{len(train_dates)} months")
        x.flush()
        y.flush()
        fit_tree(folder, x, y, read(data / "training.json")["gefs"])
        del x, y


def train(data, work):
    data, work = Path(data), Path(work)
    verify_data(data)
    prepared = read(work / "prepared.json")
    if prepared["data_manifest"] != sha256(data / "manifest.json"):
        raise ValueError("Prepared inputs do not match the data package")
    if prepared["source_files"] != source_fingerprint():
        raise ValueError("Training source differs from the OOF preparation source")
    availability = read(work / "availability.json")
    for seed in SEEDS:
        fit_temporal(data, work, seed)
    fit_gefs(data, work)
    atomic_json(
        work / "trained.json",
        {
            "models": ["unet", "tabular", "temporal_42", "temporal_2026", "gefs"],
            "training_end": "2022-12",
            "data_manifest": sha256(data / "manifest.json"),
            "strict_origin_available": availability["strict_origin_available"],
            "source_files": source_fingerprint(),
        },
    )


def fresh_test_contexts(data, work):
    """Execute the final U-Net and tabular weights on the test origins."""
    builder = Builder(data, work / "ancestors")
    stats_path = work / "ancestors/2023/unet_statistics.npz"
    external_path = work / "ancestors/2023/atmosphere_statistics.npz"
    with np.load(stats_path, allow_pickle=False) as z:
        stats = dict(z)
    with np.load(external_path, allow_pickle=False) as z:
        external = dict(z)
    if int(stats["training_end"]) != index("2023-01") - 1:
        raise ValueError("U-Net normalisation includes future labels")
    if int(external["training_end"]) != index("2023-01") - 1:
        raise ValueError("Atmospheric normalisation includes future rows")
    if not done(work / "ancestors/2023") or not done(work / "models/tabular_2023"):
        raise ValueError("Missing verified final U-Net or tabular model")
    configure_torch(42, 1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = unet_model().cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(
        torch.load(
            work / "ancestors/2023/unet.pt", map_location="cuda", weights_only=True
        )
    )
    tree = lgb.Booster(model_file=str(work / "models/tabular_2023/model.txt"))
    examples = Examples(builder.data, builder.archive, stats, True)
    result = []
    with torch.inference_mode():
        for target in dates("2023-01", "2025-01"):
            item = examples.example(index(target), training=False)
            x = torch.from_numpy(item["features"][None]).cuda()
            baseline = torch.from_numpy(item["baseline"][None]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                unet = model(x, baseline).float().clamp_min(0).cpu().numpy()[0]
            base = builder.base_features(2023, target, unet, stats, external)
            centres = builder.centres(
                2023, target, base[:, 43].reshape(builder.raw.shape)
            )
            correction = tree.predict(base, num_threads=8).astype(np.float32)
            reference = np.maximum(0, base[:, 44] + np.float32(0.35) * correction)
            value = np.column_stack([base, reference, centres.reshape(11, -1).T])
            value = value.T.reshape(COLUMNS, *builder.raw.shape).astype(np.float32)
            if not np.isfinite(value).all():
                raise ValueError("Non-finite fresh test context")
            result.append(value)
    del model
    torch.cuda.empty_cache()
    return result


def predict(data, work, output):
    data, work, output = Path(data), Path(work), Path(output)
    verify_data(data)
    trained = read(work / "trained.json")
    if trained["data_manifest"] != sha256(data / "manifest.json"):
        raise ValueError("Model/data identity mismatch")
    if trained["source_files"] != source_fingerprint():
        raise ValueError("Inference source differs from the model training source")
    if not done(work / "models/gefs"):
        raise ValueError("Missing verified GEFS corrector")
    contexts = fresh_test_contexts(data, work)
    raw = RawData(data / "raw")
    boosters = lgb.Booster(model_file=str(work / "models/gefs/model.txt"))
    temporals = {}
    for seed in SEEDS:
        folder = work / "models" / f"temporal_{seed}"
        if not done(folder):
            raise ValueError(f"Missing temporal corrector {seed}")
        options = read(folder / "result.json")["options"]
        runtime(options, seed)
        model = Corrector(options).cuda().eval()
        model.load_state_dict(
            torch.load(folder / "model.pt", map_location="cuda", weights_only=True)
        )
        with np.load(folder / "statistics.npz", allow_pickle=False) as z:
            temporals[seed] = (model, z["mean"], z["scale"])
    grids = []
    for target, value in zip(dates("2023-01", "2025-01"), contexts, strict=True):
        baseline = value[BASELINE]
        encoded = members(raw, target, baseline, 4)
        predictions = []
        with torch.inference_mode():
            for seed in SEEDS:
                model, mean, scale = temporals[seed]
                x, inputs, reference = _tensor(value, encoded, mean, scale)
                predictions.append(
                    (reference + model(x, inputs)).clamp_min(0).cpu().numpy()
                )
        temporal = np.mean(predictions, axis=0, dtype=np.float64).astype(np.float32)
        extra = extra_columns(
            target,
            read_source(data / "weather", "gefs", target),
            raw.lat,
            raw.lon,
            baseline.ravel(),
        )
        features = np.column_stack([value.reshape(COLUMNS, -1).T, extra]).astype(
            np.float32
        )
        delta = boosters.predict(features, num_threads=8).astype(np.float32)
        reference = baseline.ravel()
        first_clip = np.maximum(0, reference + np.float32(0.35) * delta)
        gefs = (
            np.maximum(
                0,
                reference.astype(np.float64)
                + (1 / 0.35) * (first_clip.astype(np.float64) - reference),
            )
            .astype(np.float32)
            .reshape(raw.shape)
        )
        mixed = (
            gefs.astype(np.float64) + TEMPORAL_WEIGHT * (temporal.astype(np.float64) - gefs)
        ).astype(np.float32)
        grids.append(mixed)
        log(f"Predicted {target}")
    predictions = np.stack(grids)
    with np.load(data / "raw/coordinates.npz", allow_pickle=False) as z:
        lat, lon = z["latitude"], z["longitude"]
    rows = write_submission(
        data / "sample_submission.csv",
        output,
        predictions,
        np.arange("2023-01", "2025-01", dtype="datetime64[M]"),
        lat,
        lon,
    )
    atomic_json(
        output.with_suffix(".json"),
        {
            "rows": rows,
            "sha256": sha256(output),
            "training_end": "2022-12",
            "generated_by": "fresh_model_inference",
            "data_manifest": sha256(data / "manifest.json"),
            "reported_kaggle_rmse": None,
            "strict_origin_available": read(work / "availability.json")[
                "strict_origin_available"
            ],
            "source_files": source_fingerprint(),
            "work_files": {
                name: sha256(work / name)
                for name in ("binding.json", "prepared.json", "trained.json", "availability.json")
            },
            "model_receipts": {
                name: sha256(work / name)
                for name in (
                    "ancestors/2023/result.json",
                    "models/tabular_2023/result.json",
                    "models/temporal_42/result.json",
                    "models/temporal_2026/result.json",
                    "models/gefs/result.json",
                )
            },
            "environment": environment_receipt(),
        },
    )
    return rows
