"""Chronological U-Net and tabular reconstruction from verified meteorology."""

from __future__ import annotations

import gc
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .base import unet_model
from .features import ResidualExamples, divergence, smooth
from .io import checked, read
from .preprocessing import FIRST, Bilinear, Examples, statistics
from .raw import END, RawData, dates, index, origin
from .runtime import atomic_json, configure_torch, log, sha256
from .weather import interpolate, route


def save_torch(path, state):
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


class Seasonal:
    def __init__(self, root, raw):
        self.first = FIRST
        self.root, self.raw = root, raw
        with np.load(root / "audit/seasonal.npz", allow_pickle=False) as z:
            self.mean, self.spread = z["mean"], z["spread"]
            self.interpolation = Bilinear(
                z["latitude"], z["longitude"], raw.lat, raw.lon
            )

    def fields(self, target, climate):
        if target < END:
            i = target - FIRST
            return (
                self.interpolation(self.mean[i] - climate[target % 12]),
                self.interpolation(self.spread[i]),
            )
        month = str(np.datetime64("1940-01", "M") + np.timedelta64(target, "M"))
        p = self.root / "forecast/seasonal_precipitation" / f"{month}.npz"
        record = checked(p)
        origin(month, record)
        if record["leadtime_month"] != 2 or record["units"] != "mm/day":
            raise ValueError("SEAS5 future target has wrong horizon or units")
        with np.load(p, allow_pickle=False) as z:
            interpolation = Bilinear(
                z["latitude"], z["longitude"], self.raw.lat, self.raw.lon
            )
            return interpolation(z["mean"] - climate[target % 12]), interpolation(
                z["spread"]
            )


class Builder:
    def __init__(self, data, output):
        self.root, self.output = Path(data), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.raw = RawData(self.root / "raw")
        self.archive = Seasonal(self.root, self.raw)
        h, w = self.raw.shape
        self.data = SimpleNamespace(
            height=h,
            width=w,
            n_locations=h * w,
            observed_months=END,
            coordinates={"latitude": self.raw.lat, "longitude": self.raw.lon},
            precipitation=self.raw.precipitation,
            atmosphere=self.raw.atmosphere,
            forecast_atmosphere=[a[11:] for a in self.raw.test],
        )
        self.config = read(self.root / "training.json")
        self.recipes = read(self.root / "audit/recipes.json")
        self.calibration = lru_cache(maxsize=72)(self.calibration)
        for name, digest in read(self.root / "audit/manifest.json")["files"].items():
            path = (self.root / "audit" / name).resolve()
            if (
                not path.is_relative_to((self.root / "audit").resolve())
                or sha256(path) != digest
            ):
                raise ValueError(f"Changed ancestor audit input: {name}")

    def validate(self, year):
        if year not in range(1999, 2024, 2):
            raise ValueError("Unregistered chronological block")

    def prefix_stats(self, year):
        self.validate(year)
        return statistics(self.data, self.archive, index(f"{year}-01") - 1)

    def external_stats(self, year):
        means, spreads = [], []
        for target in dates(last=f"{year}-01"):
            p = self.root / "raw/seasonal_atmosphere" / f"{target}.npz"
            origin(target, checked(p))
            with np.load(p, allow_pickle=False) as z:
                means.append(z["mean"])
                spreads.append(z["spread"])
        prefix = np.stack(means).astype(np.float64)
        climate = prefix.reshape(-1, 12, *prefix.shape[1:]).mean(0)
        anomaly = prefix - climate[np.arange(len(prefix)) % 12]
        scale = np.sqrt(
            np.mean(anomaly.reshape(-1, 12, *prefix.shape[1:]) ** 2, axis=0)
        )
        floor = np.maximum(np.sqrt(np.mean(anomaly**2, axis=(0, 2, 3))) * 0.01, 1e-6)
        return {
            "training_end": np.asarray(index(f"{year}-01") - 1),
            "seas5_climate": climate.astype(np.float32),
            "seas5_scale": np.maximum(scale, floor[None, :, None, None]).astype(
                np.float32
            ),
            "seas5_spread_scale": np.maximum(
                np.mean(spreads, axis=(0, 2, 3), dtype=np.float64), floor
            ).astype(np.float32),
        }

    def unet(self, year, stats, predict_block=True):
        torch = configure_torch(42, 1)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = unet_model().cuda().to(memory_format=torch.channels_last)
        examples = Examples(self.data, self.archive, stats, True)
        folder = self.output / str(year)
        folder.mkdir(parents=True, exist_ok=True)
        c = self.config["unet"]
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
        groups = {}
        for p in model.parameters():
            groups.setdefault(c["weight_decay"] if p.ndim >= 2 else 0.0, []).append(
                p
            )
        optimizer = torch.optim.AdamW(
            [
                {"params": p, "lr": c["learning_rate"], "weight_decay": d}
                for d, p in groups.items()
            ]
        )
        ids = np.arange(FIRST, index(f"{year}-01"))
        epoch, cursor = 0, 0
        order = np.random.default_rng(42).permutation(ids)
        model.train()
        for step in range(self.recipes[str(year)]["unet_updates"]):
            if cursor == len(order):
                epoch, cursor = epoch + 1, 0
                order = np.random.default_rng(42 + 100003 * epoch).permutation(ids)
            batch = order[cursor : cursor + c["batch_size"]]
            multiplier = (
                (step + 1) / c["warmup_updates"]
                if step < c["warmup_updates"]
                else 0.1
                + 0.9
                * (
                    1
                    + math.cos(
                        math.pi
                        * (step + 1 - c["warmup_updates"])
                        / (c["schedule_updates"] - c["warmup_updates"])
                    )
                )
                / 2
            )
            for group in optimizer.param_groups:
                group["lr"] = c["learning_rate"] * multiplier
            optimizer.zero_grad(set_to_none=True)
            for target in batch:
                item = examples.example(int(target), training=True)
                x = torch.from_numpy(item["features"][None]).cuda()
                base = torch.from_numpy(item["baseline"][None]).cuda()
                actual = torch.from_numpy(item["target"][None]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    error = (model(x, base).float() - actual).square()
                    loss = error.sum() / (len(batch) * actual.numel())
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            cursor += len(batch)
        save_torch(folder / "unet.pt", model.state_dict())
        del optimizer
        if not predict_block:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return None
        model.eval()
        result = []
        with torch.inference_mode():
            for t in range(index(f"{year}-01"), index(f"{year + 2}-01")):
                item = examples.example(t)
                x = torch.from_numpy(item["features"][None]).cuda()
                base = torch.from_numpy(item["baseline"][None]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result.append(model(x, base).float().clamp_min(0).cpu().numpy()[0])
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return np.stack(result)

    def base_features(self, year, target, unet, stats, external):
        t = index(target)
        if t <= int(stats["training_end"]) or t <= int(external["training_end"]):
            raise ValueError("Features are not outside the training prefix")
        base = ResidualExamples(self.data, self.archive, stats, True).features(t, unet)
        p = self.root / "raw/seasonal_atmosphere" / f"{target}.npz"
        origin(target, checked(p))
        with np.load(p, allow_pickle=False) as z:
            transform = Bilinear(
                z["latitude"], z["longitude"], self.raw.lat, self.raw.lon
            )
            anomaly = transform(z["mean"] - external["seas5_climate"][t % 12])
            normalized = anomaly / transform(external["seas5_scale"][t % 12])
            spread = (
                transform(z["spread"]) / external["seas5_spread_scale"][:, None, None]
            )
            fields = [v for pair in zip(normalized, spread, strict=True) for v in pair]
            _, _, q, u, v = transform(z["mean"])
        qu, qv = q * u, q * v
        fields += [
            qu,
            qv,
            -divergence(smooth(u, 5), smooth(v, 5), self.raw.lat, self.raw.lon),
            -divergence(smooth(qu, 5), smooth(qv, 5), self.raw.lat, self.raw.lon),
        ]
        return np.column_stack([base, np.stack(fields).reshape(14, -1).T]).astype(
            np.float32
        )

    def calibration(self, centre, system, month, year):
        forecasts, observed = [], []
        for past in range(1994, year):
            target = f"{past}-{month:02d}"
            init = str(np.datetime64(target, "M") - np.timedelta64(1, "M"))
            p = self.root / "audit/calibration" / centre / system / f"{init}.npz"
            if not p.exists():
                continue
            origin(target, checked(p))
            with np.load(p, allow_pickle=False) as z:
                mean = (
                    z["members"].mean(0, dtype=np.float64)
                    if "members" in z
                    else z["mean"]
                )
                forecasts.append(
                    interpolate(
                        mean, z["latitude"], z["longitude"], self.raw.lat, self.raw.lon
                    )
                )
            observed.append(self.raw.truth(target, f"{year}-01"))
        if len(forecasts) < 10:
            raise ValueError("Insufficient matched-system calibration history")
        return (
            np.mean(forecasts, axis=0, dtype=np.float64).astype(np.float32),
            np.std(forecasts, axis=0, ddof=1, dtype=np.float64).astype(np.float32),
            np.mean(observed, axis=0, dtype=np.float64).astype(np.float32),
        )

    def centres(self, year, target, s0):
        result, calibrated = [], []
        for centre in ("meteo_france", "dwd"):
            init = str(np.datetime64(target, "M") - np.timedelta64(1, "M"))
            p = self.root / "raw/seasonal" / centre / f"{target}.npz"
            origin(target, checked(p), centre)
            with np.load(p, allow_pickle=False) as z:
                mean = (
                    z["members"].mean(0, dtype=np.float64)
                    if "members" in z
                    else z["mean"]
                )
                spread = (
                    z["members"].std(0, ddof=1, dtype=np.float64)
                    if "members" in z
                    else z["spread"]
                )
                mean, spread = (
                    interpolate(
                        v, z["latitude"], z["longitude"], self.raw.lat, self.raw.lon
                    )
                    for v in (mean, spread)
                )
            climate, sigma, truth = self.calibration(
                centre, route(centre, init), int(target[5:]), year
            )
            anomaly = mean - climate
            corrected = np.maximum(0, truth + anomaly)
            calibrated.append(corrected)
            result += [mean, spread, anomaly, corrected - s0, anomaly / (sigma + 0.5)]
        return np.stack([*result, calibrated[1] - calibrated[0]])

    def block(self, year):
        self.validate(year)
        folder = self.output / str(year)
        folder.mkdir(parents=True, exist_ok=True)
        receipt = folder / "result.json"
        if receipt.exists():
            record = read(receipt)
            for name, digest in record["files"].items():
                if sha256(folder / name) != digest:
                    raise ValueError(f"Changed OOF block file: {name}")
            return record
        stats, external = self.prefix_stats(year), self.external_stats(year)
        np.savez(folder / "unet_statistics.npz", **stats)
        np.savez(folder / "atmosphere_statistics.npz", **external)
        unet = self.unet(year, stats, predict_block=year < 2023)
        targets = dates(f"{year}-01", f"{year + 2}-01")
        if year < 2023:
            np.save(folder / "unet.npy", unet)
            for i, target in enumerate(targets):
                matrix = self.base_features(year, target, unet[i], stats, external)
                np.save(folder / f"{target}.npy", matrix)
                if year >= 2007:
                    np.save(
                        folder / f"{target}_centres.npy",
                        self.centres(year, target, matrix[:, 43].reshape(self.raw.shape)),
                    )
        record = {
            "year": year,
            "training_end": f"{year - 1}-12",
            "test_predictions_cached": False,
            "targets": targets.tolist(),
            "files": {
                p.name: sha256(p)
                for p in sorted(folder.iterdir())
                if p.suffix in (".npz", ".npy", ".pt")
            },
        }
        atomic_json(receipt, record)
        log(f"Rebuilt U-Net and base features for {year}–{year + 1}")
        return record

    def samples(self):
        rows, labels, targets = [], [], []
        for year in range(1999, 2023, 2):
            folder = self.output / str(year)
            for target in dates(f"{year}-01", f"{year + 2}-01"):
                t = index(target)
                matrix = np.load(folder / f"{target}.npy", mmap_mode="r")
                ids = np.sort(
                    np.random.default_rng(42 + 100003 * t).choice(
                        self.data.n_locations,
                        min(4096, self.data.n_locations),
                        replace=False,
                    )
                )
                rows.append(np.asarray(matrix[ids, :86]))
                labels.append(
                    self.raw.truth(target, "2023-01").ravel()[ids] - matrix[ids, 44]
                )
                targets.append(np.full(len(ids), target, dtype="datetime64[M]"))
        folder = self.output / "tabular_samples"
        folder.mkdir(exist_ok=True)
        for name, parts in (("x", rows), ("y", labels), ("dates", targets)):
            np.save(folder / f"{name}.npy", np.concatenate(parts))
        return {p.name: sha256(p) for p in folder.iterdir()}
