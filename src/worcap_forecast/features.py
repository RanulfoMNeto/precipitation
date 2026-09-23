"""Causal tabular features from observations and seasonal forecasts."""

from __future__ import annotations

import numpy as np

from .preprocessing import ATMOSPHERE, Examples, require

RADIUS = 6_371_000.0
SPATIAL = (4, 6, 8, 12, 14, 16, 23, 24, 43, 44)


class TreeExamples(Examples):
    """Entradas atmosféricas, médias temporais e base de chuva em mm/dia."""

    def features(self, target):
        item = super().example(target, training=False)
        channels = list(item["features"])
        times = np.arange(target - 12, target)
        require(
            times[0] >= 0 and times[-1] < self.data.observed_months + 23,
            "Histórico tabular indisponível",
        )
        observed = self.data.observed_months
        for variable in range(9):
            values = np.stack(
                [
                    self.data.atmosphere[variable][t]
                    if t < observed
                    else self.data.forecast_atmosphere[variable][t - observed + 1]
                    for t in times
                ]
            ).astype(np.float32)
            values = (
                values - self.stats["atmosphere_mean"][variable, times % 12, None, None]
            ) / self.stats["atmosphere_std"][variable, times % 12, None, None]
            channels.extend([values[-6:].mean(0), values.mean(0)])
        channels.append(item["baseline"])
        features = np.stack(channels).reshape(44, -1).T.copy()
        require(np.isfinite(features).all(), "Atributos LightGBM não finitos")
        return features, item["baseline"].reshape(-1)


def names():
    columns = [f"{v}_{lag}" for v in ATMOSPHERE for lag in ("lag1_z", "lag3_z")]
    columns += [
        "climatology_scaled",
        "latitude",
        "longitude",
        "month_sin",
        "month_cos",
        "seas5_anomaly_z",
        "seas5_spread_scaled",
    ]
    columns += [f"{v}_{lag}" for v in ATMOSPHERE for lag in ("lag6_z", "lag12_z")]
    columns += ["baseline", "unet", "unet_minus_baseline"]
    columns += [f"{columns[i]}_mean{size}" for i in SPATIAL for size in (5, 17)]
    columns += [
        "pressure_contrast17",
        "humidity_contrast17",
        "q850_u850",
        "q850_v850",
        "wind_convergence850_s",
        "moisture_convergence850_s",
    ]
    return columns


def smooth(field, size):
    from scipy.ndimage import uniform_filter

    # Somente dimensões espaciais; bordas replicadas, sem envolver lados do domínio.
    return uniform_filter(
        np.asarray(field, np.float64), size=size, mode="nearest"
    ).astype(np.float32)


def divergence(east, north, latitude, longitude):
    """Divergência horizontal esférica; eixos ascendentes ou descendentes em graus."""
    phi, lam = np.deg2rad(latitude), np.deg2rad(longitude)
    for axis in (phi, lam):
        require(len(axis) >= 2 and np.isfinite(axis).all(), "Eixo espacial inválido")
        require(
            np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0),
            "Eixo não monotônico",
        )
    cosine = np.cos(phi)[:, None]
    require(np.min(cosine) > 1e-5, "Grade inclui singularidade polar")
    require(
        np.shape(east) == np.shape(north) == (len(phi), len(lam)), "Grade incompatível"
    )
    return (
        (
            np.gradient(np.asarray(east, np.float64), lam, axis=1)
            + np.gradient(np.asarray(north, np.float64) * cosine, phi, axis=0)
        )
        / (RADIUS * cosine)
    ).astype(np.float32)


class ResidualExamples(TreeExamples):
    """72 atributos: 44 existentes, U-Net, correção U-Net e 26 descritores espaciais/físicos."""

    def raw_lag(self, variable, target):
        t = target - 1
        observed = self.data.observed_months
        require(0 <= t < observed + 23, "Mês atmosférico indisponível")
        array = (
            self.data.atmosphere[variable][t]
            if t < observed
            else self.data.forecast_atmosphere[variable][t - observed + 1]
        )
        return np.asarray(array, np.float32)

    def features(self, target, unet):
        # Esta classe não admite estatísticas que incluam o próprio alvo.
        require(
            target > int(self.stats["training_end"]), "Atributos não são fora do treino"
        )
        x, base = super().features(target)
        unet = np.asarray(unet, np.float32).reshape(self.height, self.width)
        require(
            np.isfinite(unet).all() and (unet >= 0).all(),
            "U-Net deve estar finito e truncado em zero",
        )
        fields = list(x.T.reshape(44, self.height, self.width))
        fields.extend([unet, unet - base.reshape(unet.shape)])
        fields.extend(smooth(fields[i], size) for i in SPATIAL for size in (5, 17))
        fields.extend(
            [fields[4] - smooth(fields[4], 17), fields[6] - smooth(fields[6], 17)]
        )
        q, u, v = [self.raw_lag(i, target) for i in (3, 7, 8)]
        qu, qv = q * u, q * v
        lat, lon = self.data.coordinates["latitude"], self.data.coordinates["longitude"]
        fields.extend(
            [
                qu,
                qv,
                -divergence(smooth(u, 5), smooth(v, 5), lat, lon),
                -divergence(smooth(qu, 5), smooth(qv, 5), lat, lon),
            ]
        )
        matrix = np.stack(fields).reshape(len(names()), -1).T.copy()
        require(
            matrix.shape == (self.n_locations, len(names())),
            "Esquema de atributos diferente",
        )
        require(np.isfinite(matrix).all(), "Atributos residuais não finitos")
        return matrix
