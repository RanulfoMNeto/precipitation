"""Causal U-Net features and training-prefix statistics."""

import numpy as np

FIRST = (1994 - 1940) * 12
ATMOSPHERE = (
    "air_temperature_2m",
    "cloud_cover",
    "surface_pressure",
    "specific_humidity_850hpa",
    "relative_humidity_850hpa",
    "air_temperature_850hpa",
    "geopotential_850hpa",
    "eastward_wind_850hpa",
    "northward_wind_850hpa",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Bilinear:
    """Interpolação geográfica bilinear, sem extrapolação e sem dependência de SciPy."""

    def __init__(self, latitude, longitude, target_lat, target_lon):
        self.y, self.wy = self.axis(latitude, target_lat)
        self.x, self.wx = self.axis(longitude, target_lon)

    @staticmethod
    def axis(source, destination):
        source, destination = np.asarray(source), np.asarray(destination)
        require(
            len(source) >= 2 and np.all(np.diff(source) > 0),
            "Grade fonte não crescente",
        )
        require(
            np.isfinite(destination).all()
            and destination.min() >= source[0]
            and destination.max() <= source[-1],
            "Interpolação exigiria extrapolação",
        )
        lower = np.clip(
            np.searchsorted(source, destination, side="right") - 1, 0, len(source) - 2
        )
        weight = (destination - source[lower]) / (source[lower + 1] - source[lower])
        return lower, weight.astype(np.float32)

    def __call__(self, array):
        y, x, wy, wx = self.y, self.x, self.wy[:, None], self.wx[None, :]
        return (
            (1 - wy)
            * (
                (1 - wx) * array[..., y[:, None], x]
                + wx * array[..., y[:, None], x + 1]
            )
            + wy
            * (
                (1 - wx) * array[..., y[:, None] + 1, x]
                + wx * array[..., y[:, None] + 1, x + 1]
            )
        ).astype(np.float32)


def statistics(data, archive, end, check=lambda: None):
    require(
        end % 12 == 11 and FIRST <= end < len(data.precipitation),
        "Corte SEAS5 inválido",
    )
    require(archive.first == FIRST, "Início do arquivo externo incompatível")
    years = (end + 1 - FIRST) // 12
    climate = np.zeros((12, data.height, data.width), np.float64)
    for start in range(FIRST, end + 1, 12):
        check()
        climate += data.precipitation[start : start + 12]
    climate = (climate / years).astype(np.float32)
    forecast = archive.mean[: end + 1 - FIRST]
    seas_climate = (
        forecast.reshape(years, 12, *forecast.shape[-2:])
        .mean(axis=0, dtype=np.float64)
        .astype(np.float32)
    )
    variance = np.zeros_like(climate, dtype=np.float64)
    spread_sum = 0.0
    for target in range(FIRST, end + 1):
        check()
        anomaly, spread = archive.fields(target, seas_climate)
        variance[target % 12] += np.square(anomaly.astype(np.float64))
        spread_sum += float(spread.mean(dtype=np.float64))
    seas_scale = np.maximum(np.sqrt(variance / years), 1e-3).astype(np.float32)
    # Escala mensal por variável, compartilhada espacialmente; sem cache 9×12×H×W.
    atmosphere_mean, atmosphere_std = [], []
    for array in data.atmosphere:
        means, deviations = [], []
        for month in range(12):
            check()
            ids = np.arange(FIRST + month, end + 1, 12)
            count = len(ids) * data.n_locations
            mean = sum(array[t].sum(dtype=np.float64) for t in ids) / count
            variance = (
                sum(np.square(array[t].astype(np.float64) - mean).sum() for t in ids)
                / count
            )
            means.append(mean)
            deviations.append(max(float(np.sqrt(variance)), 1e-6))
        atmosphere_mean.append(means)
        atmosphere_std.append(deviations)
    return {
        "schema": np.asarray(1, np.int64),
        "first_target": np.asarray(FIRST, np.int64),
        "training_end": np.asarray(end, np.int64),
        "climatology": climate,
        "climatology_scale": np.asarray(max(float(climate.mean()), 1e-6), np.float32),
        "seas_climatology": seas_climate,
        "seas_scale": seas_scale,
        "spread_scale": np.asarray(
            max(spread_sum / (end + 1 - FIRST), 1e-3), np.float32
        ),
        "atmosphere_mean": np.asarray(atmosphere_mean, np.float32),
        "atmosphere_std": np.asarray(atmosphere_std, np.float32),
    }


class Examples:
    def __init__(self, data, archive, stats, use_seas5):
        self.data, self.archive, self.stats, self.use_seas5 = (
            data,
            archive,
            stats,
            use_seas5,
        )
        self.height, self.width, self.n_locations = (
            data.height,
            data.width,
            data.n_locations,
        )
        self.precipitation = getattr(data, "precipitation", None)
        self.latitude, self.longitude = np.meshgrid(
            data.coordinates["latitude"] / 90,
            data.coordinates["longitude"] / 180,
            indexing="ij",
        )

    def example(self, target, training=False):
        stats, data = self.stats, self.data
        if training:
            require(
                FIRST <= target <= int(stats["training_end"]),
                "Alvo fora do treino SEAS5",
            )
        times = np.arange(target - 3, target)
        observed = data.observed_months
        require(
            times[0] >= 0 and times[-1] < observed + 23,
            "Contexto atmosférico indisponível",
        )
        channels = []
        for variable in range(9):
            values = np.stack(
                [
                    data.atmosphere[variable][t]
                    if t < observed
                    else data.forecast_atmosphere[variable][t - observed + 1]
                    for t in times
                ]
            ).astype(np.float32)
            values = (
                values - stats["atmosphere_mean"][variable, times % 12, None, None]
            ) / stats["atmosphere_std"][variable, times % 12, None, None]
            channels += [values[-1], values.mean(axis=0)]
        month, phase = target % 12, 2 * np.pi * (target % 12) / 12
        climate = stats["climatology"][month]
        baseline = climate.copy()
        channels += [
            climate / stats["climatology_scale"],
            self.latitude,
            self.longitude,
            np.full_like(climate, np.sin(phase)),
            np.full_like(climate, np.cos(phase)),
        ]
        if self.use_seas5:
            anomaly, spread = self.archive.fields(target, stats["seas_climatology"])
            channels += [
                anomaly / stats["seas_scale"][month],
                spread / stats["spread_scale"],
            ]
            baseline += anomaly
        else:
            # Canais sazonais zerados preservam o formato de entrada da rede.
            channels += [np.zeros_like(climate), np.zeros_like(climate)]
        result = {
            "features": np.stack(channels).astype(np.float32),
            "baseline": baseline,
        }
        if training:
            result.update(
                target=data.precipitation[target].copy(),
                target_mask=np.ones_like(climate, bool),
            )
        return result
