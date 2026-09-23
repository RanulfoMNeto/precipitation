# WORCAP 2026 — Monthly precipitation forecast

**Team:** NoTime

## Task and result

Predict monthly precipitation in mm/day on the competition grid for January 2023–December 2024. The participant reported Kaggle RMSE **1.53208** for this execution and **1.53551** for the original recipe. These scores were not measured locally; private score and rank remain pending.

## Architecture and features

The five models are a spatial U-Net, a LightGBM residual corrector, two PoET temporal correctors with seeds 42 and 2026, and a LightGBM corrector using GEFS. Inputs combine competition observations with ECMWF SEAS5, DWD, Météo-France and NOAA GEFS forecasts.

The U-Net uses 25 channels: atmospheric lags, climatology, location, calendar, and SEAS5 precipitation anomaly and spread. The first LightGBM adds six- and twelve-month atmospheric summaries, precipitation references, the U-Net prediction and spatial descriptors. Its output joins SEAS5 atmospheric statistics and calibrated DWD/Météo-France forecasts in a 98-channel context. PoET attends to SEAS5, DWD and GEFS ensemble members and applies spatial convolutions. The final LightGBM adds GEFS weekly precipitation and coverage features.

The final blend assigns **0.6337257586287943** to the mean PoET prediction and **0.3662742413712057** to the GEFS corrector. Exact transformations and hyperparameters are defined in `src/worcap_forecast/` and its `config/training.json`.

## Training and validation

Chronological prefix training produces 192 out-of-sample monthly contexts for 2007–2022, using 13 U-Net fits and nine first-stage LightGBM fits. These contexts train both PoET models and the GEFS corrector. Normalization and calibration use the available prefix; training labels end in December 2022. Each test prediction uses observations through its own previous month.

First-stage LightGBM OOF RMSE is **1.754724** for 2007–2022 and **1.751633** for 2015–2020. These values do not evaluate the final ensemble. Inference runs all five models and writes 1,885,464 rows in official sample order. Ten integrity, chronology and reconstruction tests passed.

## Execution and data audit

Local execution requires Linux, Python 3.12, PyTorch 2.8.0/CUDA 12.8 and LightGBM 4.6.0. Recommended resources are 64 GiB RAM and a CUDA GPU with 16 GiB VRAM. OOF preparation took 17min37s, final training 8min18s and inference 32s. The [README](../README.md) provides both prepared-artifact and complete reconstruction commands, with pinned environments and storage requirements.

Competition originals were imported and hash-verified; CDS and NOAA inputs were downloaded independently. Receipts bind data, source code, models and CSV. The [audit](DATA_AUDIT.md) records one unresolved evidence question: publication dates of historical hindcasts and reforecasts at simulated forecast origins. Copernicus public access was confirmed by an organizer; historical availability requires review. Data and weight redistribution terms are separate from the code's MIT license.
