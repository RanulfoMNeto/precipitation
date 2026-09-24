# WORCAP 2026 — Monthly precipitation forecast

**Team:** NoTime

## Task and result

The task is to predict mean daily precipitation (mm/day) for each competition grid cell and month from January 2023 through December 2024. **Kaggle submission RMSE: 1.53208.**

## Model and inputs

The five models are a spatial U-Net, a LightGBM residual corrector, two PoET correctors trained with seeds 42 and 2026, and a LightGBM corrector using GEFS. Inputs combine competition observations through December 2022 with ECMWF SEAS5, DWD, Météo-France and NOAA GEFS forecasts. For each target month, observed atmospheric inputs stop at the previous month.

The U-Net uses 25 channels: atmospheric lags, climatology, location, calendar, and SEAS5 precipitation anomaly and spread. The first LightGBM adds longer atmospheric summaries, the U-Net prediction and spatial descriptors. Its output joins SEAS5 atmospheric statistics and calibrated DWD/Météo-France forecasts in a 98-channel context. Each PoET attends to SEAS5, DWD and GEFS ensemble members. The final LightGBM adds GEFS precipitation and coverage features. The final blend assigns **0.6337257586287943** to the mean PoET prediction and **0.3662742413712057** to the GEFS corrector.

## Training and reproduction

Chronological training produces 192 out-of-sample monthly contexts for 2007–2022 using 13 U-Net fits and nine first-stage LightGBM fits. These contexts train both PoET models and the GEFS corrector. Normalization and calibration use the available training prefix; no 2023–2024 precipitation labels are used. Historical first-stage LightGBM RMSE is **1.754724** for 2007–2022 and **1.751633** for 2015–2020; these are component metrics.

The [README](../README.md) specifies the pinned environment, hardware, prepared-artifact path and complete reconstruction commands. The [data audit](DATA_AUDIT.md) records acquisition, provenance and temporal checks. The code is licensed under MIT; data and model artifacts retain separate terms.
