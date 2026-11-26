# Wind forecasts at La Baule: which model should you trust?

*[Version française](README_FR.md)*

Evaluation of past wind forecasts from **GFS**, **AROME** and **ECMWF** against real
measurements in the bay of La Baule (France), followed by machine-learning correction of these forecasts.

## Goal

When preparing a sailing session, you compare several models (Windguru, Windy...)
without really knowing which one is least wrong, or in which direction. This project answers
these questions with data:

1. Which model best forecasts the wind at La Baule: speed, direction, gusts?
2. Are the errors systematic (bias)? Do they depend on wind strength, direction, time of day?
3. Can these errors be corrected with a machine-learning model (gradient boosting), better than a simple bias correction?

The core of the project is **sailable conditions**: mean wind of at least 12 knots
from the SE to WNW sector. That is where a forecast error costs you a session.

## Spot and stations

| | Location | Role |
|---|---|---|
| **Spot**: La Baule / Pont Mahé | bay of La Baule (Loire-Atlantique) | sailable wind: SE to WNW sector (135° to 292.5°), ≥ 12 knots |
| **Météo-France station 44184001**, Pointe de Chémoulin | 47°14'01"N, 2°17'54"W, alt. 14 m | **reference**: well-exposed station facing the sea |
| **Winds-Up beacon** (spot 7) | 47.2586 N, 2.347 W | secondary: closest to the spot, but possibly sheltered |

Forecasts are extracted at the grid points closest to each station.

## Data

### Sources and periods

**Forecasts**: [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)
(10 m wind speed and direction, gusts), hourly data.

| Model | Resolution | Available since |
|---|---|---|
| GFS (`gfs_global`) | 0.11° (≈ Windguru's "GFS 13 km") | 2021-03-23 |
| AROME France HD (`meteofrance_arome_france_hd`) | ≈ 1.5 km | 2022-11-13 |
| ECMWF IFS (`ecmwf_ifs025`) | 0.25° | 2024-02-03 |

This API only keeps the first hours of each run, i.e. the freshest forecast.
The errors measured here are therefore those of a **very short-range** forecast.

**Observations**:
- Chémoulin station: [Météo-France open data](https://meteo.data.gouv.fr), hourly files
  for the Loire-Atlantique department (44) since 2021. We use the 10-min mean wind (FF), its direction (DD),
  the maximum 10-min mean wind within the hour (FXY) and the maximum gust (FXI), converted from m/s to knots.
  Météo-France quality codes are kept (`q_*` columns).
- Winds-Up beacon: CSV export (not public) since 2021, one reading every ~2 min during the day
  and every ~2 h at night.

Everything is in **UTC** and all speeds are in **knots**.

### Cleaning the Winds-Up beacon (`src/parse_windsup.py`)

1. Time taken from the Unix timestamp (UTC) rather than the text date, which is in Paris local time.
2. Duplicate timestamps removed.
3. Readings above 60 knots removed: these are outliers, going up to 127 knots.
4. Manually identified outage removed (24/11 → 13/12/2024, ~19 days at 0 knots).
5. Runs of 0 knots longer than 3 h removed: these are outages, not dead calm.
6. Hourly aggregation: mean speed, direction as a **circular mean** (otherwise the mean of
   350° and 10° would give 180°), maximum and max gust.
7. `reliable` flag: true if the hour contains more than 10 readings. At night, an "hourly mean"
   often boils down to a single reading.

### Final dataset (`src/build_dataset.py`)

One row per observed hour, with the forecasts of the 3 models aligned on the same UTC hour.
The "sailable" and "strong wind" subsets are not stored in this file: they depend
on both the observation and the evaluated forecast (observed OR forecast), so they are computed at analysis time.
Files: `data/processed/dataset_chemoulin.csv` and `data/processed/dataset_la_baule.csv`.

## Method

### Metrics (`src/metrics.py`)

Convention: **error = forecast − observation**. A positive error means the model overestimates the wind.

- **Speed**: bias (mean error), MAE (mean absolute error), RMSE (penalises large errors more).
- **Direction**: **circular** error, wrapped between −180° and +180° (350° forecast for 10° observed
  gives −20°, not 340°). It is only computed when both the observed and forecast wind exceed
  5 knots: below that, direction is not stable enough to be meaningful.
- **Gusts**: bias.
- **"Sailable hour" event**: forecast / observed contingency table, with
  - POD (probability of detection): share of actual sailable hours that the model had forecast;
  - FAR (false alarm ratio): share of hours forecast as sailable that turned out not to be.

Each metric uses all hours where both the model and the observation have a value.
The models therefore do not all cover the same period.

### Sailable filter: observed **OR** forecast

An hour enters the "sailable" subset if it is sailable **in the observation OR
in the model's forecast**. The same rule applies to "strong winds" (≥ 25 knots).

Why: if we only keep hours where the *measured* wind was strong, we precisely select
the cases where the wind exceeded the forecast. All models then look too weak,
through a simple regression-to-the-mean effect. Filtering on the forecast alone produces the opposite effect.
The union of both favours neither error direction.

### Common hours

With the `--common` option, we only keep hours where all 3 models have a speed
and direction forecast, i.e. from February 2024 onwards (start of ECMWF). The models are then
compared **on exactly the same weather situations**. ECMWF rows do not change,
since its period defines the common hours.

### Comparing the two stations (`src/compare_stations.py`)

We keep hours present in both datasets where the beacon is reliable (`reliable`).
We compute the bias (La Baule − Chémoulin) and the speed correlation, per 45° sector
of the direction measured at Chémoulin.

## Results

All figures come from the files in [`results/`](results/). Speeds in knots.

### Reference station: Chémoulin

All available hours (`metriques_erreur_chemoulin.csv`):

| Subset | Model | Hours | Bias | MAE | RMSE | Direction MAE | Gust bias |
|---|---|---|---|---|---|---|---|
| all | GFS | 48,442 | +0.60 | 3.53 | 4.42 | 16.1° | −2.07 |
| all | AROME | 34,631 | −0.67 | **2.95** | **3.80** | **12.9°** | −0.83 |
| all | ECMWF | 24,012 | −2.69 | 4.23 | 5.67 | 17.8° | −2.82 |
| sailable | GFS | 14,796 | −2.95 | **4.15** | **5.05** | 13.1° | −4.28 |
| sailable | AROME | 10,661 | −3.98 | 4.42 | 5.30 | **11.2°** | −2.66 |
| sailable | ECMWF | 6,661 | −8.25 | 8.28 | 9.23 | 16.1° | −5.79 |
| strong winds | GFS | 2,649 | −5.96 | **6.99** | **7.87** | 10.9° | −7.75 |
| strong winds | AROME | 1,911 | −7.12 | 7.42 | 8.18 | **9.8°** | −5.92 |
| strong winds | ECMWF | 967 | −14.09 | 14.09 | 14.68 | 14.8° | −11.04 |

Key takeaways:
- **Over all hours, AROME is the most accurate**, both in speed and direction.
- **In sailable conditions, all models underestimate the wind**, by about 3 knots for GFS
  and 4 for AROME. The stronger the wind, the larger the underestimation: about −6 to −7 knots
  above 25 knots for GFS and AROME.
- **ECMWF clearly underestimates**: −8.3 knots in sailable conditions, −14.1 in strong winds.
  This may come from its 0.25° grid, whose grid point probably mixes land and sea.
  This is a hypothesis, not yet verified.
- On the same hours (`metriques_erreur_chemoulin_heures_communes.csv`), the ranking stays
  the same. In sailable conditions: bias GFS −2.70, AROME −3.77, ECMWF −8.25;
  MAE GFS 3.97, AROME 4.22, ECMWF 8.28.

Detection of sailable hours (`contingence_navigable_chemoulin.csv`):

| Model | Hits | Misses | False alarms | POD | FAR |
|---|---|---|---|---|---|
| GFS | 9,550 | 3,887 | 1,345 | **0.71** | 0.12 |
| AROME | 6,968 | 3,171 | 504 | 0.69 | 0.07 |
| ECMWF | 2,209 | 4,344 | 96 | 0.34 | **0.04** |

GFS detects the most sailable hours. AROME, with almost as many detections, gives
half as many false alarms. ECMWF is rarely wrong when it forecasts wind,
but it misses two sailable hours out of three.

### Winds-Up beacon at La Baule

All available hours (`metriques_erreur_la_baule.csv`):

| Subset | Model | Hours | Bias | MAE | RMSE | Direction MAE |
|---|---|---|---|---|---|---|
| all | GFS | 42,231 | +1.49 | 2.95 | 3.81 | 21.7° |
| all | AROME | 29,332 | −0.09 | **2.54** | **3.31** | 20.0° |
| all | ECMWF | 19,970 | −1.85 | 3.03 | 3.98 | **19.3°** |
| sailable | GFS | 11,052 | +1.40 | **3.35** | 4.47 | 20.9° |
| sailable | AROME | 7,327 | −0.61 | 3.36 | **4.35** | 20.0° |
| sailable | ECMWF | 3,904 | −4.77 | 5.33 | 6.37 | **17.3°** |

Detection of sailable hours (`contingence_navigable_la_baule.csv`):
GFS POD 0.84 / FAR 0.28; AROME POD 0.78 / FAR 0.21; ECMWF POD 0.49 / FAR 0.12.

Biases are higher than at Chémoulin because the beacon measures weaker wind
(see below). Direction errors (~20°) are inflated by the beacon,
which only reports direction in 45° sectors.

### Gap between the beacon and the exposed station

`ecart_la_baule_chemoulin_par_secteur.csv`: 38,356 common hours, reliable beacon.
Sector based on the direction measured at Chémoulin.

| Sector | Hours | Chémoulin | La Baule | Bias (La Baule − Chémoulin) | Correlation |
|---|---|---|---|---|---|
| N | 1,803 | 3.05 | 6.15 | +3.10 | 0.40 |
| NE | 5,595 | 4.92 | 9.02 | +4.10 | 0.39 |
| E | 7,463 | 6.79 | 7.62 | +0.83 | 0.52 |
| SE | 1,541 | 10.78 | 8.44 | −2.34 | 0.83 |
| S | 1,942 | 11.51 | 9.32 | −2.18 | 0.81 |
| SW | 3,771 | 14.60 | 12.17 | −2.43 | 0.87 |
| W | 12,377 | 15.54 | 11.47 | **−4.07** | 0.89 |
| NW | 3,864 | 7.58 | 7.08 | −0.50 | 0.64 |
| **All** | 38,356 | 10.41 | 9.51 | −0.90 | 0.76 |

- **Sailable sectors (SE to W)**: the two stations vary together (correlation 0.81 to 0.89),
  but the beacon measures **2 to 4 knots less**. It is probably more sheltered or lower.
  Since the gap is consistent, it can be corrected.
- **Offshore wind (N, NE)**: Chémoulin measures less, and the link between the two stations
  is weak (correlation ≈ 0.4). Chémoulin is sheltered from these directions.

### Errors by time of day and thermal days

To be completed.

### Charts

To be completed (saved in [`grafics/`](grafics/)).

## ML correction

Notebook: [`notebooks/correction_ml.ipynb`](notebooks/correction_ml.ipynb) (Chémoulin station, AROME and GFS).

**Method**
- The model learns the forecast **error** (forecast − measurement); the corrected wind is forecast − predicted error.
  Three forecasts are corrected: AROME, GFS and the average of both.
- Features, forecast values only: mean wind and gusts of both models, min/max forecast wind
  from t−3 h to t+3 h, direction (sine/cosine), hour and month (cyclically encoded).
- Model: `HistGradientBoostingRegressor` (scikit-learn), simple settings, identical for the three forecasts.
- Baselines: **raw** forecast and **mean-bias** correction (learned on the training years).
- **Temporal** split: years run from November to October. Leave-one-year-out cross-validation on 2023, 2024 and 2025
  (never a random shuffle, which would leak near-identical hours of the same day).
  The Nov. 2025 – Oct. 2026 year is held out as the **final test**, evaluated only once.

**Cross-validation results** (average over the 3 years, [`results/cv_correction_vent_chemoulin.csv`](results/cv_correction_vent_chemoulin.csv),
per-year detail in `cv_correction_vent_chemoulin_par_annee.csv`):

| Hours | Forecast | Raw MAE (kn) | Mean-bias MAE (kn) | Gradient boosting MAE (kn) |
|---|---|---|---|---|
| sailable | AROME | 4.41 | 3.87 | **2.18** |
| sailable | GFS | 4.10 | 4.41 | **2.18** |
| all | AROME | 2.99 | 3.02 | **1.67** |
| all | GFS | 3.52 | 3.51 | **1.67** |

- Gradient boosting roughly halves the MAE and brings the sailable-hours bias from −3/−4 kn down to about −0.2 kn.
- The simple mean-bias correction barely helps (it even worsens GFS on sailable hours): the error depends on the situation
  (direction, strength, hour), not on a constant offset.
- All three variants get the same score: they receive the same features (both models), so gradient boosting
  rebuilds the same wind whatever the starting forecast.
- Final test on Nov. 2025 – Oct. 2026: to do.

## Limitations

- **Very short lead time**: the Historical Forecast API only keeps the first hours of each run.
  A forecast checked the day before (D-1) or several days ahead is probably less accurate.
  Open-Meteo's "Previous Runs" API (D-1 to D-7, since 2024, no gusts) will allow measuring this:
  to be completed.
- **Different periods per model**: GFS since 2021, AROME late 2022, ECMWF 2024.
  Comparisons on common hours fix this issue, at the cost of a shorter period.
- **Winds-Up beacon**: probably sheltered (underestimated wind in sailable sectors), direction
  only given in 45° sectors, sparse readings at night. The
  `metriques_erreur_la_baule*.csv` metrics do not yet filter out unreliable hours (`reliable`).
- **Forecast point for La Baule**: the coordinates requested from Open-Meteo are an approximate
  beach location, not exactly that of the beacon.
- **Chémoulin**: sheltered from offshore wind (N, NE). This does not affect the analysis of the sailable sector.
- **Strong winds**: few hours (fewer than 1,000 for ECMWF), so less robust results.

## Reproduce

Requirements: Python 3 with `pandas` and `requests` (`requirements.txt`).

```bash
# Environment
uv venv
uv pip install -r requirements.txt

# 1. Downloads (long) -> data/raw/, not versioned
python fetch_forecasts.py        # Open-Meteo forecasts, 3 models x 2 locations
python fetch_observations.py     # Météo-France observations at Chémoulin

# 2. Winds-Up observations: put the CSV export in data/raw/observations/, then
python src/parse_windsup.py --start 2021-01-01

# 3. Hour-by-hour aligned datasets -> data/processed/
python src/build_dataset.py
python src/build_dataset.py --location la_baule --obs data/raw/observations/windsup.csv

# 4. Results -> results/
python src/metrics.py --location chemoulin [--common]
python src/metrics.py --location la_baule [--common]
python src/compare_stations.py
```

The Winds-Up export is not public and not provided: without it, only the Chémoulin part
is reproducible. The prepared datasets are nevertheless versioned in `data/processed/`.

## Use of AI

I developed this project with the help of the Claude Code coding assistant.
I set the goals and methodological choices (reference station, definition of sailable conditions, observed OR forecast filter, temporal split), I review and run the code, I interpret the results and I make the commits myself.

The field knowledge comes from my own sailing on this spot: sailable sectors, 12-knot threshold, doubts about the beacon's exposure.
