"""
Construit le jeu de données d'analyse pour la Pointe de Chémoulin :
relevés Météo-France + prévisions GFS, AROME HD et ECMWF, alignés heure par heure (UTC).

Résultat : data/processed/dataset_chemoulin.csv, une ligne par heure observée,
vitesses en nœuds, directions en degrés.

Usage (depuis la racine du projet) :  python src/build_dataset.py
"""
from pathlib import Path

import pandas as pd

OBS_FILE = Path("data/raw/observations/chemoulin.csv")
FORECAST_DIR = Path("data/raw/forecasts")
OUT = Path("data/processed/dataset_chemoulin.csv")

# Nom court du modèle (suffixe des colonnes) -> nom du fichier Open-Meteo
MODELS = {
    "gfs": "gfs_global",
    "arome": "meteofrance_arome_france_hd",
    "ecmwf": "ecmwf_ifs025",
}

# Conditions navigables à La Baule : secteur SE à ONO, vent moyen >= 12 nœuds
DIR_MIN, DIR_MAX = 135, 292.5
NAVIGABLE_SPEED = 12
STRONG_WIND_SPEED = 30


def load_observations():
    """Relevés de la station, renommés avec le suffixe _obs."""
    df = pd.read_csv(OBS_FILE, parse_dates=["time"])
    return df.rename(columns={
        "wind_speed": "speed_obs",
        "wind_direction": "dir_obs",
        "wind_speed_max": "speed_max_obs",
        "wind_gusts": "gust_obs",
    })


def load_forecast(short_name, file_name):
    """Prévisions d'un modèle, colonnes suffixées par son nom court (speed_gfs...)."""
    df = pd.read_csv(FORECAST_DIR / f"chemoulin_{file_name}.csv", parse_dates=["time"])
    return df.rename(columns={
        "wind_speed_10m": f"speed_{short_name}",
        "wind_direction_10m": f"dir_{short_name}",
        "wind_gusts_10m": f"gust_{short_name}",
    })


if __name__ == "__main__":
    # 1. Les relevés servent de référence : on part d'eux et on y accroche les prévisions
    df = load_observations()
    for short_name, file_name in MODELS.items():
        df = df.merge(load_forecast(short_name, file_name), on="time", how="left")

    # 2. Indicateurs calculés sur le vent MESURÉ (ce qui s'est vraiment passé sur l'eau)
    good_sector = df["dir_obs"].between(DIR_MIN, DIR_MAX)
    df["navigable"] = good_sector & (df["speed_obs"] >= NAVIGABLE_SPEED)
    df["strong_wind"] = df["speed_obs"] >= STRONG_WIND_SPEED

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"-> {OUT} : {len(df)} lignes")
    print(f"Période : {df['time'].min()} -> {df['time'].max()}")
    print(f"Heures navigables : {df['navigable'].mean():.1%}")
