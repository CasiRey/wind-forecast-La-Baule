"""
Construit le jeu de données d'analyse pour un lieu (chemoulin ou la_baule) :
relevés observés + prévisions GFS, AROME HD et ECMWF, alignés heure par heure (UTC).

Le fichier d'observations doit contenir les colonnes time (UTC), wind_speed,
wind_direction, wind_gusts (et éventuellement wind_speed_max), vitesses en nœuds.
C'est le format produit par fetch_observations.py ; il suffira de produire le même
format à partir des relevés Winds-Up pour les brancher ici sans rien changer d'autre.

Résultat : data/processed/dataset_<lieu>.csv, une ligne par heure observée.

Usage (depuis la racine du projet) :
    python src/build_dataset.py                       # Chémoulin, fichier par défaut
    python src/build_dataset.py --location la_baule --obs data/raw/observations/windsup.csv
"""
import argparse
from pathlib import Path

import pandas as pd

FORECAST_DIR = Path("data/raw/forecasts")
OBS_DIR = Path("data/raw/observations")
OUT_DIR = Path("data/processed")

LOCATIONS = ["chemoulin", "la_baule"]   # mêmes noms que dans fetch_forecasts.py

# Nom court du modèle (suffixe des colonnes) -> nom du fichier Open-Meteo
MODELS = {
    "gfs": "gfs_global",
    "arome": "meteofrance_arome_france_hd",
    "ecmwf": "ecmwf_ifs025",
}

# Conditions navigables à La Baule : secteur SE à ONO, vent moyen >= 12 nœuds
DIR_MIN, DIR_MAX = 135, 292.5
NAVIGABLE_SPEED = 12
STRONG_WIND_SPEED = 25   # vents forts, analysés à part


def load_observations(obs_file):
    """Relevés de la station, renommés avec le suffixe _obs."""
    df = pd.read_csv(obs_file, parse_dates=["time"])
    return df.rename(columns={
        "wind_speed": "speed_obs",
        "wind_direction": "dir_obs",
        "wind_speed_max": "speed_max_obs",
        "wind_gusts": "gust_obs",
    })


def load_forecast(location, short_name, file_name):
    """Prévisions d'un modèle au lieu choisi, colonnes suffixées par son nom court (speed_gfs...)."""
    df = pd.read_csv(FORECAST_DIR / f"{location}_{file_name}.csv", parse_dates=["time"])
    return df.rename(columns={
        "wind_speed_10m": f"speed_{short_name}",
        "wind_direction_10m": f"dir_{short_name}",
        "wind_gusts_10m": f"gust_{short_name}",
    })


def build(location, obs_file):
    """Fusionne relevés et prévisions, et ajoute les indicateurs navigable / vent fort."""
    # 1. Les relevés servent de référence : on part d'eux et on y accroche les prévisions
    df = load_observations(obs_file)
    for short_name, file_name in MODELS.items():
        df = df.merge(load_forecast(location, short_name, file_name), on="time", how="left")

    # 2. Indicateurs calculés sur le vent MESURÉ (ce qui s'est vraiment passé sur l'eau)
    good_sector = df["dir_obs"].between(DIR_MIN, DIR_MAX)
    df["navigable"] = good_sector & (df["speed_obs"] >= NAVIGABLE_SPEED)
    df["strong_wind"] = df["speed_obs"] >= STRONG_WIND_SPEED
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--location", choices=LOCATIONS, default="chemoulin",
                        help="lieu des prévisions à utiliser")
    parser.add_argument("--obs", type=Path,
                        help="fichier d'observations (défaut : data/raw/observations/<lieu>.csv)")
    args = parser.parse_args()
    obs_file = args.obs or OBS_DIR / f"{args.location}.csv"

    df = build(args.location, obs_file)

    out = OUT_DIR / f"dataset_{args.location}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"-> {out} : {len(df)} lignes (observations : {obs_file})")
    print(f"Période : {df['time'].min()} -> {df['time'].max()}")
    print(f"Heures navigables : {df['navigable'].mean():.1%}")
