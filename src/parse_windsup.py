"""
Lecture, nettoyage et agrégation horaire des relevés de la balise Winds-Up de La Baule (spot 7).

Entrée : export CSV Winds-Up (séparateur ";" avec un ";" final, une ligne d'en-tête).
On utilise le timestamp Unix (UTC) et pas la date texte.
Relevés toutes les ~2 min le jour, ~2 h la nuit. Vitesses déjà en nœuds.

Attention : la direction n'est donnée que par secteurs de 45° (N, NE, E, ...).
La moyenne horaire peut tomber entre deux secteurs, mais la précision réelle reste de ±22,5°.

Nettoyage, dans l'ordre :
  1. horodatages en double retirés (on garde le premier) ;
  2. relevés avec une vitesse > 60 nœuds retirés (valeurs aberrantes, jusqu'à 127 nœuds) ;
  3. pannes repérées à la main retirées (KNOWN_OUTAGES) ;
  4. séries de 0 nœud durant plus de 3 h retirées : ce sont des pannes, pas du calme.

Sortie : même format que fetch_observations.py (time, wind_speed, wind_direction,
wind_speed_max, wind_gusts), pour être utilisée directement par build_dataset.py.

Usage (depuis la racine du projet) :
    python src/parse_windsup.py --start 2021-01-01
    python src/build_dataset.py --location la_baule --obs data/raw/observations/windsup.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("data/raw/observations/Station7 - 2008-1-1 - 2026-11-6 23_59_59.csv")
DEFAULT_OUTPUT = Path("data/raw/observations/windsup.csv")

MAX_SPEED = 60          # nœuds : au-delà, valeur jugée aberrante
MAX_ZERO_RUN = pd.Timedelta(hours=3)  # série de 0 plus longue = panne

# Pannes repérées à la main (UTC, début inclus, fin exclue) : la règle des 0 ne suffit pas
# car quelques relevés non nuls isolés coupent la série de 0 en morceaux.
KNOWN_OUTAGES = [
    ("2024-11-24 22:00", "2024-12-13 09:00"),  # ~19 jours de 0, avec 31 relevés non nuls parasites
]


def read_raw(path):
    """Lit les colonnes utiles par position (le ";" final crée une colonne vide en trop)."""
    df = pd.read_csv(path, sep=";", usecols=[1, 3, 5, 6], header=0,
                     names=["timestamp", "speed", "speed_max", "direction"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.drop(columns="timestamp").sort_values("time").reset_index(drop=True)


def remove_zero_runs(df):
    """Retire les séries consécutives de vent moyen nul qui durent plus de 3 h.

    Une série = relevés successifs tous à 0. Sa durée = du premier au dernier relevé à 0.
    Renvoie le tableau nettoyé, le nombre de séries retirées et la plus longue.
    """
    is_zero = df["speed"] == 0
    run_id = (is_zero != is_zero.shift()).cumsum()   # change de numéro à chaque bascule 0 / non-0
    zero_runs = df[is_zero].groupby(run_id[is_zero])["time"].agg(["min", "max"])
    duration = zero_runs["max"] - zero_runs["min"]
    bad_runs = duration[duration > MAX_ZERO_RUN].index
    longest = duration.max() if len(duration) else pd.Timedelta(0)
    return df[~run_id.isin(bad_runs)], len(bad_runs), longest


def to_hourly(df):
    """Agrège à l'heure la plus proche (relevés de hh:30 à hh+1:30 → hh+1:00)."""
    rad = np.deg2rad(df["direction"])
    df = df.assign(hour=df["time"].dt.round("h"), u=np.sin(rad), v=np.cos(rad))
    hourly = df.groupby("hour").agg(
        wind_speed=("speed", "mean"),
        wind_speed_max=("speed", "max"),
        wind_gusts=("speed_max", "max"),
        u=("u", "mean"),
        v=("v", "mean"),
    )
    # Direction : moyenne circulaire (sinon la moyenne de 350° et 10° donnerait 180°)
    hourly["wind_direction"] = np.rad2deg(np.arctan2(hourly["u"], hourly["v"])) % 360
    hourly = hourly.drop(columns=["u", "v"]).round(1)
    return hourly.rename_axis("time").reset_index()[
        ["time", "wind_speed", "wind_direction", "wind_speed_max", "wind_gusts"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", help="date de début (UTC), ex. 2021-01-01")
    args = parser.parse_args()

    df = read_raw(args.input)
    print(f"Relevés lus : {len(df)} ({df['time'].min()} à {df['time'].max()})")

    if args.start:
        df = df[df["time"] >= pd.Timestamp(args.start, tz="UTC")]
        print(f"Depuis {args.start} : {len(df)}")

    n = len(df)
    df = df.drop_duplicates(subset="time", keep="first")
    print(f"Doublons d'horodatage retirés : {n - len(df)}")

    n = len(df)
    df = df[(df["speed"] <= MAX_SPEED) & (df["speed_max"] <= MAX_SPEED)]
    print(f"Relevés > {MAX_SPEED} nœuds retirés : {n - len(df)}")

    n = len(df)
    for start, end in KNOWN_OUTAGES:
        df = df[~((df["time"] >= pd.Timestamp(start, tz="UTC")) & (df["time"] < pd.Timestamp(end, tz="UTC")))]
    print(f"Pannes connues retirées : {n - len(df)} relevés")

    n = len(df)
    df, n_runs, longest = remove_zero_runs(df)
    print(f"Pannes (0 nœud > 3 h) : {n_runs} séries, {n - len(df)} relevés retirés, "
          f"plus longue série de 0 : {longest}")

    hourly = to_hourly(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(args.output, index=False)
    print(f"Heures écrites : {len(hourly)} dans {args.output}")
    print(hourly[["wind_speed", "wind_gusts"]].describe().round(1))


if __name__ == "__main__":
    main()
