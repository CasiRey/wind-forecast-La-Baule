"""
Métriques d'erreur des prévisions de vent, modèle par modèle.

Convention : erreur = prévision - observation.
  -> erreur positive = le modèle surestime, négative = il sous-estime.

Chaque métrique n'utilise que les heures où le modèle ET l'observation ont une valeur :
les modèles n'ont pas la même période (AROME depuis fin 2022, ECMWF depuis 2024),
on ne les pénalise donc pas pour les heures où ils n'existaient pas encore.

Usage (depuis la racine du projet) :  python src/metrics.py [--location la_baule]
"""
import argparse

import numpy as np
import pandas as pd

MODELS = ["gfs", "arome", "ecmwf"]


def circular_diff(forecast_deg, observed_deg):
    """Écart de direction ramené entre -180° et 180° (350° vs 10° -> -20°, pas 340°)."""
    return (forecast_deg - observed_deg + 180) % 360 - 180


def model_metrics(df, model):
    """Biais, MAE, RMSE (vitesse), erreur de direction et biais des rafales pour un modèle."""
    speed = df[[f"speed_{model}", "speed_obs"]].dropna()
    err = speed[f"speed_{model}"] - speed["speed_obs"]

    direction = df[[f"dir_{model}", "dir_obs"]].dropna()
    dir_err = circular_diff(direction[f"dir_{model}"], direction["dir_obs"])

    gust = df[[f"gust_{model}", "gust_obs"]].dropna()
    gust_err = gust[f"gust_{model}"] - gust["gust_obs"]

    return {
        "heures": len(speed),
        "biais (kn)": err.mean(),
        "MAE (kn)": err.abs().mean(),
        "RMSE (kn)": np.sqrt((err ** 2).mean()),
        "biais dir (°)": dir_err.mean(),
        "MAE dir (°)": dir_err.abs().mean(),
        "biais rafales (kn)": gust_err.mean(),
    }


def metrics_table(df, models=MODELS):
    """Tableau des métriques, une ligne par modèle."""
    return pd.DataFrame({m: model_metrics(df, m) for m in models}).T


def metrics_by_subset(df, models=MODELS):
    """Métriques sur toutes les heures, les heures navigables et les vents forts."""
    subsets = {
        "toutes": df,
        "navigables": df[df["navigable"]],
        "vents forts": df[df["strong_wind"]],
    }
    return pd.concat({name: metrics_table(sub, models) for name, sub in subsets.items()},
                     names=["sous-ensemble", "modèle"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="chemoulin")
    args = parser.parse_args()

    df = pd.read_csv(f"data/processed/dataset_{args.location}.csv", parse_dates=["time"])
    table = metrics_by_subset(df)
    table["heures"] = table["heures"].astype(int)
    print(table.round(2).to_string())
