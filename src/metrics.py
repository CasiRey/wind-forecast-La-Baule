"""
Métriques d'erreur des prévisions de vent, modèle par modèle.

Convention : erreur = prévision - observation.
  -> erreur positive = le modèle surestime, négative = il sous-estime.

Chaque métrique n'utilise que les heures où le modèle ET l'observation ont une valeur :
les modèles n'ont pas la même période (AROME depuis fin 2022, ECMWF depuis 2024),
on ne les pénalise donc pas pour les heures où ils n'existaient pas encore.
Avec --common, on ne garde que les heures où les 3 modèles ont vitesse et direction :
les modèles sont alors comparés exactement sur les mêmes situations météo.

Sous-ensembles « navigables » et « vents forts » : une heure y entre si la condition est
vraie dans l'observation OU dans la prévision du modèle. Filtrer sur l'observation seule
ne garderait que les heures où le vent réel était fort, et ferait paraître tous les
modèles trop faibles (régression vers la moyenne) ; filtrer sur la prévision seule
ferait l'inverse. L'union des deux ne favorise aucun sens d'erreur.

Usage (depuis la racine du projet) :  python src/metrics.py [--location la_baule] [--common]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import DIR_MAX, DIR_MIN, NAVIGABLE_SPEED, STRONG_WIND_SPEED

MODELS = ["gfs", "arome", "ecmwf"]
RESULTS_DIR = Path("results")

# En dessous de ce vent (mesuré ou prévu), la direction est trop instable pour être comparée
MIN_DIR_SPEED = 5


def circular_diff(forecast_deg, observed_deg):
    """Écart de direction ramené entre -180° et 180° (350° vs 10° -> -20°, pas 340°)."""
    return (forecast_deg - observed_deg + 180) % 360 - 180


def is_navigable(speed, direction):
    """Vent >= 12 nœuds dans le secteur SE à ONO (135° à 292,5°)."""
    return (speed >= NAVIGABLE_SPEED) & direction.between(DIR_MIN, DIR_MAX)


def model_metrics(df, model):
    """Biais, MAE, RMSE (vitesse), erreur de direction et biais des rafales pour un modèle.

    Chaque groupe de métriques a son propre nombre d'heures, car les données
    disponibles diffèrent (direction seulement au-dessus de 5 nœuds, rafales parfois absentes).
    """
    speed = df[[f"speed_{model}", "speed_obs"]].dropna()
    err = speed[f"speed_{model}"] - speed["speed_obs"]

    direction = df[[f"dir_{model}", "dir_obs", f"speed_{model}", "speed_obs"]].dropna()
    enough_wind = ((direction["speed_obs"] >= MIN_DIR_SPEED)
                   & (direction[f"speed_{model}"] >= MIN_DIR_SPEED))
    direction = direction[enough_wind]
    dir_err = circular_diff(direction[f"dir_{model}"], direction["dir_obs"])

    gust = df[[f"gust_{model}", "gust_obs"]].dropna()
    gust_err = gust[f"gust_{model}"] - gust["gust_obs"]

    return {
        "h vitesse": len(speed),
        "biais (kn)": err.mean(),
        "MAE (kn)": err.abs().mean(),
        "RMSE (kn)": np.sqrt((err ** 2).mean()),
        "h dir": len(direction),
        "biais dir (°)": dir_err.mean(),
        "MAE dir (°)": dir_err.abs().mean(),
        "h rafales": len(gust),
        "biais rafales (kn)": gust_err.mean(),
    }


def subsets(df, model):
    """Toutes les heures, heures navigables et vents forts (observés OU prévus par le modèle)."""
    nav_obs = is_navigable(df["speed_obs"], df["dir_obs"])
    nav_model = is_navigable(df[f"speed_{model}"], df[f"dir_{model}"])
    strong = (df["speed_obs"] >= STRONG_WIND_SPEED) | (df[f"speed_{model}"] >= STRONG_WIND_SPEED)
    return {
        "toutes": df,
        "navigables": df[nav_obs | nav_model],
        "vents forts": df[strong],
    }


def metrics_by_subset(df, models=MODELS):
    """Tableau des métriques, une ligne par (sous-ensemble, modèle)."""
    rows = {}
    for model in models:
        for name, sub in subsets(df, model).items():
            rows[(name, model)] = model_metrics(sub, model)
    table = pd.DataFrame(rows).T
    table.index.names = ["sous-ensemble", "modèle"]
    return table.reindex(["toutes", "navigables", "vents forts"], level=0)


def contingency(df, model):
    """Tableau de contingence de l'événement « heure navigable », prévu vs observé.

    POD (probabilité de détection) : part des heures navigables réelles que le modèle a prévues.
    FAR (taux de fausses alertes) : part des heures prévues navigables qui ne l'ont pas été.
    """
    d = df[["speed_obs", "dir_obs", f"speed_{model}", f"dir_{model}"]].dropna()
    obs = is_navigable(d["speed_obs"], d["dir_obs"])
    fc = is_navigable(d[f"speed_{model}"], d[f"dir_{model}"])

    hits = (obs & fc).sum()
    misses = (obs & ~fc).sum()
    false_alarms = (~obs & fc).sum()
    correct_neg = (~obs & ~fc).sum()
    return {
        "succès": hits,
        "manqués": misses,
        "fausses alertes": false_alarms,
        "rejets corrects": correct_neg,
        "POD": hits / (hits + misses),
        "FAR": false_alarms / (hits + false_alarms),
    }


def contingency_table(df, models=MODELS):
    """Tableau de contingence, une ligne par modèle."""
    return pd.DataFrame({m: contingency(df, m) for m in models}).T


def common_hours(df, models=MODELS):
    """Ne garde que les heures où tous les modèles ont vitesse et direction.

    Les rafales ne sont pas exigées : ECMWF n'en fournit pas avant fin 2024, cela
    ferait perdre des mois entiers. Le biais rafales garde son propre nombre d'heures.
    """
    cols = [f"{var}_{m}" for m in models for var in ("speed", "dir")]
    return df.dropna(subset=cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="chemoulin")
    parser.add_argument("--common", action="store_true",
                        help="ne garder que les heures communes aux 3 modèles")
    args = parser.parse_args()

    df = pd.read_csv(f"data/processed/dataset_{args.location}.csv", parse_dates=["time"])
    if args.common:
        df = common_hours(df)
        print(f"Heures communes aux 3 modèles : {len(df)} "
              f"({df['time'].min():%Y-%m-%d} -> {df['time'].max():%Y-%m-%d})\n")

    table = metrics_by_subset(df)
    counts = ["h vitesse", "h dir", "h rafales"]
    table[counts] = table[counts].astype(int)
    print(table.round(2).to_string())

    print("\nÉvénement « heure navigable » (prévu vs observé) :")
    cont = contingency_table(df)
    cont = cont.astype({c: int for c in cont.columns[:4]})
    print(cont.round(2).to_string())

    # Sauvegarde des tableaux (le suffixe distingue les heures communes aux 3 modèles)
    suffix = f"{args.location}_heures_communes" if args.common else args.location
    RESULTS_DIR.mkdir(exist_ok=True)
    table.round(3).to_csv(RESULTS_DIR / f"metriques_erreur_{suffix}.csv")
    cont.round(3).to_csv(RESULTS_DIR / f"contingence_navigable_{suffix}.csv")
    print(f"\nTableaux enregistrés dans {RESULTS_DIR}/ (suffixe {suffix})")
