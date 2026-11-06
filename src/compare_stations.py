"""
Compare la balise Winds-Up de La Baule à la station Météo-France de Chémoulin,
sur les heures communes où la balise est fiable (> 10 relevés dans l'heure).

Pour chaque secteur de 45° (direction mesurée à Chémoulin, station dégagée) :
biais de vitesse La Baule - Chémoulin et corrélation entre les deux vitesses.

Usage (depuis la racine du projet) :
    python src/compare_stations.py
"""
import pandas as pd

SECTORS = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]


def load_common_hours():
    """Heures présentes dans les deux jeux, balise fiable, vitesses et direction connues."""
    cols = ["time", "speed_obs", "dir_obs"]
    chem = pd.read_csv("data/processed/dataset_chemoulin.csv", usecols=cols, parse_dates=["time"])
    baule = pd.read_csv("data/processed/dataset_la_baule.csv", usecols=cols + ["reliable"],
                        parse_dates=["time"])
    baule = baule[baule["reliable"]].drop(columns="reliable")
    df = chem.merge(baule, on="time", suffixes=("_chem", "_baule"))   # inner : heures communes
    return df.dropna()


def sector(direction):
    """Secteur de 45° centré sur N, NE... (N = 337,5° à 22,5°)."""
    return pd.Categorical([SECTORS[i] for i in ((direction + 22.5) // 45 % 8).astype(int)],
                          categories=SECTORS)


def summarize(group):
    diff = group["speed_obs_baule"] - group["speed_obs_chem"]
    return pd.Series({
        "heures": len(group),
        "Chémoulin moy": group["speed_obs_chem"].mean(),
        "La Baule moy": group["speed_obs_baule"].mean(),
        "biais": diff.mean(),
        "corrélation": group["speed_obs_baule"].corr(group["speed_obs_chem"]),
    })


if __name__ == "__main__":
    df = load_common_hours()
    print(f"Heures communes (balise fiable) : {len(df)} "
          f"({df['time'].min():%Y-%m-%d} -> {df['time'].max():%Y-%m-%d})\n")

    df["secteur"] = sector(df["dir_obs_chem"])
    table = df.groupby("secteur", observed=True).apply(summarize, include_groups=False)
    table.loc["Tous"] = summarize(df)
    table["heures"] = table["heures"].astype(int)
    print("Vitesses en nœuds, biais = La Baule - Chémoulin, secteur = direction à Chémoulin")
    print(table.round(2).to_string())
