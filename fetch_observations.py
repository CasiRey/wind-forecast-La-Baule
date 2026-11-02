"""
Télécharge les relevés de vent horaires de la station Météo-France
44184001 Pointe de Chémoulin (données ouvertes, meteo.data.gouv.fr).

Météo-France publie un fichier par département et par période, avec toutes
les stations du département : on les télécharge, puis on ne garde que Chémoulin.

Résultat : data/raw/observations/chemoulin.csv, une ligne par heure (UTC),
vitesses en nœuds.

Usage :  python fetch_observations.py
"""
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR/"
# Fichiers du département 44 couvrant la période des prévisions (depuis 2021)
FILES = ["H_44_previous-2020-2024.csv.gz", "H_44_latest-2025-2026.csv.gz"]

STATION = "44184001"   # Pointe de Chémoulin, alt. 14 m
START = "2021-01-01"
MS_TO_KN = 1.943844    # 1 m/s = 1,943844 nœud

# Colonnes utiles (voir H_descriptif_champs.csv sur meteo.data.gouv.fr) :
#   FF  : vent moyen sur 10 min, à 10 m (m/s)    DD  : sa direction (degrés)
#   FXY : max du vent moyen 10 min dans l'heure  FXI : rafale max instantanée dans l'heure
#   Q.. : code qualité (1 = validée, 9 = pas encore validée, 2 = douteuse)
COLUMNS = ["NUM_POSTE", "AAAAMMJJHH", "FF", "QFF", "DD", "QDD", "FXY", "QFXY", "FXI", "QFXI"]

RAW_DIR = Path("data/raw/meteofrance")
OUT = Path("data/raw/observations/chemoulin.csv")


def download(name):
    """Télécharge un fichier départemental, sauf s'il est déjà sur le disque."""
    path = RAW_DIR / name
    if path.exists():
        print(f"Déjà téléchargé : {path}")
        return path
    print(f"Téléchargement de {name}...")
    # On écrit d'abord dans un fichier .part, renommé seulement une fois complet :
    # un téléchargement interrompu ne sera donc jamais pris pour un fichier valide.
    tmp = path.with_suffix(".part")
    with requests.get(BASE_URL + name, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    tmp.rename(path)
    return path


def read_station(path):
    """Lit un fichier départemental et ne garde que les lignes de la station."""
    df = pd.read_csv(path, sep=";", usecols=COLUMNS, dtype={"NUM_POSTE": str, "AAAAMMJJHH": str})
    return df[df["NUM_POSTE"] == STATION]


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.concat(read_station(download(name)) for name in FILES)

    # Date "2025010100" -> horodatage UTC (Météo-France fournit les données horaires en UTC)
    df["time"] = pd.to_datetime(df["AAAAMMJJHH"], format="%Y%m%d%H", utc=True)
    df = df[df["time"] >= START].drop_duplicates("time").sort_values("time")

    # Conversion en nœuds pour être comparable aux prévisions
    for col in ["FF", "FXY", "FXI"]:
        df[col] = (df[col] * MS_TO_KN).round(1)

    df = df.rename(columns={
        "FF": "wind_speed", "DD": "wind_direction", "FXY": "wind_speed_max", "FXI": "wind_gusts",
        "QFF": "q_speed", "QDD": "q_direction", "QFXY": "q_speed_max", "QFXI": "q_gusts",
    })
    df = df[["time", "wind_speed", "wind_direction", "wind_speed_max", "wind_gusts",
             "q_speed", "q_direction", "q_speed_max", "q_gusts"]]
    df.to_csv(OUT, index=False)

    # Petit contrôle : nombre d'heures attendues vs présentes
    expected = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    print(f"-> {OUT} : {len(df)} heures sur {len(expected)} attendues "
          f"({df['time'].min()} -> {df['time'].max()})")
    print("Valeurs manquantes :", df.isna().mean().round(3).to_dict())
    print("Codes qualité (vitesse) :", df["q_speed"].value_counts().to_dict())
