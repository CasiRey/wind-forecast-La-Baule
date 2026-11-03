"""
Télécharge les prévisions de vent passées (GFS, AROME HD, ECMWF) depuis
l'API "Historical Forecast" d'Open-Meteo, pour La Baule et la Pointe de Chémoulin.

Résultat : un fichier CSV par (lieu, modèle) dans data/raw/forecasts/,
avec une ligne par heure (UTC) : vitesse moyenne, direction, rafales, en nœuds.

Usage :  pip install requests pandas
         python fetch_forecasts.py
"""
from pathlib import Path
import time

import pandas as pd
import requests

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Lieux d'intérêt : la balise de La Baule (position approximative de la plage,
# à ajuster avec les coordonnées exactes de la balise) et la station Météo-France.
SITES = {
    "la_baule": (47.270, -2.370),
    "chemoulin": (47.2336, -2.2983),  # station Météo-France 44184001
}

# Nom du modèle côté Open-Meteo -> date de début de l'archive.
# (identifiants à vérifier dans la doc si l'API renvoie une erreur)
MODELS = {
    "gfs_global": "2021-03-23",                   # GFS 0.11° ≈ "GFS 13 km" de Windguru
    "meteofrance_arome_france_hd": "2022-11-13",  # AROME HD ≈ 1,5 km
    "ecmwf_ifs025": "2024-02-03",                 # ECMWF IFS 0.25°
}

END_DATE = "2026-10-31"
VARIABLES = ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
OUT_DIR = Path("data/raw/forecasts")


def fetch_one(lat, lon, model, start, end):
    """Appelle l'API pour une période et renvoie un DataFrame horaire."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "models": model,
        "wind_speed_unit": "kn",   # nœuds, comme sur Windguru
        "timezone": "UTC",         # on reste en UTC, la balise sera convertie aussi
    }
    # Le serveur gratuit est parfois surchargé (erreur 503), limite le débit (429)
    # ou ne répond pas à temps : on réessaie en attendant de plus en plus longtemps
    # (10 s, 20 s, 40 s...).
    for attempt in range(6):
        try:
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            problem = f"erreur {r.status_code}"
        except (requests.Timeout, requests.ConnectionError):
            problem = "pas de réponse"
        wait = 10 * 2**attempt
        print(f"    serveur indisponible ({problem}), nouvel essai dans {wait} s")
        time.sleep(wait)
    else:  # les 6 essais ont échoué
        raise RuntimeError(f"Open-Meteo indisponible ({problem}) pour {model} {start} -> {end}")
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def fetch_model(site, lat, lon, model, start):
    """Découpe la période en tranches d'un an pour ne pas surcharger l'API."""
    chunks = []
    for year_start in pd.date_range(start, END_DATE, freq="YS").union([pd.Timestamp(start)]):
        year_end = min(pd.Timestamp(year_start.year, 12, 31), pd.Timestamp(END_DATE))
        print(f"  {site} / {model} : {year_start.date()} -> {year_end.date()}")
        chunks.append(fetch_one(lat, lon, model, str(year_start.date()), str(year_end.date())))
        time.sleep(1)  # politesse envers l'API gratuite
    df = pd.concat(chunks).drop_duplicates("time").sort_values("time")
    # On retire le suffixe éventuel du modèle dans les noms de colonnes
    df.columns = [c.replace(f"_{model}", "") for c in df.columns]
    return df


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for site, (lat, lon) in SITES.items():
        for model, start in MODELS.items():
            out = OUT_DIR / f"{site}_{model}.csv"
            if out.exists():
                print(f"Déjà téléchargé : {out}")
                continue
            df = fetch_model(site, lat, lon, model, start)
            df.to_csv(out, index=False)
            print(f"-> {out} ({len(df)} heures, {df.isna().mean().round(3).to_dict()} de valeurs manquantes)")
