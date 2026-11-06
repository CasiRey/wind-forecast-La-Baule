# Prévisions de vent à La Baule : quel modèle croire ?

Évaluation des prévisions de vent passées de **GFS**, **AROME** et **ECMWF** face aux relevés
réels sur la baie de La Baule, puis correction de ces prévisions par apprentissage automatique.

## Objectif

Quand on prépare une session de voile, on compare plusieurs modèles (Windguru, Windy...)
sans vraiment savoir lequel se trompe le moins, ni dans quel sens. Ce projet répond à ces questions
avec des données :

1. Quel modèle prévoit le mieux le vent à La Baule : vitesse, direction, rafales ?
2. Les erreurs sont-elles systématiques (biais) ? Dépendent-elles de la force du vent, de sa direction, de l'heure de la journée ?
3. Peut-on corriger ces erreurs avec un modèle de machine learning (gradient boosting), mieux qu'une simple correction de biais ?

Le cœur du projet porte sur les **conditions navigables** : vent moyen d'au moins 12 nœuds
venant du secteur SE à ONO. C'est là qu'une erreur de prévision coûte une session.

## Spot et stations

| | Position | Rôle |
|---|---|---|
| **Spot** : La Baule / Pont Mahé | baie de La Baule (Loire-Atlantique) | vent navigable : secteur SE à ONO (135° à 292,5°), ≥ 12 nœuds |
| **Station Météo-France 44184001**, Pointe de Chémoulin | 47°14'01"N, 2°17'54"O, alt. 14 m | **référence** : station bien dégagée face à la mer |
| **Balise Winds-Up** (spot 7) | 47,2586 N, 2,347 O | secondaire : au plus près du spot, mais peut-être abritée |

Les prévisions sont extraites aux points de grille les plus proches de chaque station.

## Données

### Sources et périodes

**Prévisions** : [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)
(vitesse et direction du vent à 10 m, rafales), données horaires.

| Modèle | Résolution | Disponible depuis |
|---|---|---|
| GFS (`gfs_global`) | 0,11° (≈ « GFS 13 km » de Windguru) | 2021-03-23 |
| AROME France HD (`meteofrance_arome_france_hd`) | ≈ 1,5 km | 2022-11-13 |
| ECMWF IFS (`ecmwf_ifs025`) | 0,25° | 2024-02-03 |

Cette API ne garde que les premières heures de chaque run, c'est-à-dire la prévision la plus
fraîche. Les erreurs mesurées ici sont donc celles d'une prévision **à très courte échéance**.

**Relevés** :
- Station de Chémoulin : [données ouvertes Météo-France](https://meteo.data.gouv.fr), fichiers
  horaires du département 44 depuis 2021. On utilise le vent moyen sur 10 min (FF), sa direction (DD),
  le maximum du vent moyen dans l'heure (FXY) et la rafale maximale (FXI), convertis de m/s en nœuds.
  Les codes qualité de Météo-France sont conservés (colonnes `q_*`).
- Balise Winds-Up : export CSV (non public) depuis 2021, un relevé toutes les ~2 min le jour
  et toutes les ~2 h la nuit.

Tout est en **UTC** et toutes les vitesses sont en **nœuds**.

### Nettoyage de la balise Winds-Up (`src/parse_windsup.py`)

1. Heure prise dans le timestamp Unix (UTC) plutôt que dans la date texte, qui est en heure de Paris.
2. Horodatages en double retirés.
3. Relevés au-dessus de 60 nœuds retirés : ce sont des valeurs aberrantes, qui montent jusqu'à 127 nœuds.
4. Panne repérée à la main retirée (24/11 → 13/12/2024, ~19 jours à 0 nœud).
5. Séries de 0 nœud de plus de 3 h retirées : il s'agit de pannes, pas de calme plat.
6. Agrégation à l'heure : vitesse moyenne, direction en **moyenne circulaire** (sinon la moyenne de
   350° et 10° donnerait 180°), maximum et rafale max.
7. Indicateur `reliable` : vrai si l'heure contient plus de 10 relevés. La nuit, une « moyenne horaire »
   se réduit souvent à un seul relevé.

### Jeu final (`src/build_dataset.py`)

Une ligne par heure observée, avec les prévisions des 3 modèles alignées sur la même heure UTC.
On y ajoute les indicateurs `navigable` et `strong_wind` (≥ 25 nœuds), calculés sur le vent mesuré.
Fichiers : `data/processed/dataset_chemoulin.csv` et `data/processed/dataset_la_baule.csv`.

## Méthode

### Métriques (`src/metrics.py`)

Convention : **erreur = prévision − observation**. Une erreur positive veut dire que le modèle surestime le vent.

- **Vitesse** : biais (erreur moyenne), MAE (erreur absolue moyenne), RMSE (pénalise davantage les grosses erreurs).
- **Direction** : erreur **circulaire**, ramenée entre −180° et +180° (350° prévu pour 10° observé
  donne −20°, et non 340°). Elle n'est calculée que si le vent mesuré et le vent prévu dépassent
  tous deux 5 nœuds : en dessous, la direction n'a pas de sens stable.
- **Rafales** : biais.
- **Événement « heure navigable »** : tableau de contingence prévu / observé, avec
  - POD (probabilité de détection) : part des heures navigables réelles que le modèle avait prévues ;
  - FAR (taux de fausses alertes) : part des heures annoncées navigables qui ne l'ont pas été.

Chaque métrique utilise toutes les heures où le modèle et l'observation ont une valeur.
Les modèles n'ont donc pas tous la même période.

### Filtre navigable : observé **OU** prévu

Une heure entre dans le sous-ensemble « navigables » si elle est navigable **dans l'observation OU
dans la prévision** du modèle. C'est la même règle pour les « vents forts » (≥ 25 nœuds).

Pourquoi : si l'on ne garde que les heures où le vent *mesuré* était fort, on sélectionne justement
les cas où le vent a dépassé ce qui était prévu. Tous les modèles paraissent alors trop faibles,
par un simple effet de régression vers la moyenne. Filtrer sur la prévision seule produit l'effet inverse.
L'union des deux ne favorise aucun sens d'erreur.

### Heures communes

Avec l'option `--common`, on ne garde que les heures où les 3 modèles ont une prévision de vitesse
et de direction, c'est-à-dire à partir de février 2024 (début d'ECMWF). Les modèles sont alors
comparés **sur exactement les mêmes situations météo**. Les lignes ECMWF ne changent pas,
puisque c'est sa période qui définit les heures communes.

### Comparaison des deux stations (`src/compare_stations.py`)

On garde les heures présentes dans les deux jeux où la balise est fiable (`reliable`).
On calcule le biais (La Baule − Chémoulin) et la corrélation des vitesses, par secteur de 45°
de la direction mesurée à Chémoulin.

## Résultats

Tous les chiffres viennent des fichiers de [`results/`](results/). Vitesses en nœuds.

### Station de référence : Chémoulin

Toutes les heures disponibles (`metriques_erreur_chemoulin.csv`) :

| Sous-ensemble | Modèle | Heures | Biais | MAE | RMSE | MAE direction | Biais rafales |
|---|---|---|---|---|---|---|---|
| toutes | GFS | 48 442 | +0,60 | 3,53 | 4,42 | 16,1° | −2,07 |
| toutes | AROME | 34 631 | −0,67 | **2,95** | **3,80** | **12,9°** | −0,83 |
| toutes | ECMWF | 24 012 | −2,69 | 4,23 | 5,67 | 17,8° | −2,82 |
| navigables | GFS | 14 796 | −2,95 | **4,15** | **5,05** | 13,1° | −4,28 |
| navigables | AROME | 10 661 | −3,98 | 4,42 | 5,30 | **11,2°** | −2,66 |
| navigables | ECMWF | 6 661 | −8,25 | 8,28 | 9,23 | 16,1° | −5,79 |
| vents forts | GFS | 2 649 | −5,96 | **6,99** | **7,87** | 10,9° | −7,75 |
| vents forts | AROME | 1 911 | −7,12 | 7,42 | 8,18 | **9,8°** | −5,92 |
| vents forts | ECMWF | 967 | −14,09 | 14,09 | 14,68 | 14,8° | −11,04 |

Ce qu'on en retient :
- **Sur l'ensemble des heures, AROME est le plus précis**, en vitesse comme en direction.
- **En conditions navigables, tous les modèles sous-estiment le vent**, d'environ 3 nœuds pour GFS
  et 4 pour AROME. Plus le vent est fort, plus la sous-estimation grandit : environ −6 à −7 nœuds
  au-dessus de 25 nœuds pour GFS et AROME.
- **ECMWF sous-estime nettement** : −8,3 nœuds en conditions navigables, −14,1 en vent fort.
  Cela peut venir de sa maille de 0,25°, dont le point de grille mélange probablement terre et mer.
  C'est une hypothèse, pas encore vérifiée.
- Sur les mêmes heures (`metriques_erreur_chemoulin_heures_communes.csv`), le classement reste
  le même. En conditions navigables : biais GFS −2,70, AROME −3,77, ECMWF −8,25 ;
  MAE GFS 3,97, AROME 4,22, ECMWF 8,28.

Détection des heures navigables (`contingence_navigable_chemoulin.csv`) :

| Modèle | Succès | Manqués | Fausses alertes | POD | FAR |
|---|---|---|---|---|---|
| GFS | 9 550 | 3 887 | 1 345 | **0,71** | 0,12 |
| AROME | 6 968 | 3 171 | 504 | 0,69 | 0,07 |
| ECMWF | 2 209 | 4 344 | 96 | 0,34 | **0,04** |

GFS détecte le plus d'heures navigables. AROME, avec presque autant de détections, donne
deux fois moins de fausses alertes. ECMWF se trompe rarement quand il annonce du vent,
mais il manque deux heures navigables sur trois.

### Balise Winds-Up de La Baule

Toutes les heures disponibles (`metriques_erreur_la_baule.csv`) :

| Sous-ensemble | Modèle | Heures | Biais | MAE | RMSE | MAE direction |
|---|---|---|---|---|---|---|
| toutes | GFS | 42 231 | +1,49 | 2,95 | 3,81 | 21,7° |
| toutes | AROME | 29 332 | −0,09 | **2,54** | **3,31** | 20,0° |
| toutes | ECMWF | 19 970 | −1,85 | 3,03 | 3,98 | **19,3°** |
| navigables | GFS | 11 052 | +1,40 | **3,35** | 4,47 | 20,9° |
| navigables | AROME | 7 327 | −0,61 | 3,36 | **4,35** | 20,0° |
| navigables | ECMWF | 3 904 | −4,77 | 5,33 | 6,37 | **17,3°** |

Détection des heures navigables (`contingence_navigable_la_baule.csv`) :
GFS POD 0,84 / FAR 0,28 ; AROME POD 0,78 / FAR 0,21 ; ECMWF POD 0,49 / FAR 0,12.

Les biais sont plus élevés qu'à Chémoulin parce que la balise mesure un vent plus faible
(voir ci-dessous). Les erreurs de direction (~20°) sont gonflées par la balise,
qui ne donne la direction que par secteurs de 45°.

### Écart entre la balise et la station exposée

`ecart_la_baule_chemoulin_par_secteur.csv` : 38 356 heures communes, balise fiable.
Secteur selon la direction mesurée à Chémoulin.

| Secteur | Heures | Chémoulin | La Baule | Biais (La Baule − Chémoulin) | Corrélation |
|---|---|---|---|---|---|
| N | 1 803 | 3,05 | 6,15 | +3,10 | 0,40 |
| NE | 5 595 | 4,92 | 9,02 | +4,10 | 0,39 |
| E | 7 463 | 6,79 | 7,62 | +0,83 | 0,52 |
| SE | 1 541 | 10,78 | 8,44 | −2,34 | 0,83 |
| S | 1 942 | 11,51 | 9,32 | −2,18 | 0,81 |
| SO | 3 771 | 14,60 | 12,17 | −2,43 | 0,87 |
| O | 12 377 | 15,54 | 11,47 | **−4,07** | 0,89 |
| NO | 3 864 | 7,58 | 7,08 | −0,50 | 0,64 |
| **Tous** | 38 356 | 10,41 | 9,51 | −0,90 | 0,76 |

- **Secteurs navigables (SE à O)** : les deux stations varient ensemble (corrélation de 0,81 à 0,89),
  mais la balise mesure **2 à 4 nœuds de moins**. Elle est probablement plus abritée ou plus basse.
  Comme l'écart est régulier, il peut être corrigé.
- **Vent de terre (N, NE)** : c'est Chémoulin qui mesure moins, et le lien entre les deux stations
  est faible (corrélation ≈ 0,4). Chémoulin est abritée dans ces directions.

### Erreurs selon l'heure de la journée et journées thermiques

À compléter.

### Graphiques

À compléter (enregistrés dans [`grafics/`](grafics/)).

## Correction par ML

À compléter.

Méthode prévue : un gradient boosting qui prédit le vent observé à partir des prévisions
(vitesse, direction, rafales des 3 modèles, heure, mois), comparé à une base simple de correction de biais.
Le découpage sera **temporel** : apprentissage sur les périodes anciennes, test sur les plus récentes,
sans jamais de mélange aléatoire, qui laisserait le modèle « voir le futur ».

## Limites

- **Échéance très courte** : l'API Historical Forecast ne garde que les premières heures de chaque run.
  Une prévision consultée la veille (J-1) ou plusieurs jours avant se trompe probablement davantage.
  L'API « Previous Runs » d'Open-Meteo (J-1 à J-7, depuis 2024, sans rafales) permettra de le mesurer :
  à compléter.
- **Périodes différentes selon les modèles** : GFS depuis 2021, AROME fin 2022, ECMWF 2024.
  Les comparaisons sur les heures communes corrigent ce problème, au prix d'une période plus courte.
- **Balise Winds-Up** : probablement abritée (vent sous-estimé par secteur navigable), direction
  donnée par secteurs de 45° seulement, relevés espacés la nuit. Les métriques
  `metriques_erreur_la_baule*.csv` ne filtrent pas encore les heures non fiables (`reliable`).
- **Point de prévision pour La Baule** : les coordonnées demandées à Open-Meteo sont une position
  approximative de la plage, pas exactement celle de la balise.
- **Chémoulin** : abritée par vent de terre (N, NE). Cela ne gêne pas l'analyse du secteur navigable.
- **Vents forts** : peu d'heures (moins de 1 000 pour ECMWF), donc des résultats moins robustes.

## Reproduire

Prérequis : Python 3 avec `pandas` et `requests` (`requirements.txt`).

```bash
# Environnement
uv venv
uv pip install -r requirements.txt

# 1. Téléchargements (longs) -> data/raw/, non versionné
python fetch_forecasts.py        # prévisions Open-Meteo, 3 modèles x 2 lieux
python fetch_observations.py     # relevés Météo-France de Chémoulin

# 2. Relevés Winds-Up : placer l'export CSV dans data/raw/observations/, puis
python src/parse_windsup.py --start 2021-01-01

# 3. Jeux de données alignés heure par heure -> data/processed/
python src/build_dataset.py
python src/build_dataset.py --location la_baule --obs data/raw/observations/windsup.csv

# 4. Résultats -> results/
python src/metrics.py --location chemoulin [--common]
python src/metrics.py --location la_baule [--common]
python src/compare_stations.py
```

L'export Winds-Up n'est pas public et n'est pas fourni : sans lui, seule la partie Chémoulin
est reproductible. Les jeux préparés sont tout de même versionnés dans `data/processed/`.

## Usage de l'IA

J'ai développé ce projet avec l'aide de l'assistant de code Claude Code.
Je fixe les objectifs et les choix de méthode (station de référence, définition des conditions navigables, filtre observé OU prévu, découpage temporel), je relis et lance le code, j'interprète les résultats et je fais moi-même les commits. 

Les connaissances terrain viennent de ma pratique de la voile sur ce spot : secteurs navigables, seuil de 12 nœuds, doutes sur l'exposition de la balise.
