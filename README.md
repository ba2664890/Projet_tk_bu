# EDA-Desk Pro Modern (Tkinter)

Application desktop Python pour l'Exploratory Data Analysis (EDA), avec interface moderne `CustomTkinter`, analyses statistiques, visualisations, filtres avances, import multi-sources, et generation de rapport PDF avec interpretation IA optionnelle via Hugging Face.

## Fonctionnalites

- Import de donnees:
  - Fichiers: `CSV`, `Excel`, `JSON`, `Parquet`, `Stata`, `SAS`, `SPSS`
  - API REST (JSON)
  - SQL via connection string + requete
- Parametrage CSV avance (separateur, encodage, decimal, NA custom, header)
- Diagnostics qualite:
  - Valeurs manquantes
  - Variables problematiques (manquants, quasi constantes, cardinalite)
  - Outliers (IQR)
  - Doublons
- Statistiques descriptives:
  - Numeriques (moyenne, mediane, std, skewness, kurtosis, normalite)
  - Categorielles
  - Resume global des variables
- Analyses multivariees:
  - ANOVA
  - Test t
  - Correlations (Pearson/Spearman/Kendall selon les modules utilises)
  - Chi2
  - Regression lineaire
  - ACP (PCA)
- Visualisations:
  - Histogramme, boxplot, scatter, heatmap
  - Bar, pie, pairplot, distribution
- Filtres avances:
  - Recherche texte globale
  - Filtre numerique min/max
  - Filtre categoriel multi-valeurs
  - Option exclusion des NA apres filtrage
- Rapport PDF institutionnel (ReportLab), avec:
  - tableaux + graphiques automatiques
  - recommandations
  - interpretation IA Hugging Face (optionnelle)

## Structure du projet

```text
.
├── eda_desk_modern.py        # Point d'entree
├── eda_modern/
│   ├── __init__.py
│   ├── app.py                # UI + logique metier
│   ├── data_sources.py       # Chargement fichiers/API + optimisation memoire
│   ├── analysis.py           # ANOVA, regression, PCA, correlations
│   ├── ai_report.py          # Integration Hugging Face
│   ├── widgets.py            # Composants UI reutilisables
│   └── theme.py              # Palette couleurs
└── README.md
```

## Pre-requis

- Python `3.10+` recommande
- Environnement graphique actif (Tkinter / `DISPLAY`)
- `pip` a jour

## Installation rapide (pip)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install customtkinter pandas numpy scipy matplotlib seaborn scikit-learn requests sqlalchemy reportlab openpyxl pyarrow pyreadstat pillow opencv-python
```

Notes:
- `openpyxl`: lecture Excel
- `pyarrow`: lecture Parquet
- `pyreadstat`: lecture SPSS
- `reportlab`: export PDF
- `requests`: API REST + rapport IA
- `sqlalchemy`: source SQL
- `opencv-python` + `pillow`: lecture video de fond (optionnelle)

## Lancement

```bash
python3 eda_desk_modern.py
```

## Configuration IA (Hugging Face, optionnel)

1. Exporter votre token:

```bash
export HF_API_TOKEN="hf_xxx"
```

2. Lancer l'application puis ouvrir `Config IA`.
3. Au premier lancement, ces fichiers sont crees automatiquement a la racine du projet:
   - `hf_report_config.json`
   - `hf_report_prompt.txt`
4. Generer un rapport PDF avec `Generer Rapport`.

Si l'IA est active, l'application peut aussi exporter un fichier contexte:
- `<nom_rapport>_hf_context.json`

## Utilisation rapide

1. Choisir une source (`CSV`, `Excel`, `JSON`, etc.)
2. Charger les donnees (fichier/API/SQL)
3. Explorer les onglets: `Diagnostic`, `Statistiques`, `Multivarie`, `Visualisation`, `Filtres`
4. Cliquer `Generer Rapport` pour produire un PDF

## Depannage

- Erreur `Tkinter` / `DISPLAY`:
  - lancer depuis une session graphique locale
- Conflit `numpy/matplotlib` (ABI) possible selon l'environnement:
  - le script integre deja une strategie de fallback
  - sur Debian/Ubuntu, installer de preference les paquets systeme:
    - `sudo apt install -y python3-pandas python3-seaborn python3-sklearn`
- Erreur PDF:
  - installer `reportlab`
- Erreur IA:
  - verifier `HF_API_TOKEN`
  - verifier la connectivite reseau
  - verifier les URLs/modeles dans `hf_report_config.json`

## Limites connues

- Outil principalement concu pour un usage desktop local.
- Les performances dependent de la taille du dataset et de la memoire disponible.
- Certaines sources/formats necessitent des dependances optionnelles (voir section installation).

## Licence

Aucune licence n'est fournie actuellement dans ce depot.
