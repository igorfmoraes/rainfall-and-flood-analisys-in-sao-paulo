# Rainfall and flood-related disaster records in the São Paulo Metropolitan Region (RMSP)

Code for the final project of the Machine Learning course (UFABC, 3rd quadrimester 2026).

**Question.** Using daily rainfall history alone, how well can we anticipate which days of the RMSP (39 municipalities) have at least one official record of a flash flood (*enxurrada*), river flood (*inundação*) or urban flooding (*alagamento*)?

**Approach.** Binary classification of RMSP-days (≈1.4–2% positives), validated with leave-one-year-out cross-validation. Three models are compared: a one-variable baseline, logistic regression (own NumPy gradient-descent implementation, checked against scikit-learn) and gradient boosting. Metric: PR-AUC, with year-block bootstrap confidence intervals. A municipal descriptive analysis complements the models.

> The target is *registered* events (Atlas of Disasters), not physical flooding. Registrations depend on municipal reporting and changed regime around 2013/2014; see the article for limitations.

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── run_all.sh                  # full pipeline, step by step
├── src/
│   ├── rmsp.py                 # the 39 RMSP municipalities (IBGE 7-digit codes)
│   ├── 01_extrair_chirps.py    # daily CHIRPS rainfall per municipality (Google Earth Engine)
│   ├── 02_montar_dataset.py    # builds the RMSP-day dataset (features + label)
│   ├── 03_modelagem.py         # models, leave-one-year-out validation, bootstrap, importance
│   └── 04_descritiva_municipio.py  # per-municipality table and figures
├── data/
│   ├── raw/                    # Atlas export (xlsx) and IBGE municipal mesh (shapefile)
│   ├── interim/                # CHIRPS daily series (per year and combined)
│   └── processed/              # model-ready datasets
└── results/
    ├── <run>/                  # <run>_metricas.csv, <run>_importancia.csv, <run>_pr.png
    └── descriptive/            # municipios_resumo.csv and figures
```

## Data sources

| Data | Source | Notes |
| --- | --- | --- |
| Disaster records | [Atlas Digital de Desastres no Brasil](https://atlasdigital.mdr.gov.br) (MIDR / S2iD) | Export of the sheet "Atlas Valores Corrigidos", state of São Paulo, records up to 2025-12-31. Not redistributed here (75 MB); download and save as `data/raw/Atlas_Apenas_SP.xlsx`. |
| Rainfall | CHIRPS v2.0 daily (`UCSB-CHG/CHIRPS/DAILY`), via Google Earth Engine | Funk et al. (2015). ~5 km grid. Requires an Earth Engine-enabled Google Cloud project. |
| Municipal mesh | IBGE, *Malhas territoriais*, SP municipalities 2025 | Place in `data/raw/SP_Municipios_2025/`. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
earthengine authenticate          # only needed for step 1
export GEE_PROJECT=<your-google-cloud-project>
```

Tested with Python 3.14.10, pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2, openpyxl 3.1.5, geopandas 1.2.0, earthengine-api 1.7.53 (pinned in `requirements.txt`).

## Reproducing the analysis

```bash
export GEE_PROJECT=<your-google-cloud-project>   # only needed for step 1
bash run_all.sh
```

If `data/interim/chirps_rmsp_diario.csv` already exists and `GEE_PROJECT` is unset, step 1 is skipped.

Or step by step:

```bash
# 1. CHIRPS extraction (slow; needs network and Earth Engine access)
python src/01_extrair_chirps.py --projeto $GEE_PROJECT \
    --shapefile data/raw/SP_Municipios_2025/SP_Municipios_2025.shp \
    --saida data/interim/chirps_rmsp_diario.csv --pasta-anos data/interim/chirps_anos

# 2. Dataset (main label: Enxurradas, Inundações, Alagamentos)
python src/02_montar_dataset.py --atlas data/raw/Atlas_Apenas_SP.xlsx \
    --chuva data/interim/chirps_rmsp_diario.csv --saida data/processed/dataset_rmsp.csv

# 3. Main analysis: leave-one-year-out, window 1991-2013
mkdir -p results/main
python src/03_modelagem.py --dataset data/processed/dataset_rmsp.csv --prefixo results/main/main

# 4. Municipal descriptive analysis
python src/04_descritiva_municipio.py --atlas data/raw/Atlas_Apenas_SP.xlsx \
    --chuva data/interim/chirps_rmsp_diario.csv --dataset data/processed/dataset_rmsp.csv \
    --saida results/descriptive
```

Variants of step 3 (sensitivity analyses):

| Run folder | Flags / dataset |
| --- | --- |
| `no_lag` | `--sem-lag` (drops previous-day rainfall features) |
| `no_calendar` | `--excluir doy` (drops seasonality features) |
| `extended_label_1991_2013`, `extended_label_1991_2025` | dataset built with `--tipos "Enxurradas,Inundações,Alagamentos,Chuvas Intensas"`; `--ano-max 2013` / `2025`. Defined *after* seeing the main results (exploratory). |
| `short_windows_only` | `--excluir 7d,15d,30d` (keeps only 1-, 2- and 3-day windows and calendar features) |
| `full_1991_2025` | `--ano-max 2025` on the main dataset (shows the effect of the post-2013 drop in records) |

Determinism: the bootstrap and the boosting model use a fixed seed (`--semente 0`). The extraction in step 1 depends on the CHIRPS collection as served by Earth Engine at run time.

## Script options

File names and internal identifiers are in Portuguese (the course language); figures, console output and result files are in English. Every option has an English alias: `python src/<script>.py --help` lists them. Main ones in `03_modelagem.py`: `--ano-min/--year-min`, `--ano-max/--year-max` (window, default 1991-2013), `--sem-lag/--no-lag`, `--excluir/--exclude`, `--l2`, `--repeticoes/--repetitions` (bootstrap), `--semente/--seed`, `--prefixo/--prefix` (output prefix).

Feature names in the datasets: `media_Nd` / `max_Nd` = mean / maximum across the 39 municipalities of the rainfall total over N days; `*_dia_anterior` = previous day; `doy_sin`, `doy_cos` = day-of-year seasonality.

## Outputs

- `*_metricas.csv`: model, PR-AUC, ROC-AUC, Precision@k, 95% bootstrap interval of the PR-AUC, 95% interval of the difference from the baseline.
- `*_importancia.csv`: permutation importance (boosting) and standardized coefficients (logistic regression).
- `*_pr.png`: out-of-fold precision-recall curves.
- `results/descriptive/`: `municipios_resumo.csv` (records, record days, rainfall exposure per municipality), `records_by_month.csv` (records per month, 1991-2025) and five figures.

## AI use

An AI assistant (Claude, Anthropic) was used for scoping, data-source research, writing and debugging the scripts, and drafting text. The author ran all code and is responsible for the results. The full statement is in the article.

## License

GNU Affero General Public License v3.0 (see `LICENSE`).

## References

Funk, C. et al. (2015). The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066.
