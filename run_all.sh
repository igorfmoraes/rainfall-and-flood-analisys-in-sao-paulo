#!/usr/bin/env bash
# Full pipeline. Run from the project root with the virtualenv active.
# Step 1 (CHIRPS extraction) needs GEE_PROJECT (Google Cloud project with Earth Engine enabled)
# and `earthengine authenticate`. If data/interim/chirps_rmsp_diario.csv already exists and
# GEE_PROJECT is unset, step 1 is skipped. Force it with FORCE_EXTRACT=1.
set -euo pipefail
ATLAS=data/raw/Atlas_Apenas_SP.xlsx
SHP=data/raw/SP_Municipios_2025/SP_Municipios_2025.shp
CHIRPS=data/interim/chirps_rmsp_diario.csv

# 1. Daily CHIRPS rainfall per municipality (Google Earth Engine) -- slow, needs network
if [[ -n "${GEE_PROJECT:-}" && ( ! -f "$CHIRPS" || "${FORCE_EXTRACT:-0}" == "1" ) ]]; then
    mkdir -p data/interim
    python src/01_extrair_chirps.py --projeto "$GEE_PROJECT" --shapefile "$SHP" \
        --saida "$CHIRPS" --pasta-anos data/interim/chirps_anos
elif [[ -f "$CHIRPS" ]]; then
    echo "Step 1 skipped: $CHIRPS already exists (set GEE_PROJECT and FORCE_EXTRACT=1 to redo)."
else
    echo "ERROR: $CHIRPS not found and GEE_PROJECT is not set. Run: export GEE_PROJECT=<project>" >&2
    exit 1
fi

# 2. Daily RMSP dataset: main label and extended label
mkdir -p data/processed
python src/02_montar_dataset.py --atlas "$ATLAS" --chuva "$CHIRPS" --saida data/processed/dataset_rmsp.csv
python src/02_montar_dataset.py --atlas "$ATLAS" --chuva "$CHIRPS" \
    --tipos "Enxurradas,Inundações,Alagamentos,Chuvas Intensas" --saida data/processed/dataset_rmsp_extended.csv

# 3. Models (leave-one-year-out). Prefix = results/<run>/<run>
run() { name=$1; shift; mkdir -p "results/$name"; python src/03_modelagem.py --prefixo "results/$name/$name" "$@"; }
run main        --dataset data/processed/dataset_rmsp.csv
run no_lag      --dataset data/processed/dataset_rmsp.csv --sem-lag
run no_calendar --dataset data/processed/dataset_rmsp.csv --excluir doy
run short_windows_only --dataset data/processed/dataset_rmsp.csv --excluir 7d,15d,30d
run full_1991_2025      --dataset data/processed/dataset_rmsp.csv --ano-max 2025
run extended_label_1991_2013 --dataset data/processed/dataset_rmsp_extended.csv --ano-max 2013
run extended_label_1991_2025 --dataset data/processed/dataset_rmsp_extended.csv --ano-max 2025

# 4. Municipal descriptive analysis
python src/04_descritiva_municipio.py --atlas "$ATLAS" --chuva "$CHIRPS" \
    --dataset data/processed/dataset_rmsp.csv --saida results/descriptive
