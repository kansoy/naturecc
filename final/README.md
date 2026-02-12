# Replication Package: Do Central Banks Take Climate Change Seriously?

This folder is a self-contained replication package for the submitted manuscript and is prepared for code/data checks requested by Nature Climate Change.

## Contents

- `data/`: input datasets used by the replication scripts.
- `src/analysis.py`: computes core analysis numbers.
- `src/tables.py`: generates manuscript tables.
- `src/figures.py`: generates manuscript figures.
- `run_all.py`: runs analysis, tables, and figures end-to-end.

## Requirements

- Python 3.14 (tested).
- Packages listed in `requirements.txt`.
- Tested on: macOS (Apple Silicon), Python 3.14.
- Non-standard hardware: none.
- Git LFS (required to fetch large raw CSV files from GitHub).

## Run

From this `final` folder:

```bash
git lfs install
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Typical install time on a normal desktop: ~2-5 minutes (including environment creation and package installation).

Typical runtime for `python run_all.py`: ~10-60 seconds.

## Generated outputs

- `outputs/analysis/main_numbers.json`
- `outputs/analysis/within_institution_rates.csv`
- `outputs/tables/table1_overview.csv`
- `outputs/tables/table2_institution_heterogeneity.csv`
- `outputs/figures/fig1_temporal_trends.pdf`
- `outputs/figures/fig11_temporal_commitment.pdf`
- `outputs/figures/fig6_lagarde_effect.pdf`
- `outputs/figures/fig_commitment_grouped.pdf`
- `outputs/figures/fig_first_mention_lag.pdf`

## Use on your data

Place replacement input files in the same relative locations under `data/` and keep the same column schema, then run `python run_all.py`.
