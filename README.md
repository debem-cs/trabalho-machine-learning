# Change detection in satellite imagery

Multi-class classification of land-use change from multi-temporal satellite
observations of polygons. Project for the Machine Learning course at
CentraleSupélec (SG6, 2025-2026).

**Team "3 Mosqueteiros":** Enzo Jardim Vendramin, Matheus Carvalho da Costa,
Pedro de Bem e Canto Cantanhede.

## Problem

Each sample is a polygon observed on five dates. The data includes image
statistics per date (mean and standard deviation of the spectral bands), a
change status per date, the polygon geometry, and urban and geographic context
tags. The task is to predict the type of change:

| Label | Class |
|---:|---|
| 0 | Demolition |
| 1 | Road |
| 2 | Residential |
| 3 | Commercial |
| 4 | Industrial |
| 5 | Mega Projects |

The main difficulty is telling real changes (a new building, a new road) apart
from natural variation due to lighting, season and vegetation. The classes are
very imbalanced: *Residential* and *Commercial* make up more than 84 % of the
data. The evaluation metric is the weighted F1-score.

## Approach

**Feature engineering** ([`src/features.py`](src/features.py)):

- geometry: area, perimeter, compactness (Polsby–Popper);
- dates: durations between consecutive observations, and day of year encoded
  as sine/cosine to keep seasonality cyclical;
- multi-label one-hot encoding of the urban and geographic tags, plus encoding
  of the per-date change status;
- temporal evolution of each band: absolute and relative differences between
  consecutive dates, plus amplitude (max − min) and standard deviation over
  time; overall image mean and standard deviation across bands.

**Models** ([`src/train.py`](src/train.py), [`src/submit.py`](src/submit.py)):

- Random Forest and XGBoost were compared with cross-validation on macro,
  weighted and micro F1. XGBoost hyper-parameters were tuned by randomised
  search optimising weighted F1.
- The final model is a soft-voting ensemble (50/50 average of class
  probabilities) of a standard Random Forest and XGBoost. Class balancing was
  not used, because it lowered weighted F1.
- Pseudo-labelling: test predictions with confidence above 0.95 are added to the
  training set and both models are retrained before the final prediction.

The ensemble reached a cross-validated weighted F1 of about 0.78.

## Repository structure

```text
src/
├── features.py         # data loading (GeoJSON) and feature engineering
├── train.py            # model comparison with CV, optional XGBoost tuning
├── submit.py           # train final ensemble + pseudo-labelling, write submission
├── inference.py        # predict with a saved model.joblib
└── plot_importance.py  # XGBoost feature importance plot
notebooks/Exploration.py  # data exploration (VS Code / Jupytext cell format)
dados/                    # course skeleton code and sample submission
Latex/report.tex          # project report
submission.csv            # final predictions
```

## Running

The data files `train.geojson` and `test.geojson` are not versioned. Put them
in `dados/`, then run from the repository root:

```bash
pip install -r requirements.txt

python src/train.py --sample 5000   # quick model comparison on a subset
python src/train.py --tune          # full comparison + XGBoost random search
python src/submit.py                # final model -> submission.csv and model.joblib
python src/inference.py             # predictions from the saved model
```
