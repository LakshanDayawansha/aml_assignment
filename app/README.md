# Churn Predictor (EBM front end)

Streamlit app for the EBM churn models in `../models/`.

## Run

```bash
cd app
../.venv/bin/pip install -r requirements.txt   # or: uv pip install --python ../.venv/bin/python -r requirements.txt
../.venv/bin/streamlit run app.py
```

Open http://localhost:8501. On the first visit to **Customer lookup**, the raw usage for test customers is
read from `data/raw/main.csv` and cached in `app/cache/test_usage.parquet`. This takes about 15 seconds.

## Pages

| Page | What it does |
|---|---|
| Overview | KPIs, probability distribution, risk bands, highest-risk test customers |
| Customer lookup | Churn gauge, per-feature EBM explanation, 90-day usage timeline and activity map |
| Batch scoring | Upload raw usage (`main.csv` format) or a features CSV, then download scored results with top reasons |
| Model insights | Global importance, shape functions and interaction heatmaps, metrics for both models |

Use the sidebar to switch between the **90-day** model (`ebm_churn`, threshold 0.43) and the
**30-day early-warning** model (`ebm_churn_30d`, threshold 0.5).

## Code

- `core/features.py`: feature engineering copied from `data_preprocess/preprocess.ipynb` (verified to give identical features)
- `core/models.py`: model loading, scoring, explanations, data caching
- `core/charts.py`: Plotly charts
- `views/`: one file per page
