# Superstore Sales Dashboard (Streamlit)

A 4-page interactive Streamlit app built on the Superstore sales dataset (`data/train.csv`):

1. **Sales Overview** — total sales by year, monthly trend, sales by region/category with filters
2. **Forecast Explorer** — Holt-Winters forecasts by Category or Region, 1–3 month horizon, MAE/RMSE
3. **Anomaly Report** — STL-decomposition based anomaly detection on monthly sales
4. **Product Demand Segments** — K-Means clustering of sub-categories into High/Medium/Low demand

All the heavy lifting (model training, forecasting, anomaly detection, clustering) is done
**once**, offline, by `build_artifacts.py`, which saves small CSV/JSON files into `artifacts/`.
The Streamlit app only reads those pre-computed files at runtime — this keeps the deployed
app fast and avoids re-training models on every page load / every user session.

---

## 1. Run it locally (optional, to check everything first)

```bash
pip install -r requirements.txt

# Only needed once, or whenever data/train.csv changes:
python build_artifacts.py

# Launch the app:
streamlit run Home.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

---

## 2. Deploy to Streamlit Community Cloud (free)

### Step A — Push this folder to GitHub

1. Create a new **public** GitHub repository, e.g. `superstore-dashboard`.
2. From inside this folder, run:

```bash
git init
git add .
git commit -m "Initial commit: Superstore Streamlit dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/superstore-dashboard.git
git push -u origin main
```

> **Important:** make sure `data/train.csv` and the `artifacts/` folder (already generated)
> are both committed — the app reads `artifacts/*.csv` / `*.json` directly, so as long as
> `artifacts/` is pushed you do **not** need to re-run `build_artifacts.py` on the cloud.
> If you'd rather regenerate artifacts on every deploy instead of committing them, add a
> `python build_artifacts.py` step, but for a free/simple deployment, committing the
> pre-built `artifacts/` folder is simplest and fastest.

### Step B — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Choose:
   - **Repository:** `<your-username>/superstore-dashboard`
   - **Branch:** `main`
   - **Main file path:** `Home.py`
4. Click **"Deploy"**.
5. Streamlit Cloud will install everything from `requirements.txt` and launch the app.
   First deploy usually takes 1–3 minutes.
6. Once it's live, you'll get a shareable URL like:
   `https://<your-username>-superstore-dashboard-<hash>.streamlit.app`

That URL is what you submit.

### If you update the app later

Just `git push` your changes to `main` — Streamlit Community Cloud auto-redeploys on every push.

---

## Project structure

```
superstore_dashboard/
├── Home.py                          # Page 1 — Sales Overview (entry point)
├── pages/
│   ├── 2_Forecast_Explorer.py       # Page 2 — Forecast Explorer
│   ├── 3_Anomaly_Report.py          # Page 3 — Anomaly Report
│   └── 4_Product_Demand_Segments.py # Page 4 — Product Demand Segments
├── build_artifacts.py               # Offline pipeline: builds all artifacts/
├── data/
│   └── train.csv                    # Raw Superstore sales data
├── artifacts/                       # Pre-computed outputs the app reads (generated)
│   ├── yearly_sales.csv
│   ├── monthly_sales.csv
│   ├── monthly_sales_by_region.csv
│   ├── monthly_sales_by_category.csv
│   ├── region_category_sales.csv
│   ├── forecasts.json
│   ├── forecast_metrics.json
│   ├── anomalies.csv
│   └── clusters.csv
├── requirements.txt
├── .streamlit/config.toml           # theme
└── README.md
```

## Methodology notes

- **Forecasting:** Holt-Winters / triple exponential smoothing (`statsmodels`), one model
  per Category (3 models) and per Region (4 models). Each model is backtested by holding
  out the last 3 real months, forecasting them, and comparing to what actually happened —
  that comparison produces the MAE/RMSE shown in the app. The model is then refit on the
  full series to produce the actual 1–3 month forward forecast shown to the user.
- **Anomaly detection:** Monthly sales are highly seasonal (spikes every Sep/Nov/Dec), so a
  plain rolling z-score doesn't work well. Instead the series is STL-decomposed into
  Trend + Seasonal + Residual; months whose Residual is a statistical outlier
  (|z-score| > 1.5) are flagged as anomalies — i.e., months that were unusual even after
  accounting for the expected seasonal pattern.
- **Clustering:** Each Sub-Category is described by 4 features — total sales, average
  monthly sales, volatility (coefficient of variation), and growth (2nd half vs 1st half
  average) — standardized and clustered with K-Means (k=3). Clusters are labelled
  High / Medium / Low Demand based on average total sales per cluster.

## Note on `vgsales.csv`

The other uploaded file, `vgsales.csv` (video game sales by platform/genre/year), does not
contain the Region, Category, or monthly date fields the assignment's Page 1 and Page 2
requirements call for, so it was not used in this dashboard. If you actually need the
dashboard built on that dataset instead (or in addition), let me know and I'll adapt it —
it would need Genre/Platform used in place of Category/Region, and only yearly (not
monthly) granularity is available in that file.
