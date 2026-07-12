# Superstore Sales Intelligence Dashboard

A 4-page Streamlit app built on the Superstore order dataset (`data/train.csv`):

1. **Sales Overview Dashboard** — yearly sales bar chart, monthly trend line, sales by region & category with interactive filters.
2. **Forecast Explorer** — pick Category or Region, choose a 1/2/3-month horizon, view the Holt-Winters forecast plus backtest MAE/RMSE.
3. **Anomaly Report** — monthly sales chart with anomalies flagged (rolling-trend residual z-score), plus a table of anomaly dates/values.
4. **Product Demand Segments** — KMeans clustering of sub-categories by demand size, volatility, and trend, with a PCA scatter plot and cluster assignment table.

The app is fully self-contained — the dataset ships inside the repo, so it works immediately after deployment with no extra setup.

---

## 1. Run locally (optional, to check before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 2. Deploy to Streamlit Community Cloud (free) — step by step

**Step 1 — Push this folder to GitHub**
1. Create a new **public** GitHub repository (e.g. `superstore-dashboard`).
2. Upload everything in this folder to the repo root, keeping the structure:
   ```
   your-repo/
   ├── app.py
   ├── requirements.txt
   ├── .streamlit/
   │   └── config.toml
   └── data/
       └── train.csv
   ```
   Easiest way: on the new repo's GitHub page, click **"Add file → Upload files"**, drag in all the files/folders from this project, and commit.

**Step 2 — Deploy on Streamlit Community Cloud**
1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **"Create app"** (or **"New app"**).
3. Choose **"Deploy a public app from GitHub"**.
4. Select:
   - **Repository:** `your-username/superstore-dashboard`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy"**.

Streamlit Cloud will install everything from `requirements.txt` and launch the app. First build takes 1–3 minutes. You'll get a live link like:

```
https://your-app-name.streamlit.app
```

That link is what you submit.

**Step 3 — Verify before submitting**
Open the live link and click through all four pages (sidebar navigation) to confirm charts and tables render correctly.

---

## Notes on the data / modeling choices

- **Dataset:** `data/train.csv` (Superstore-style order data, 2015–2018, ~9,800 orders) — the only uploaded file with order-level dates, Region, Category, and Sub-Category, which is what all four pages require. (`vgsales.csv` was also provided but only has a Year column, no monthly dates, so it isn't suited to a monthly-trend/forecast/anomaly dashboard and was not used.)
- **Forecast model:** Holt-Winters Exponential Smoothing (additive trend + yearly seasonality) via `statsmodels`. MAE/RMSE are computed by holding out the last few months as a backtest window.
- **Anomaly detection:** matches your Task 5 approach — weekly sales totals, flagged where the z-score (deviation from the overall mean, scaled by the overall std) exceeds a threshold (default 3.0). Sensitivity is adjustable in the app.
- **Clustering:** KMeans on standardized sub-category features (total sales, average monthly sales, volatility, trend slope). The **Elbow Method** chart (WCSS vs. k) from Task 6 is shown first to justify the cluster count, followed by the PCA cluster map and the sub-category → cluster assignment table.

## Troubleshooting

- **"ModuleNotFoundError" on deploy:** Make sure `requirements.txt` was uploaded to the repo root exactly as included here.
- **App shows old data after editing `train.csv`:** Streamlit Cloud rebuilds on every push to the branch — just commit again.
- **Slow first load:** normal on the free tier; the app "sleeps" after inactivity and takes ~30s to wake up.
