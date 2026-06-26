# 🛒 Rossmann Store Sales Forecasting

### _Can a neural network predict what 1,115 stores will sell — six weeks in advance?_

---

## The Problem

Rossmann operates over 3,000 drug stores across Europe. Store managers are expected to submit sales forecasts up to six weeks ahead — and they do it mostly from gut feel.

Bad forecasts mean overstocked shelves, understaffed counters, or worse: empty shelves on a promo day. The cost isn't just revenue — it's customer trust.

This project asks: **can we do better than intuition?**

---

## What I Built

A full forecasting pipeline — from raw transaction logs to a live interactive dashboard — using **1.01 million rows** of historical sales data across 1,115 stores.

```
Raw Data → Feature Engineering → LSTM + XGBoost → Streamlit Dashboard
```

The dashboard lets you pick any store, toggle a promotion on/off, and watch the LSTM forecast update in real time.

---

## The Approach

### Step 1 — Feature Engineering

Sales data alone isn't enough. The model needs context:

- **Lag features** — what did this store sell 7 and 30 days ago?
- **Rolling statistics** — what's the 7-day mean and volatility?
- **Calendar features** — day of week, month, school holidays, promotions

One subtle but critical detail: lag and rolling features were computed **per store**, not globally. Mixing stores during the shift operation would leak Store 500's sales into Store 1's lag — a silent bug that corrupts every prediction.

### Step 2 — Sequence Construction

For the LSTM, sales history was converted into sliding windows of **30 days**. Each window becomes one training sample, and the model learns to predict day 31.

- Training sequences: **718,881**
- Test sequences: **25,161**
- Features per timestep: **8**

### Step 3 — Two Models, One Question

I trained both an LSTM and XGBoost on the same sequences — on purpose. The goal wasn't to pick a winner before training; it was to let the data decide.

| Model          | MAE     | RMSE    | R²        |
| -------------- | ------- | ------- | --------- |
| LSTM           | 784     | 1,083   | 0.867     |
| XGBoost        | 602     | 843     | 0.920     |
| XGBoost Tuned  | **570** | **799** | **0.928** |
| Naive Baseline | —       | 3,221   | —         |

XGBoost won. And that's actually the interesting result — not a failure.

---

## Why XGBoost Beat the LSTM

LSTMs are powerful sequence learners, but they shine when the signal is genuinely temporal and hard to hand-engineer — speech, text, sensor data.

Here, we already gave the model explicit lag and rolling features. XGBoost can exploit those directly. The LSTM has to _rediscover_ the same patterns from the raw sequence, with far more parameters and far less data per store.

The LSTM still beats a naive baseline by **66.4%**. It learned something real. XGBoost just learned it more efficiently.

> "Use the simplest model that solves the problem." — this project is the demonstration.

---

## The Dashboard

A Streamlit app with five tabs:

| Tab                   | What it shows                                                     |
| --------------------- | ----------------------------------------------------------------- |
| 📊 Overview           | Dataset summary, store type breakdown, sales distributions        |
| 📈 Store Explorer     | Per-store sales history and trends                                |
| 🔬 Model Performance  | MAE, RMSE, MAPE, R², naive baseline comparison                    |
| 💡 Business Insights  | Promo impact, seasonality, staffing and inventory recommendations |
| 🔮 Forecast Simulator | Pick a store → toggle promo → see live LSTM forecast              |

---

## Results That Mean Something

A RMSE of 1,083 units sounds abstract. Here's what it means in practice:

- **Inventory:** Safety stock = `1.65 × 1,083 × √3 ≈ 3,094 units` — a concrete buffer derived from model error
- **Promotions:** Model captures ~20% uplift on promo days; plan staffing accordingly
- **Planning horizon:** Forecasts are reliable at 6-week range, degrading gracefully beyond

---

## Tech Stack

```
Python · Pandas · NumPy
TensorFlow / Keras  (LSTM)
XGBoost
Scikit-learn
Streamlit
Matplotlib
Joblib
```

---

## Project Structure

```
rossmann-store-sales/
├── notebooks/
│   ├── 01_eda_preprocessing_fixed.ipynb   # Feature engineering & EDA
│   └── 02_lstm_model_training_fixed.ipynb # Model training & evaluation
├── dashboard/
│   └── app.py                             # Streamlit dashboard
├── models/
│   ├── best_model.keras                   # Trained LSTM
│   ├── xgb_model.pkl                      # XGBoost baseline
│   └── xgb_tuned_model.pkl               # XGBoost tuned (best)
├── sequences/                             # Saved numpy sequences
├── outputs/                               # Predictions & visualisations
└── data/rossmann-store-sales/            # Raw CSVs
```

---

## Run Locally

```bash
git clone https://github.com/akshita-singh-2808/rossmann-store-sales
cd rossmann-store-sales
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

---

## What I'd Do Next

- **Store embeddings** — let the model learn store-specific patterns without training 1,115 separate models
- **Uncertainty quantification** — predict a range, not just a point estimate
- **Online learning** — retrain incrementally as new sales data arrives

---

_Built by [Akshita Singh](https://linkedin.com/in/akshita-singh2808) · B.Tech Mechanical Engineering · MNNIT Allahabad_
_Data source: [Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales)_
