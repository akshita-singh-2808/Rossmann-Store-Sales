
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "none",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "grid.color": "white",
    "grid.alpha": 0.08,
    "axes.grid": True,
    "font.size": 11
})

# ----------------------------------------------------------------
# PAGE CONFIG — must be first streamlit command
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------
@st.cache_data
def load_data():
    train = pd.read_csv(BASE_DIR /'data'/'rossmann-store-sales'/ "train.csv", low_memory=False)

    store = pd.read_csv(BASE_DIR /'data'/'rossmann-store-sales'/ "store.csv")

    store["CompetitionDistance"]       = store["CompetitionDistance"].fillna(store["CompetitionDistance"].median())
    store["CompetitionOpenSinceMonth"] = store["CompetitionOpenSinceMonth"].fillna(0)
    store["CompetitionOpenSinceYear"]  = store["CompetitionOpenSinceYear"].fillna(0)
    store["Promo2SinceWeek"]           = store["Promo2SinceWeek"].fillna(0)
    store["Promo2SinceYear"]           = store["Promo2SinceYear"].fillna(0)
    store["PromoInterval"]             = store["PromoInterval"].fillna("NoPromo")

    df = train.merge(store, on="Store", how="left")
    df = df[df["Open"] == 1].copy()
    df["Date"]           = pd.to_datetime(df["Date"])
    df["Year"]           = df["Date"].dt.year
    df["Month"]          = df["Date"].dt.month
    # Fix: sort by Store+Date and compute lag/rolling per-store to avoid cross-store leakage
    df = df.sort_values(["Store", "Date"])
    df["Lag_7"]          = df.groupby("Store")["Sales"].shift(7)
    df["Lag_30"]         = df.groupby("Store")["Sales"].shift(30)
    df["Rolling_Mean_7"] = df.groupby("Store")["Sales"].transform(lambda x: x.rolling(7).mean())
    df["Rolling_Std_7"]  = df.groupby("Store")["Sales"].transform(lambda x: x.rolling(7).std())
    return df.dropna()

@st.cache_data
def load_predictions():
    return  pd.read_csv(BASE_DIR /'outputs'/ "final_predictions.csv")

# Load model once — @st.cache_resource keeps it in memory
@st.cache_resource
def load_model_and_scalers():
    from tensorflow.keras.models import load_model
    model = load_model(BASE_DIR /'models'/ "best_model.keras",compile=False)
    feature_scaler = joblib.load(BASE_DIR /'models'/ "feature_scaler.pkl")
    target_scaler = joblib.load(BASE_DIR /'models'/ "target_scaler.pkl")
    return model, feature_scaler, target_scaler

df     = load_data()
preds  = load_predictions()
y_true = preds["Actual"].values
y_pred = np.clip(preds["Predicted"].values, 0, None)

# Lazy-load TensorFlow model only when Forecast Simulator is opened
model = None
feature_scaler = None
target_scaler = None

# ----------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------
with st.sidebar:

    st.title("📊 Rossmann Forecasting")

    st.markdown(
        "Deep Learning Sales Forecasting using LSTM & XGBoost"
    )

    st.markdown("---")

    st.subheader("Dataset")

    st.write(
        "1,115 stores\n\n"
        "Jan 2013 – Jul 2015"
    )

    st.markdown("---")

    st.subheader("Model")

    st.write(
        "2-layer LSTM | XGBoost \n\n "
        "8 engineered features\n\n"
        "TensorFlow / Keras"
    )

    st.markdown("---")

    st.subheader("Performance")

    st.metric("MAE", "784")

    st.metric("MAPE", "12.27%")

    st.metric(
        "Improvement vs Naive",
        "66.4%"
    )

    st.markdown("---")

    st.caption("Built by Akshita Singh")

    st.markdown(
        """
        🌐 [GitHub](https://github.com/akshita-singh-2808)

        💼 [LinkedIn](https://linkedin.com/in/akshita-singh2808)
        """
    )

# ----------------------------------------------------------------
# TABS
# ----------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " OVERVIEW           ",
    "EDA                 ",
    "MODEL PERFORMANCE    ",
    "BUSINESS INSIGHTS    ",
    "FORECAST SIMULATOR    "
])


# ================================================================
# TAB 1 — OVERVIEW
# ================================================================
with tab1:
    st.title("Rossmann Store Sales Forecasting")
    st.markdown(
        "**Business Problem:** Rossmann operates 1,115 drug stores across Germany. "
        "This model forecasts daily sales per store to optimise "
        "inventory planning, staffing, and promotions."
    )
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Stores",   "1,115")
    c2.metric("MAE",            "784 units")
    c3.metric("RMSE",           "1,083 units")
    c4.metric("MAPE",           "12.27%")
    c5.metric("Beats Naive By", "66.4%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What this model does")
        st.markdown("""
        - Predicts **daily sales** for each Rossmann store
        - Trained on **2.5 years** of historical data
        - Uses promotions, holidays, day of week, lag features
        - Single model covering **all 1,115 stores** simultaneously
        """)

    with col2:
        st.subheader("Business value")
        st.markdown("""
        - **Inventory:** Set safety stock using per-store RMSE
        - **Staffing:** Use day-of-week index to schedule staff
        - **Promotions:** Pre-order extra units 3 days before promo
        - **Supply chain:** Automate reordering for low-MAPE stores
        """)

    st.markdown("---")
    st.subheader("Project Pipeline")
    st.markdown("""
    `Raw Data` → `EDA` → `Feature Engineering` → `Sequence Creation (window=30)`
    → `Model Training` → `Evaluation` → `Business Insights` → `Dashboard`
    """)


# ================================================================
# TAB 2 — EDA
# ================================================================
with tab2:
    st.title("Exploratory Data Analysis")
    st.markdown("Understanding the data before modelling.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Daily Sales Over Time")
        daily = df.groupby("Date")["Sales"].sum()
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(daily.index, daily.values / 1e6, color="#2563eb", linewidth=0.7)
        ax.set_ylabel("Sales (millions)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Weekly dips = Sundays. December spikes = Christmas season.")

    with col2:
        st.subheader("Average Sales by Day of Week")
       
        dow      = df.groupby("DayOfWeek")["Sales"].mean()
        day_names= {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
        fig, ax  = plt.subplots(figsize=(10,5))
        ax.bar([day_names[d] for d in dow.index], dow.values,
               color="#7c3aed", edgecolor="white")
        ax.set_ylabel("Avg Sales (units)")
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(dow.values):
            ax.text(i, v + 50, f"{v/1000:.1f}k", ha="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Monday and Sunday peak. Saturday is consistently slowest.")

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Promo vs No-Promo Sales")
        promo = df.groupby("Promo")["Sales"].mean()
        lift  = (promo[1] / promo[0] - 1) * 100
       
        fig,ax=plt.subplots(figsize=(10,5))
        
        ax.bar(["No Promo", "Promo"],  [promo[0], promo[1]],color=["orange","#7c3aed"], edgecolor="white")
        for i, v in enumerate([promo[0], promo[1]]):
            ax.text(i, v + 100, f"{v:,.0f}", ha="center", fontweight="bold")
        ax.set_title(f"Promotion Lift = +{lift:.1f}%", fontweight="bold")
        ax.set_ylabel("Avg daily sales")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Lift is calculated as: (Avg Sales with Promo / Avg Sales without Promo - 1) * 100")
    with col4:
        st.subheader("Monthly Seasonality Heatmap")
        monthly      = df.groupby(["Year","Month"])["Sales"].mean().unstack()
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        available    = [month_labels[m-1] for m in monthly.columns]
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(monthly, annot=True, fmt=".0f", cmap="Blues",
                    ax=ax, xticklabels=available,
                    annot_kws={"size": 7}, linewidths=0.3)
        ax.set_ylabel("Year")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("December peaks every year. January dips post-Christmas.")
with tab3:
    st.title("Model Performance")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("[MAE](https://www.datacamp.com/tutorial/mean-absolute-error)")
        st.metric("MAE", "784 units", "avg daily error")

    with c2:
        st.markdown("[RMSE](https://www.datacamp.com/tutorial/rmse)")
        st.metric("RMSE", "1,083 units", "penalises big errors")

    with c3:
        st.markdown("[MAPE](https://coralogix.com/ai-blog/a-comprehensive-guide-to-mean-absolute-percentage-error-mape/)")
        st.metric("MAPE", "12.27%", "industry bench: 10-15%")

    with c4:
        st.markdown("[Naive Baseline](https://insightful-data-lab.com/2025/08/22/naive-baseline-forecast/)")
        st.metric('RMSE', "3,221 units", "-66.4% improvement")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted Sales")
        fig, ax = plt.subplots(figsize=(10,5)) 
        ax.plot(y_true[:500], label="Actual",    color="#2563eb", linewidth=1.5) #only plot first 500 for clarity
        ax.plot(y_pred[:500], label="Predicted", color="#dc2626",
                linewidth=1.5, linestyle="--", alpha=0.85)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sales (units)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("First 500 test samples across all stores.")
    with col2:
        st.subheader("Predicted vs Actual (Scatter)")
        idx = np.random.choice(len(y_true), 3000, replace=False)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(y_true[idx], y_pred[idx], alpha=0.3, color="lightblue", s=8)
        lim = max(y_true.max(), y_pred.max()) * 1.02
        ax.plot([0, lim], [0, lim], "r--", linewidth=2, label="Perfect forecast")
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Points close to red line = accurate predictions.")

        st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Residual Distribution")
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(10,5))
        ax.hist(residuals, bins=60, color="lightgreen", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="#dc2626", linewidth=2,
                   linestyle="--", label="Zero error")
        ax.axvline(residuals.mean(), color="#2563eb", linewidth=2,
                   linestyle=":", label=f"Mean ({residuals.mean():,.0f})")
        ax.set_xlabel("Prediction Error (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Centered around 0 = no systematic bias.")


    with col4:
        st.subheader("Model Comparison")
        comp = pd.DataFrame({
        "Model": [
            "LSTM",
            "XGBoost",
            "XGBoost Tuned",
        ],

        "MAE": [
            784.13,
            601.59,
            569.94,
        ],

        "RMSE": [
            1082.69,
            842.73,
            798.72,
        ],

        "R² Score": [
            0.867,
            0.920,
            0.928,
        ],

        "Train Time": [
            "2 hrs",
            "5 min",
            "10 min",
        ]
    })
        st.dataframe(
        comp,
        use_container_width=True,
        hide_index=True
    )

        st.caption(
            "XGBoost Tuned achieved the best overall forecasting accuracy "
            "with lowest MAE/RMSE and highest R² score."
        )
# TAB 4 — BUSINESS INSIGHTS
# ================================================================
with tab4:
    st.title("Business Insights")
    st.markdown("Translating model results into operational decisions.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Promotion Strategy")
        promo = df.groupby("Promo")["Sales"].mean()
        lift  = (promo[1] / promo[0] - 1) * 100
        extra = promo[1] - promo[0]
        st.success(f"""
**Promo lift: +{lift:.1f}% sales increase**

- No Promo avg : {promo[0]:,.0f} units/day
- Promo avg    : {promo[1]:,.0f} units/day
- Extra units  : +{extra:,.0f} per store per day

**Action:** Pre-order extra stock 3 days before
promo launch using 7-day forecast as signal.
        """)

    with col2:
        st.subheader("Staffing Index by Day")
        dow     = df.groupby("DayOfWeek")["Sales"].mean()
        overall = dow.mean()
        idx     = (dow / overall).round(2)
        day_names = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
        idx_df = pd.DataFrame({
            "Day":            [day_names[d] for d in idx.index],
            "Staffing Index": idx.values,
            "Action":         ["More staff" if v >= 1 else "Fewer staff"
                               for v in idx.values]
        })
        st.dataframe(idx_df, use_container_width=True, hide_index=True)
        st.info("Multiply your base headcount by the index for each day.")

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Safety Stock Formula")
        
        st.markdown("""
        **Industry formula:**
        ```
        Safety Stock = 1.65 × std(error) × √(lead_time)
        ```
        - `1.65` = Z-score for 95% service level
        - `std(error)` = std dev of model's daily forecast error
        - `lead_time` = 3 days (typical supplier lead time)

        **With your model (RMSE = 1,083):**

        Safety Stock ≈ **1.65 × 1083 × √3 ≈ 3,094 units**

        Keep ~3,213 units buffer per store to avoid
        stockouts 95% of the time.
        """)
        st.markdown("[TO LEARN MORE](https://pallitegroup.com/en/news/safety-stock-formula/)")

    with col4:
        st.subheader("Store Type Recommendations")
        st.markdown("""
        | Type | Avg Sales | Recommendation |
        |------|-----------|----------------|
        | A | ~6,500 | Standard reorder cycle |
        | B | ~10,000 | Higher safety stock |
        | C | ~6,600 | Standard reorder cycle |
        | D | ~6,700 | Standard reorder cycle |

        **Type B** has highest sales and promo sensitivity.
        Prioritise for promotion campaigns and larger buffers.
        """)

    st.markdown("---")
    st.subheader("Monthly Action Calendar")
    st.markdown("""
    | Month | Action |
    |-------|--------|
    | Jan–Feb | Run clearance promos — slowest months |
    | Mar–Apr | Spring uptick — restock early |
    | Oct–Nov | **Build inventory for December peak** |
    | Dec | Peak month — maximum stock, maximum staff |
    """)

with tab5:
    st.title("Forecast Simulator")

    with st.spinner("Loading model..."):
        model, feature_scaler, target_scaler = load_model_and_scalers()
    st.write(
    "The forecasting model was trained using historical data from all Rossmann stores. "
    "In the dashboard, the selected store's historical sales and engineered features "
    "are used to generate store-level sales forecasts."
    )
    st.markdown("Select a store and explore its historical sales alongside the model forecasts.")
    st.markdown("---")

    # User inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        store_id = st.selectbox(
            "Select Store",
            options=sorted(df["Store"].unique()),
            index=0
        )
    with col2:
        promo_on = st.radio(
            "Promotion Active?",
            options=["No", "Yes"],
            index=0,
            horizontal=True
        )
    with col3:
        show_days = st.slider(
            "Days to display",
            min_value=61,   # Fix Bug 3: must be > WINDOW(30)+1 to produce at least one prediction
            max_value=120,
            value=90
        )

    # Filter to selected store
    store_data = df[df["Store"] == store_id].sort_values("Date").tail(show_days).copy()

    if len(store_data) == 0:
        st.warning(f"No data found for Store {store_id}")

    else:
        # Features must match Notebook 01 exactly
        FEATURES = ["Promo", "SchoolHoliday", "DayOfWeek", "Month",
                    "Lag_7", "Lag_30", "Rolling_Mean_7", "Rolling_Std_7"]
        WINDOW   = 30

        # Apply promo toggle: set Promo flag AND recompute rolling/lag features
        # Fix Bug 2: promo toggle previously only changed Promo column; rolling features unchanged
        store_seq = store_data.copy()
        if promo_on == "Yes":
            store_seq["Promo"] = 1
            # Recompute lag/rolling on the hypothetical promo-boosted series
            # Use a rough +20% uplift on Sales to reflect expected promo effect on lag features
            boosted = store_seq["Sales"] * 1.20
            store_seq["Lag_7"]          = boosted.shift(7).bfill().fillna(boosted.mean())
            store_seq["Lag_30"]         = boosted.shift(30).bfill().fillna(boosted.mean())
            store_seq["Rolling_Mean_7"] = boosted.rolling(7, min_periods=1).mean()
            store_seq["Rolling_Std_7"]  = boosted.rolling(7, min_periods=1).std().fillna(0)

        # Model was trained on raw (unscaled) features — do NOT apply feature_scaler
        X_raw = store_seq[FEATURES].values

        # Guard 1: not enough rows to build even one window -> would give an
        # empty array with the wrong shape and crash model.predict()
        if len(X_raw) <= WINDOW:
            st.warning(
                f"Store {store_id} only has {len(X_raw)} day(s) of data in this "
                f"view, but the model needs more than {WINDOW} days of history "
                f"to produce a forecast. Try increasing 'Days to display' or "
                f"picking a different store."
            )
            forecast = pd.Series(np.full(len(store_seq), np.nan), index=store_seq.index)
        else:
            # Guard 2: the promo-boost recompute can leave NaNs (e.g. rolling std
            # on a single value, or an all-NaN slice before bfill runs out of
            # values to fill). NaNs silently propagate through predict/inverse
            # transform and can crash or produce garbage — so clean them up.
            X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

            # Build sequences
            X_seq = np.array([X_raw[i:i + WINDOW] for i in range(len(X_raw) - WINDOW)])

            # Predict
            preds_scaled = model.predict(X_seq, verbose=0)
            preds_actual = target_scaler.inverse_transform(preds_scaled).flatten()
            preds_actual = np.clip(preds_actual, 0, None)

            # Pad first WINDOW rows with NaN (no prediction for them)
            padding  = np.full(WINDOW, np.nan)
            forecast = pd.Series(
                np.concatenate([padding, preds_actual]),
                index=store_seq.index
            )

        # KPI cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Store",           f"#{store_id}")
        c2.metric("Avg Daily Sales",  f"{store_data['Sales'].mean():,.0f}")
        c3.metric("Peak Sales",       f"{store_data['Sales'].max():,.0f}")
        c4.metric("Promo Days",       f"{int(store_data['Promo'].sum())}")

        st.markdown("---")

        # Forecast chart
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(store_data["Date"].values, store_data["Sales"].values,
                label="Actual Sales", color="#2563eb", linewidth=1.8)

        ax.plot(store_data["Date"].values, forecast.values,
                label=f"LSTM Forecast {'(Promo ON)' if promo_on=='Yes' else ''}",
                color="#ff4d00", linewidth=1.8, linestyle="--", alpha=0.85)

        # Mark promotion days (axvspan with same start/end is zero-width; use axvline)
        promo_dates = store_data[store_data["Promo"] == 1]["Date"]
        for d in promo_dates:
            ax.axvline(d, alpha=0.12, color="#03fc13", linewidth=1.5)

        ax.set_title(f"Store {store_id} — Sales & Model Forecast",
                     fontweight="bold", fontsize=13)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales (units)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("Green shading = promotion days. "
                   "First 30 days have no forecast (need full window).")

        # Day of week pattern for this store
        st.markdown("---")
        st.subheader(f"Store {store_id} — Sales by Day of Week")

        dow_store = store_data.groupby("DayOfWeek")["Sales"].mean()
        day_names = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar([day_names.get(d, str(d)) for d in dow_store.index],
               dow_store.values, color="#7c3aed", edgecolor="white")
        ax.set_ylabel("Avg Sales (units)")
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(dow_store.values):
            ax.text(i, v + 50, f"{v:,.0f}", ha="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
