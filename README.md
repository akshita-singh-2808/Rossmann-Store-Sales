# 📊 Rossmann Store Sales Forecasting using LSTM

Deep Learning based sales forecasting project built using TensorFlow, LSTM networks, and Streamlit.

---

## 🚀 Project Overview

This project predicts daily sales for Rossmann retail stores using a multivariate Long Short-Term Memory (LSTM) neural network.

The model learns temporal sales patterns such as:

* Weekly seasonality
* Promotion impact
* Holiday effects
* Rolling sales behavior
* Store demand trends

An interactive Streamlit dashboard was built to visualize:

* Exploratory Data Analysis (EDA)
* Business insights
* Model performance
* Store-level forecasting simulation

---

## 📂 Dataset

**Source:** Rossmann Store Sales Dataset (Kaggle)

### Dataset Details

* 1,115 stores
* Daily sales records
* Time period: Jan 2013 – Jul 2015

### Main Features Used

* Promo
* SchoolHoliday
* DayOfWeek
* Month
* Lag_7
* Lag_30
* Rolling_Mean_7
* Rolling_Std_7

---

## 🧠 Model Architecture

### LSTM Network

* 2 stacked LSTM layers
* Dropout regularization
* Dense hidden layer
* TensorFlow / Keras implementation

### Input Shape

* 30-day rolling sequences
* 8 engineered features

### Training Setup

* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* EarlyStopping + ReduceLROnPlateau callbacks

---

## 📈 Model Performance

| Metric | Value  |
| ------ | ------ |
| MAE    | 752    |
| RMSE   | 1124   |
| MAPE   | 11.38% |

### Baseline Comparison

The LSTM model outperformed the naive baseline forecast by approximately **70.4%**.

---

## 📊 Dashboard Features

### ✅ Exploratory Data Analysis

* Daily sales trends
* Day-of-week sales patterns
* Promotion impact analysis
* Monthly seasonality heatmap

### ✅ Model Evaluation

* Actual vs Predicted plots
* Scatter plots
* Residual distribution
* Model comparison table

### ✅ Forecast Simulator

Interactive forecasting system where users can:

* Select a store
* Toggle promotions
* Visualize store-level forecasts
* Compare actual vs predicted sales

---

## 🛠️ Tech Stack

### Languages & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* TensorFlow / Keras
* Scikit-learn
* Streamlit

---

## 📁 Project Structure

```bash
Rossmann_LSTM_Project/
│
├── notebooks/
│   ├── 01_eda_feature_engineering.ipynb
│   ├── 02_lstm_model_training.ipynb
│   └── 03_business_insights.ipynb
│
├── models/
│   ├── best_model.keras
│   ├── feature_scaler.pkl
│   └── target_scaler.pkl
│
├── outputs/
│   ├── final_predictions.csv
│   └── training_history.csv
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Streamlit App

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit

```bash
streamlit run app.py
```

---

## 💡 Key Learnings

* Time-series forecasting using LSTM
* Feature engineering for sequential models
* Sequence generation using rolling windows
* Scaling and inverse transformation
* Model evaluation for forecasting tasks
* Interactive ML dashboard deployment

---

## 🔮 Future Improvements

* Add Store ID embedding layer
* Recursive multi-step forecasting
* Hyperparameter tuning
* Attention-based forecasting models
* Cloud deployment

---

## 👩‍💻 Author

Akshita Singh

* GitHub:(https://github.com/akshita-singh-2808)
* LinkedIn: https://linkedin.com/in/akshita-singh2808
