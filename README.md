# 🚕 Taxi Orders Forecasting
*Time-series forecasting of hourly taxi demand with feature-engineered ML and classical baselines.*

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![pandas](https://img.shields.io/badge/pandas-EDA-green?logo=pandas)  
![numpy](https://img.shields.io/badge/numpy-Numerical-blue?logo=numpy)  
![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-orange?logo=matplotlib)  
![statsmodels](https://img.shields.io/badge/statsmodels-TS%20Models-lightgrey)  
![pmdarima](https://img.shields.io/badge/pmdarima-ARIMA-blue)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)  
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Overview
The goal is to **forecast the number of taxi orders for the next hour** from historical hourly demand.  
Business constraint: **test RMSE ≤ 48**.

This project benchmarks classical time-series models (**AR**, **MA**, **ARIMA via auto_arima**) against a **feature-engineered LightGBM regressor**. The winning model achieves **substantial error reduction** while respecting temporal validation.

---

## 📊 Data & Preparation
- **Granularity:** hourly orders; resampled to 1H and filled as needed.  
- **Feature engineering (for ML):**
  - **Calendar:** `hour`, `dayofweek`, `is_weekend`
  - **Lags:** `lag_1`, `lag_2`, `lag_24`
  - **Rolling means:** `roll_3`, `roll_6`, `roll_24`
  - **Target transform:** model in log space (`log1p`) with safe inverse (`expm1` + clipping)

---

## ⚙️ Methods
- **Classical baselines**
  - **AR(p)** with `statsmodels.AutoReg` (e.g., p=24)
  - **MA(q)** equivalent via ARIMA(0,0,q) on log target
  - **ARIMA(p,d,q)** via `pmdarima.auto_arima` (selected order e.g., (2,0,2))
- **Machine Learning**
  - **LightGBM Regressor** trained on engineered features (lags, rollings, calendar) using log-target
- **Validation protocol**
  - Chronological split: **80% train → 10% validation → 10% test**
  - Select the best model on validation; **refit on train+val (90%)**; evaluate once on **held-out 10% test**
  - Primary metric: **RMSE**

---

## 📈 Results
- On train/validation, **LightGBM** far outperformed classical baselines:  
  - **LightGBM RMSE ≈ 19.56** (val)  
- After refitting on train+val and testing on the final 10%:  
  - **LightGBM RMSE ≈ 31.41 (test)** — comfortably **below the 48 target**  
- Classical baselines (**AR(24), MA(24), ARIMA(2,0,2)**) did **not** meet the 48 RMSE requirement, confirming the value of feature-engineered boosting for short-horizon hourly demand.

**Takeaway:** Feature-rich **LightGBM** captures short-term dependencies (lags/rollings) and recurring patterns (calendar) better than purely classical approaches in this setting.

---

## 🗂 Repo Structure
```
taxi-orders-forecasting/
├── notebooks/ <- Final cleaned Jupyter notebook
├── data/ <- Time series data (or link to source)
├── requirements.txt <- Dependencies
├── LICENSE <- MIT (or your choice)
└── README.md <- This file
```

---

## 🚀 How to Run
1. Clone the repo:
    ```bash
    git clone https://github.com/Flamingo-Rocker/taxi-orders-forecasting.git
    cd taxi-orders-forecasting
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Open the notebook in `/notebooks` to reproduce the analysis.

---

## 📦 Requirements
```
pandas==2.3.2
numpy==2.2.5
numpy-base==2.2.5            
numpydoc==1.9.0            
matplotlib==3.10.5           
matplotlib-base==3.10.5           
matplotlib-inline==0.1.6
statsmodels==0.14.5
pmdarima==2.0.4
scikit-learn==1.7.1 
lightgbm==4.6.0
ipython==8.30.0
```

---

## 🙏 Acknowledgment
Developed as part of the TripleTen Data Science Bootcamp, combining classical TS modeling with feature-engineered gradient boosting and proper temporal validation to forecast hourly taxi demand.