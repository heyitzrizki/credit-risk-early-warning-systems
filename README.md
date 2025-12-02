# 🏦 Credit Risk Early Warning System (PD, LGD, EL)

**Author:** Rizki Anwar Syaifullah  
**Description:** End-to-end machine learning system to predict loan default risk using the US SBA Loan Dataset.  
The project covers PD (Probability of Default), LGD (Loss Given Default), EL (Expected Loss), model tuning, calibration, and Streamlit deployment.

---

## 📌 Features

### 🔹 1. Probability of Default (PD) Model  
- XGBoost (tuned) inside sklearn Pipeline  
- Metrics: ROC-AUC ~0.97, PR-AUC ~0.89, KS ~0.83  
- Includes probability calibration (sigmoid)

### 🔹 2. Loss Given Default (LGD) Model  
- XGBoost Regressor  
- Metrics: RMSE, MAE, R² (~0.59)

### 🔹 3. Expected Loss (EL)  
EL = PD × LGD × EAD


### 🔹 4. Macro Stress Testing  
Uses VIX, Treasury yields, and S&P500 returns to adjust risk levels.

---

## 📂 Repository Structure
credit-risk-ews/
│
├── models/
│   ├── PD_model_tuned_pipeline.pkl
│   ├── PD_model_calibrated_pipeline.pkl
│   └── LGD_model_pipeline.pkl
│
├── notebook/
│   └── Credit_Risk_Early_Warning_System.ipynb
│
├── app/
│   ├── streamlit_app.py
│   └── utils.py
│
├── sample_data/
│   ├── SBA_sample_20rows.csv
│   └── example_input.csv
│
├── requirements.txt
├── README.md
└── .gitignore


---

## 🚀 Streamlit App

### Run locally:
streamlit run app/streamlit_app.py

Features:
- Upload loan data (CSV)
- automatic PD, LGD, and EL scoring
- Downloadable output file


## 🔧 Setup
Install libraries:
pip install -r requirements.txt


## 📬 Contact
Created by Rizki Anwar Syaifullah
For portfolio, research, and educational purposes.
