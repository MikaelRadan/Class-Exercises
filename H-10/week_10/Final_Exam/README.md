# 🏛️ Corporate Bankruptcy Risk Intelligence

**Project Goal:** Predicting corporate bankruptcy from financial and accounting ratios using a strictly leakage-free Machine Learning pipeline, and deploying the final model as an interactive Streamlit app.

This repository has two parts:

- **Part 1:** the machine learning notebook (data, modeling, evaluation, limitations)
- **Part 2:** the Streamlit web app built on top of the saved model

---

# Part 1: Machine Learning Notebook

## 📋 1. Dataset & Problem Overview

* **Dataset:** Taiwanese Bankruptcy Prediction (Taiwan Stock Exchange, 1999–2009)
* **Samples & Features:** 6,819 companies across 95 raw financial features (94 active features after dropping the zero-variance column `Net Income Flag`) and 1 binary target (`Bankrupt?`).
* **Missing Values:** 0 missing (NaN) values.
* **Class Distribution:** 6,599 Healthy companies (96.77%) vs. 220 Bankrupt companies (3.23%), an extreme class imbalance.
* **Evaluation Strategy:** Standard Accuracy is misleading because a naive model predicting 0 (Healthy) for every company achieves 96.77% Accuracy while detecting 0% of bankruptcies (Recall = 0.0000). Therefore, the pipeline is optimized and evaluated using **Recall, F1-Score, and ROC-AUC**.

---

## 🗺️ 2. Methodology & The 10-Step Workflow

1. **Data Hygiene & Quarantining the Test Set (Phase 1, Cells 1–3):**
   * Stripped whitespace anomalies from column names and removed the constant feature `Net Income Flag`.
   * Quarantined 20% of the dataset (X_test = 1,364 companies, X_train = 5,455 companies) using stratified splitting (`stratify=y`, `random_state=13`) prior to any exploratory analysis to guarantee zero Data Leakage.
2. **Exploratory Data Analysis on Train Set Only (Phase 2, Cells 4–7):**
   * **Proving the Scaling Trap:** Identified 23 financial features with massive raw magnitudes (up to $10^{10}$) compared to normalized ratios (0 to 1), proving why `StandardScaler` inside a Pipeline is mandatory.
   * **Correlation & Outlier Analysis:** Found profitability metrics (Net Income to Total Assets, ROA) to be the strongest negative (health) indicators, and debt metrics (Debt ratio %, Borrowing dependency) to be the strongest positive (risk) indicators.
   * **Dimensionality Analysis (PCA):** Demonstrated that 52 principal components capture 95% of total variance across the 94 features, confirming high multicollinearity among accounting ratios.
3. **Baseline, Leakage-Free Pipeline & Tuning (Phase 3, Cells 8–12):**
   * **Baseline:** A default unbalanced `LogisticRegression` pipeline achieved 0.8815 ROC-AUC but caught only 21.03% of bankruptcies (Recall = 0.2103).
   * **5-Fold Cross-Validation Comparison (`class_weight='balanced'`):**

| Model | Recall Mean | Recall Std | F1-Score Mean | ROC-AUC Mean |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7563 | 0.0653 | 0.2675 | 0.8671 |
| KNN (n_neighbors=5) | 0.1986 | 0.0712 | 0.2953 | 0.7572 |
| Decision Tree (max_depth=5) | 0.6935 | 0.0692 | 0.2983 | 0.7555 |
| Random Forest (max_depth=8) | 0.6365 | 0.0828 | 0.4028 | 0.9321 |

   * **Hyperparameter Tuning (GridSearchCV):** Tuned the Random Forest pipeline (`cv=5`, `scoring='roc_auc'`), achieving a best CV ROC-AUC of **0.9349** with `n_estimators=200`, `max_depth=12`, and `min_samples_leaf=5`.
   * **Threshold Calibration:** Compared thresholds from 0.20 to 0.55 using out-of-fold probabilities (`cross_val_predict`, 5 folds) on the training set, and chose **0.40** (out-of-fold Recall = 0.69, Precision = 0.26) as a deliberate trade-off that favors catching bankruptcies over avoiding false alarms.

---

## 🏆 3. Final Test Set Results (Unlocked Once, Phase 4, Cells 13 & 14)

Evaluating the locked pipeline on the untouched test set (X_test = 1,364 companies, including 44 bankruptcies) at threshold = 0.40:

* **Final Test ROC-AUC: 0.9551** (CV ROC-AUC was 0.9349; the two are consistent, but with only 44 bankruptcies in the test set these estimates carry notable variance)
* **Bankrupt Class (1) Recall: 0.80 (79.55%)**: successfully caught 35 out of 44 bankruptcies, missing 9; the out-of-fold estimate at the same threshold was 0.69
* **Bankrupt Class (1) F1-Score: 0.52** (Precision = 0.38)
* **Overall Test Accuracy: 0.95** (TN = 1,264, FP = 56, FN = 9, TP = 35)

---

## ⚠️ 4. Error Analysis & Honest Model Limitations

1. **Why False Negatives Occur (Sudden Distress):** Inspecting the lowest-probability missed bankruptcies (9 in total) shows firms that still looked healthy in their last reported ratios (Net Income to Total Assets ~ 0.76–0.81, Retained Earnings to Total Assets ~ 0.92–0.94). Accounting ratios alone may not capture sudden macroeconomic shocks, litigation, or off-balance-sheet fraud; this is a hypothesis, since the data cannot confirm the cause.
2. **Precision–Recall Trade-off (False Positives):** Because only 3.23% of firms go bankrupt, calibrating the threshold to 0.40 for high sensitivity (80% Recall) flags 56 healthy companies as at-risk (Precision = 0.38). This model is designed as an early-warning screening tool for financial auditors rather than an automated credit rejection system.
3. **Domain Scope:** Trained on Taiwanese public companies (1999–2009); deploying to other markets or economic regimes requires retraining and drift monitoring.

---

## 🔄 5. What I Changed After the Review

* Encapsulated `StandardScaler` and `RandomForestClassifier` strictly inside a single Pipeline to eliminate any risk of preprocessing data leakage across CV folds.
* Replaced standard default 0.50 classification cutoffs with probability threshold calibration (`chosen_threshold = 0.40`) tailored to asymmetric financial bankruptcy costs.
* Moved threshold selection from training-set probabilities to out-of-fold predictions (`cross_val_predict`): the Random Forest had memorized the training data (Recall = 1.0 at every threshold), so those probabilities could not guide the choice.
* Added explicit False Negative case inspection in Cell 14 to analyze the financial ratios of missed bankruptcies.

---

## 🚀 6. Notebook Files & Quick Start

* `Final Exam 1.ipynb`: complete 16-cell Jupyter Notebook with executed outputs and analytical summaries.
* `taiwan_bankruptcy_pipeline.joblib`: compressed (`compress=3`) production-ready Scikit-Learn pipeline.
* `requirements.txt`: locked environment dependencies (`==`).

```python
import joblib
import pandas as pd

# Load serialized pipeline (Scaler + Tuned Random Forest)
pipeline = joblib.load("taiwan_bankruptcy_pipeline.joblib")

# Predict bankruptcy risk score with the chosen threshold (0.40)
scores = pipeline.predict_proba(X_new)[:, 1]
predictions = (scores >= 0.40).astype(int)
```

---

# Part 2: Streamlit App

![App screenshot](screenshot.png)

`App.py` is a web interface that loads the saved pipeline from Part 1 and scores companies interactively. The output is a **risk score** between 0 and 1 (the Random Forest uses `class_weight='balanced'` and is not probability-calibrated), compared against an adjustable decision threshold (default 0.40).

## Features

The sidebar lets you choose between three modes:

* **Batch audit (CSV upload):** upload a CSV with financial ratios for multiple companies, edit the key ratios live in a table, and download the audit report. Column names are cleaned automatically and the 94 features are aligned to the training order. If your file is missing some features, they are filled with dataset medians and the app shows a warning with the number of missing features.
* **Scenario stress testing:** adjust the 6 key drivers with sliders (Net Income to Total Assets, Retained Earnings to Total Assets, Continuous interest rate, Debt ratio %, Borrowing dependency, ROA) while all other ratios stay at their median, and see how the risk score changes.
* **Historical company inspector:** pick a company from the dataset by row index and compare the model's prediction with the real historical label.

The **Audit Decision Threshold** slider in the sidebar trades off Recall against false alarms: a lower threshold catches more distressed firms but raises more false alarms.

## App files

```
├── App.py                              # Streamlit app
├── taiwan_bankruptcy_pipeline.joblib   # trained pipeline (from Part 1)
├── Taiwanese Bankruptcy.csv            # reference dataset (medians, slider ranges, historical mode)
├── requirements.txt
└── screenshot.png
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run App.py
```

## App limitations

* Sample rows and the historical inspector use records from the original dataset, which may have been part of the training data, so a correct prediction there is not proof of generalization.
* Filling missing features with medians can make results unreliable.
* Decision-support tool only; it must not replace professional credit analysis.

## Tech stack

Python • pandas • NumPy • scikit-learn • Streamlit • joblib

---

## 🌐 Live Web Application & Interactive Demo

This project is deployed on Streamlit Community Cloud and is accessible online without requiring local setup or environment configuration:

🔗 Direct Web Application Link:  
👉 [Corporate Bankruptcy Risk Intelligence Platform](https://bankruptcy-prediction-mrhk.streamlit.app/)

### 🚀 Key Dashboard Features:
* Batch Audit & Multi-Firm Screening: Upload corporate financial records via CSV for immediate, automated risk scoring across multiple entities.
* Scenario Stress Testing: Perform sensitivity analysis by dynamically tuning key financial ratios via sliders to evaluate immediate risk score variations.
* Policy & Threshold Calibration: Adjust the classification decision threshold (calibrated at 0.40) to prioritize distressed firm detection (high Recall) against false alarm tolerances.
* Automated Inference Pipeline: Full end-to-end integration covering preprocessing, feature scaling, and calibrated Random Forest scoring.