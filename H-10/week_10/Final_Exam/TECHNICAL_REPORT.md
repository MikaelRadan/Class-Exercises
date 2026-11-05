🏛️ Corporate Bankruptcy Prediction: An End-to-End, Leakage-Free Machine Learning Architecture

Author: Mikail Radan
Domain: Financial Risk Modeling & Applied Machine Learning
Dataset: Taiwan Stock Exchange Corporate Financial Statements (1999–2009)
Primary Stack: Python, Scikit-Learn (1.9.0), Pandas (2.3.3), NumPy (2.3.4), Matplotlib (3.11.1), Seaborn (0.13.2), Joblib (1.5.3)

📌 Executive Summary

Corporate insolvency prediction is a high-stakes financial classification problem characterized by severe class imbalance and asymmetric misclassification costs. In this study, we design, validate, and serialize an end-to-end Machine Learning pipeline to predict corporate bankruptcy across 6,819 publicly traded Taiwanese companies using 95 financial and accounting ratios.

By enforcing strict data quarantine protocols prior to exploratory analysis, encapsulating feature standardization within cross-validated pipelines, and calibrating decision probability thresholds to reflect real-world auditing priorities, our tuned Random Forest architecture achieves a Test ROC-AUC of 0.9551 and identifies 79.55% (35/44) of bankrupt firms on unseen holdout data (the out-of-fold CV estimate at the same threshold is 69.3%), compared with only 21.03% Recall for a standard unbalanced baseline.

🎯 1. Problem Formulation & Evaluation Strategy
1.1 Dataset Architecture & Financial Taxonomy

The dataset comprises 6,819 corporate records and 96 columns (1 binary target variable Bankrupt? and 95 continuous/binary financial attributes derived from corporate balance sheets, income statements, and cash flow reports). To establish domain interpretability, the features are structured into seven core financial pillars:

Financial Domain	Representative Features	Economic & Accounting Significance
1. Target Variable	Bankrupt?	Binary insolvency indicator (0 = Healthy/Active, 1 = Bankrupt).
2. Profitability Ratios	Net Income to Total Assets, ROA(A), ROA(B), ROA(C), Operating Gross Margin	Quantifies capital efficiency and the ability to generate net earnings from assets and revenue.
3. Debt & Solvency	Debt ratio %, Borrowing dependency, Liability to Equity, Interest Coverage Ratio	Measures financial leverage, long-term debt burden, and reliance on external financing.
4. Liquidity & Working Capital	Current Ratio, Quick Ratio, Cash/Total Assets, Working Capital to Total Assets	Evaluates short-term solvency and capacity to cover immediate liabilities with liquid assets.
5. Operational Efficiency	Total Asset Turnover, Accounts Receivable Turnover, Inventory Turnover Rate	Tracks the velocity at which assets, inventory, and receivables convert into operating cash.
6. Cash Flow Metrics	Cash Flow to Total Assets, CFO to Assets, Cash Flow to Sales, Cash Flow to Liability	Assesses actual cash generation from core operations relative to total debt obligations.
7. Growth & Per-Share Metrics	Total Asset Growth Rate, Net Value Growth Rate, Persistent EPS in the Last Four Seasons	Captures multi-period corporate expansion and earnings stability per share.
1.2 The Imbalance Trap & Metric Selection

A distributional audit reveals extreme target skewness:

Healthy Firms (0): 6,599 companies (96.77%)
Bankrupt Firms (1): 220 companies (3.23%)

In such a regime, standard Accuracy is fundamentally flawed. A trivial majority-class classifier that predicts 0 (Healthy) for every observation achieves 96.77% Accuracy while failing to detect a single bankruptcy (0.00% Recall). Furthermore, from a financial risk perspective, a False Negative (failing to detect an impending bankruptcy, resulting in complete principal loss for creditors) is orders of magnitude more costly than a False Positive (flagging a distressed company for manual audit). Consequently, model selection and optimization are governed by three rigorous metrics:

ROC-AUC (Area Under the Receiver Operating Characteristic Curve): Evaluates threshold-independent ranking quality across both classes.
Recall (Sensitivity on Class 1): Quantifies the proportion of actual bankruptcies successfully captured.
F1-Score: Measures the harmonic mean between Precision and Recall on the minority class.
🛡️ 2. Data Governance & Leakage Prevention (Phase 1 — Cells 1 to 3)
Structural Hygiene & Zero-Variance Filtering (Cell 2):
Confirmed zero missing values (NaN = 0) across all 6,819 records.
Sanitized column headers via string whitespace stripping (.str.strip()) to eliminate schema key errors.
Identified and removed the constant zero-variance attribute Net Income Flag (which held a static value of 1 across all rows), yielding 94 active predictive features.
Pre-Analysis Test Set Quarantine (Cell 3):
To prevent Data Leakage, the dataset was partitioned before any statistical profiling, correlation ranking, outlier visualization, or dimensionality reduction.
Using stratified random sampling (train_test_split, test_size=0.20, stratify=y, random_state=13), 20% of the data (X_test = 1,364 firms) was locked in quarantine, while 80% (X_train = 5,455 firms) was allocated for training and cross-validation.
Stratification preserved the exact 3.23% bankruptcy incidence across both partitions (176 bankruptcies in X_train; 44 bankruptcies in X_test).
🔍 3. Exploratory Data Analysis & Econometric Insights (Phase 2 — Cells 4 to 7)

All exploratory analyses were restricted exclusively to the training partition (X_train = 5,455 samples):

3.1 Empirical Proof of the "Scaling Trap" (Cell 4)

Statistical profiling (X_train.describe()) exposed severe magnitude heterogeneity across the 94 features:

Minimum feature value: 0.00
Maximum feature value: 10,000,000,000.00 ($10^{10}$)
While the majority of accounting ratios are normalized within [0, 1], exactly 23 features (including Operating Expense Rate, Research and development expense rate, Interest-bearing debt interest rate, and Total Asset Growth Rate) exhibit raw magnitudes exceeding 100 and reaching up to $10^{10}$. Feeding unscaled features into regularized linear models or distance-based algorithms (KNN) would cause high-magnitude variables to dominate gradient updates and Euclidean distance calculations. This proves that StandardScaler must be embedded inside our modeling pipeline.
3.2 Linear Correlation Structure & Distributional Outliers (Cells 5 & 6)

Pearson correlation analysis against Bankrupt? (eda_df.corr()) identified the primary financial drivers of corporate survival and distress:

Top 5 Health Indicators (Negative Correlation)	Pearson $r$	Top 5 Risk Indicators (Positive Correlation)	Pearson $r$
Net Income to Total Assets	-0.3206	Debt ratio %	+0.2526
ROA(A) before interest and % after tax	-0.2868	Borrowing dependency	+0.1968
ROA(B) before interest and depreciation after tax	-0.2737	Current Liability to Assets	+0.1966
ROA(C) before interest and depreciation before interest	-0.2589	Liability to Equity	+0.1829
Net worth/Assets	-0.2526	Current Liability to Current Assets	+0.1734

Comparative boxplots (Cell 6) of Net Income to Total Assets and Debt ratio % confirmed distinct median separation between solvent and bankrupt firms while revealing substantial heavy-tailed outliers inherent to financial statement data.

3.3 Multicollinearity & Principal Component Analysis (Cell 7)

Because multiple accounting ratios are derived from overlapping balance sheet components, we evaluated feature redundancy by fitting PCA on standardized training data (X_train_scaled). Cumulative explained variance analysis (np.cumsum) demonstrated that 52 principal components capture at least 95% of the total variance across all 94 original features, confirming strong multicollinearity.

⚙️ 4. Pipeline Architecture, Cross-Validation & Tuning (Phase 3 — Cells 8 to 12)
4.1 Baseline Benchmarking (Cell 8)

To establish a rigorous lower bound on X_train, two baseline paradigms were evaluated:

Naive Majority Predictor (Always Predicting 0): Achieved 0.9677 Accuracy and 0.0000 Recall.
Unbalanced Logistic Regression Pipeline (StandardScaler + LogisticRegression): Evaluated via 5-Fold Cross-Validation (cv=5), achieving 0.9666 Accuracy, 0.8815 ROC-AUC, 0.2838 F1-Score, and a 0.2103 Recall (failing to detect nearly 79% of bankruptcies).
4.2 Leakage-Free Pipeline & Multi-Algorithm Comparison (Cells 9 & 10)

To prevent validation leakage during 5-Fold Cross-Validation, StandardScaler was encapsulated with each candidate estimator inside a Scikit-Learn Pipeline, ensuring scaling parameters ($\mu, \sigma$) were computed strictly within each training fold. Cost-sensitive learning (class_weight='balanced') was applied to linear and tree-based models to penalize minority-class errors inversely proportional to class frequencies:

Candidate Pipeline (5-Fold CV on X_train)	Recall (Mean)	Recall (Std)	F1-Score (Mean)	ROC-AUC (Mean)
Logistic Regression (class_weight='balanced')	0.7563	0.0653	0.2675	0.8671
K-Nearest Neighbors (n_neighbors=5)	0.1986	0.0712	0.2953	0.7572
Decision Tree (max_depth=5, class_weight='balanced')	0.6935	0.0692	0.2983	0.7555
Random Forest (n_estimators=100, max_depth=8, balanced)	0.6365	0.0828	0.4028	0.9321

Analytical Selection: Balanced Logistic Regression had the highest raw sensitivity (0.7563) but a low F1-Score (0.2675), meaning many false alarms for each bankruptcy caught. KNN detected few bankruptcies (Recall = 0.1986), which is plausibly due to distance-based methods struggling with high-dimensional, imbalanced data. Random Forest achieved the highest ranking quality (ROC-AUC = 0.9321) and the best F1-Score (0.4028), although not the highest Recall; it was selected because it gave the best overall balance across the three metrics.

4.3 Hyperparameter Optimization via GridSearchCV (Cell 11)

We optimized the Random Forest pipeline across a structured hyperparameter grid (n_estimators: [100, 200], max_depth: [5, 8, 12], min_samples_leaf: [2, 5]) using 5-Fold Cross-Validation (scoring='roc_auc'):

Optimal Architecture: n_estimators = 200, max_depth = 12, min_samples_leaf = 5
Best 5-Fold CV ROC-AUC: 0.9349. Note that the best n_estimators and max_depth sit at the upper edge of the grid, so a wider grid might improve the score slightly.
4.4 Decision Threshold Calibration (Cell 12)

Rather than relying on Scikit-Learn's default probability cutoff (0.50), we compared thresholds from 0.20 to 0.55 using out-of-fold probabilities (cross_val_predict, 5 folds) on the training set, so that each probability came from a model that had not seen that sample. (An earlier version of this analysis used probabilities from a model evaluated on its own training data, where Recall was 1.0 at every threshold; it was replaced because it could not guide the choice.)

As the threshold rises, Recall falls and Precision rises, for example Recall = 0.8409 / Precision = 0.1764 at 0.20, Recall = 0.6932 / Precision = 0.2612 / F1 = 0.3795 at 0.40, and Recall = 0.6364 / Precision = 0.3128 at 0.50. F1-Score alone would favor a higher threshold; we chose chosen_threshold = 0.40 as a deliberate trade-off that favors catching bankruptcies, given that a missed bankruptcy is costlier than a false alarm. The threshold was locked before unlocking the test set.

🏆 5. Final Unseen Test Set Evaluation (Phase 4 — Cell 13)

Unlocking the quarantined test set (X_test = 1,364 firms: 1,320 healthy, 44 bankrupt) for the first and only time using our tuned pipeline and locked threshold (0.40) yielded strong out-of-sample generalization:

Final Test ROC-AUC Score: 0.9551 (5-Fold CV score: 0.9349. The two are consistent, but the test set contains only 44 bankruptcies, so these estimates carry notable variance.)
Class Label	Precision	Recall (Sensitivity)	F1-Score	Support (Test Samples)
0 (Healthy Firms)	0.99	0.96	0.97	1,320
1 (Bankrupt Firms)	0.38	0.80 (79.55%)	0.52	44
Overall Accuracy	—	—	0.95 (95.23%)	1,364
Macro Average	0.69	0.88	0.75	1,364
Weighted Average	0.97	0.95	0.96	1,364
🔬 6. Error Analysis, Feature Importance & Model Limitations (Cell 14)
6.1 Confusion Matrix Decomposition
True Negatives (TN = 1,264): Healthy firms accurately cleared (95.76% specificity).
True Positives (TP = 35): Bankrupt firms successfully detected (79.55% sensitivity).
False Positives (FP = 56): Healthy firms flagged as at-risk.
False Negatives (FN = 9): Bankrupt firms missed by the model.
6.2 Top Financial Drivers (feature_importances_)

Extracting Gini importance from the 200-tree Random Forest ensemble showed that the top three predictive features are Net Income to Total Assets, Continuous interest rate (after tax), and Retained Earnings to Total Assets. Feature importance reflects contribution to the model's splits and does not imply causation; the full top-10 ranking is shown in the Cell 14 plot.

6.3 Case-Level Inspection of Worst False Negatives & Honest Limitations

To understand why the model missed 9 bankruptcies, we isolated the False Negative cases with the lowest predicted bankruptcy probabilities:

Sample Index	Retained Earnings to Total Assets	Continuous interest rate (after tax)	Net Income to Total Assets	Actual Class	Predicted Probability	Predicted Class
2577	0.937341	0.781641	0.813367	1	0.0495	0
3114	0.930413	0.781627	0.802265	1	0.0561	0
1212	0.935400	0.781638	0.807311	1	0.0769	0
5897	0.934635	0.781524	0.789536	1	0.1009	0
1213	0.920564	0.781293	0.762491	1	0.2834	0

Key Domain Limitations:

Healthy-Looking Ratios Before Bankruptcy (False Negatives): The lowest-probability missed cases (e.g., records 2577 and 3114) had high Retained Earnings to Total Assets (~0.93) and high Net Income to Total Assets (> 0.80) in their final reported statements, so their ratios looked healthy. A plausible explanation is that financial ratios alone cannot capture sudden macroeconomic shocks, fraud, or off-balance-sheet liabilities, but this is a hypothesis that the data cannot confirm.
The Precision–Recall Trade-off in Rare-Event Screening (False Positives): Because bankruptcies represent only 3.23% of the population, calibrating the model for high sensitivity (79.55% Recall) inevitably produces false alarms (FP = 56, Precision = 0.38). Consequently, this pipeline should be deployed as an automated early-warning triage system to prioritize human auditing resources, rather than an autonomous decision engine.
Temporal & Jurisdictional Scope: Trained on Taiwanese equities from 1999–2009, deploying this artifact to modern markets requires recalibration and continuous monitoring for macroeconomic Data Drift.
🚀 7. Production Serialization & Reproducibility (Cells 15 & 16)
End-to-End Artifact Serialization (Cell 15):
The complete best_pipeline (encapsulating both StandardScaler and the tuned RandomForestClassifier) was serialized into taiwan_bankruptcy_pipeline.joblib using joblib.dump(..., compress=3).
A post-serialization health check reloaded the artifact and verified consistent probabilities across 5 holdout observations ([0.0000, 0.8003, 0.0769, 0.0041, 0.0353], np.allclose = True).
Locked Environment Dependencies (Cell 16):
Generated a version-locked requirements.txt (numpy==2.3.4, pandas==2.3.3, matplotlib==3.11.1, seaborn==0.13.2, scikit-learn==1.9.0, joblib==1.5.3) to ensure deterministic reproducibility across deployment environments.