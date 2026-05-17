# 🏦 LoanSense: Intelligent Loan Approval Prediction

> *Because every application deserves a fair, data-driven decision.*

Lending institutions process thousands of loan applications daily — each one a balancing act between financial risk and human opportunity. **LoanSense** is an end-to-end machine learning pipeline that cuts through the noise, using applicant demographics and financial indicators to predict loan approval with precision.

---

## What's Under the Hood?

From raw CSV to production-ready predictions, this pipeline covers the full lifecycle:

- 🔍 **Exploratory Data Analysis** — uncovering the signals that actually drive approval
- ⚙️ **Feature Engineering** — translating raw financials into meaningful indicators like DTI, EMI, and Balance Income
- 🤖 **Multi-Model Benchmarking** — Logistic Regression, XGBoost, and Random Forest, head to head
- 🧬 **Custom SMOTE** — synthetic oversampling built from scratch to handle class imbalance fairly
- 🏆 **Ensemble Learning** — Soft Voting and Stacking classifiers for robust, combined predictions
- 🎯 **Hyperparameter Tuning** — Randomized Search over 100 iterations, optimising for ROC-AUC

---

## The Verdict

After rigorous stratified cross-validation, hyperparameter tuning, and held-out test evaluation — the **Stacking Classifier trained on engineered features with SMOTE** emerges as the recommended model: balanced, robust, and built to generalise.

---

## 📁 Project Structure

```
loan-approval-predictor/
├── Dataset/
│   ├── train_data.csv        # Training data (614 samples)
│   └── test_data.csv         # Held-out problem set
├── notebooks/
│   └── loan_approval_predictor.ipynb
├── outputs/
│   └── predicted_engineered.csv   # Generated after inference run
├── requirements.txt
└── .gitignore
```

---

## ⚡ Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/loan-approval-predictor.git
cd loan-approval-predictor
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your data

Place your CSVs in the `Dataset/` folder:
```
Dataset/train_data.csv
Dataset/test_data.csv
```

### 4. Run the notebook

```bash
jupyter notebook notebooks/loan_approval_predictor.ipynb
```

Run all cells top to bottom. The pipeline will automatically detect whether `test_data.csv` has a `Loan_Status` column:

- **If yes** → Diagnostic mode: prints metrics and confusion matrices
- **If no** → Inference mode: generates `outputs/predicted_engineered.csv`

---

## 🧪 Pipeline Stages

| Stage | Details |
|---|---|
| Preprocessing | Mode/median imputation, `3+` fix, target encoding |
| EDA | Class balance, distributions, correlation heatmap |
| Feature Engineering | DTI, EMI, BalanceIncome, log transforms |
| Cross-Validation | Stratified 5-fold on baseline + engineered datasets |
| Hyperparameter Tuning | RandomizedSearchCV, 100 iterations, ROC-AUC objective |
| Class Imbalance | Custom SMOTE (fold-level, no leakage) |
| Ensembles | Soft Voting + Stacking (LR meta-learner) |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC + confusion matrices |

---

## 📊 Features Used

**Raw features:** Gender, Married, Dependents, Education, Self\_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan\_Amount\_Term, Credit\_History, Property\_Area

**Engineered features:**

| Feature | Formula |
|---|---|
| `TotalIncome` | ApplicantIncome + CoapplicantIncome |
| `DebtToIncome` | (LoanAmount × 1000) / TotalIncome |
| `EMI` | LoanAmount / Loan\_Amount\_Term |
| `BalanceIncome` | TotalIncome − (EMI × 1000) |
| `*_log` | log1p transform on income + loan amount |

---

## 🔑 Key Findings

- `Credit_History` is by far the strongest predictor — consistent with real-world credit risk
- Feature engineering improved ROC-AUC and F1 over the raw baseline across all models
- SMOTE improved recall on rejected (minority) class with a modest precision trade-off
- Stacking outperformed Soft Voting, particularly on recall and ROC-AUC
- XGBoost was the strongest individual model

---

## 📦 Requirements

```
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost, scipy
jupyter, notebook
```

Install all with: `pip install -r requirements.txt`

---

