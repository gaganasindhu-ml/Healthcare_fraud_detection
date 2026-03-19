# 🛡️ FraudShield — AI-Powered Insurance Fraud Detection

![Model](https://img.shields.io/badge/Model-XGBoost-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-94.2%25-blue)
![Features](https://img.shields.io/badge/Features-17-orange)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-purple)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

> **Detecting fraudulent health insurance claims using machine learning — reducing losses, protecting patients, and making insurance fair.**

---

## 🚨 The Problem

India's health insurance sector loses over **$1 billion every year** to fraudulent claims. Hospitals inflate bills, procedures get billed that never happened, and manual review is slow, expensive, and inconsistent.

- Health insurers witness loss ratios of **80–100%** — some above 100%
- A human claim reviewer processes **~50 claims/day** with inconsistent accuracy  
- Fraud patterns are complex — no single rule catches them all

**FraudShield** uses machine learning to score every claim instantly — flagging suspicious ones before they're paid.

---

## 🎯 What It Does

Given details about an insurance claim — billing amount, hospital stay, patient profile, provider history, chronic conditions — FraudShield outputs:

- **A fraud probability score** (0–100%)
- **A risk level** — Normal / Suspicious / High Risk
- **A feature risk profile** — exactly which factors drove the score
- **A claim summary** — structured view of all inputs

No black box. Every prediction is explainable.

---

## 🖥️ Demo Screenshots

| FraudShield loads the trained XGBoost model and shows model stats upfront — 94.2% accuracy, 17 features, binary classification task. The sidebar takes structured claim input. After clicking Analyze, the gauge shows fraud probability, the Feature Risk Profile shows which inputs drove the score, and the Claim Summary table gives a clean structured view. |

**Live App:** *(https://healthcarefrauddetection-dmqypx4ed7ukcnqztjyair.streamlit.app/)

### Home Screen — Model Loaded
![FraudShield Home](https://github.com/gaganasindhu-ml/Healthcare_fraud_detection/blob/main/Screenshot/1.png?raw=true)
### Claim Input Form
![Claim Input](https://github.com/gaganasindhu-ml/Healthcare_fraud_detection/blob/main/Screenshot/2.png?raw=true)
(https://github.com/gaganasindhu-ml/Healthcare_fraud_detection/blob/main/Screenshot/3.png?raw=true)
(https://github.com/gaganasindhu-ml/Healthcare_fraud_detection/blob/main/Screenshot/4.png?raw=true)
### Fraud Analysis Result
![Analysis Result](https://github.com/gaganasindhu-ml/Healthcare_fraud_detection/blob/main/Screenshot/5.png?raw=true)

### Claim summary
![Feature Risk](https://github.com/gaganasindhu-ml/Healthcare_fraud_detection/blob/main/Screenshot/6.png?raw=true)
```
---

## 📊 Dataset

**Source:** [Healthcare Provider Fraud Detection Analysis — Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

The dataset contains real Medicare claims data across 4 files:

| File | Contents |
|------|----------|
| `Train_Inpatientdata.csv` | Hospital admission claims with dates, diagnosis codes, billing amounts |
| `Train_Outpatientdata.csv` | Outpatient visit claims |
| `Train_Beneficiarydata.csv` | Patient demographics, chronic conditions, insurance details |
| `Train-1542865627584.csv` | Provider-level fraud labels (Yes / No) |

**After merging and feature engineering:**
- ~500,000+ claim records
- 17 engineered features
- Binary fraud label at provider level

---

## ⚙️ Feature Engineering

Raw data alone isn't enough. The model learns from **engineered features** that encode real-world fraud patterns:

| Feature | Why It Matters |
|---------|----------------|
| `claimDuration` | Unusually long claim periods indicate inflated billing |
| `HospitalStayDays` | Short stays with high billing = suspicious |
| `PatientAge` | Calculated from DOB — older patients have more complex legitimate claims |
| `InscClaimAmtReimbursed` | Raw billing amount — high outliers flagged |
| `DeductibleAmtPaid` | Low deductibles on high claims = red flag |
| `claim_type` | Inpatient (1) vs outpatient (0) — different fraud patterns per type |
| `NumDiagnosisCodes` | Too many diagnoses on one claim = suspicious |
| `NumProcedureCodes` | Unusually high procedure counts per visit |
| `ChronicConditionCount` | Fraudsters inflate chronic condition counts to justify procedures |
| `ProviderAvgClaim` | How this provider's billing compares to their own historical average |
| `ProviderFraudRatio` | Historical fraud rate of this specific provider |
| `UniquePhysicians` | Multiple doctors on one claim = potential fraud ring signal |

---

## 🧠 Model Architecture

```
Raw Claims Data (4 CSV files from Kaggle)
        ↓
Merge & Join
(inpatient + outpatient + beneficiary + provider labels)
        ↓
Feature Engineering
(claimDuration, HospitalStayDays, PatientAge, provider aggregates)
        ↓
Preprocessing
(Label Encoding for categoricals, drop ID/leaky columns)
        ↓
Train/Test Split — 80/20, stratified by fraud label
        ↓
Median Imputation — fitted on train only (no leakage)
        ↓
SMOTE — applied to training set only to handle class imbalance
        ↓
XGBoost Classifier
(n_estimators=100, max_depth=6, learning_rate=0.1)
        ↓
Evaluation
(Accuracy, ROC-AUC, Confusion Matrix, Classification Report, SHAP)
        ↓
Export → fraud_detection_model.pkl
        ↓
Streamlit Dashboard — FraudShield UI
```

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.2% |
| **Model Type** | XGBoost Binary Classifier |
| **Features Used** | 17 |
| **Class Imbalance Handling** | SMOTE (training set only) |
| **Imputation Strategy** | Median (fit on train, transform test) |

> ⚠️ **Important Note on Metrics:** For fraud detection, **Recall on the fraud class** matters more than overall accuracy. A model that misses real fraud is more costly than one that occasionally flags a legitimate claim. See the full classification report in `insurance_model.ipynb` for per-class precision, recall, and F1.

---

## 🗂️ Project Structure

```
FraudShield/
│
├── 📓 insurance_eda.ipynb           # Exploratory Data Analysis
│   ├── Dataset loading & merging
│   ├── Date column engineering
│   ├── Fraud vs legitimate comparisons
│   ├── Billing pattern analysis
│   ├── Diagnosis & procedure code distributions
│   └── Physician frequency analysis
│
├── 📓 insurance_model.ipynb         # Model Training Pipeline  
│   ├── Feature engineering
│   ├── Column cleaning (drop IDs, leaky features, high-cardinality codes)
│   ├── Encoding & imputation
│   ├── Stratified train/test split
│   ├── SMOTE oversampling
│   ├── XGBoost training
│   ├── Confusion matrix & classification report
│   ├── ROC curve
│   ├── SHAP feature importance (global + per-claim)
│   ├── Estimated financial savings calculation
│   └── Model export with joblib
│
├── 🤖 fraud_detection_model.pkl     # Saved trained XGBoost model
│
├── 🖥️ app.py                        # Streamlit dashboard (FraudShield UI)
│
├── 📄 requirements.txt              # Python dependencies
│
└── 📄 README.md                     # This file
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fraudshield.git
cd fraudshield
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
xgboost
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
seaborn
shap
joblib
plotly
```

Install all:
```bash
pip install streamlit xgboost scikit-learn imbalanced-learn pandas numpy matplotlib seaborn shap joblib plotly
```

---

## 🔍 How to Use the App

**Step 1 — Claim Details** (sidebar top)
Fill in claim amount, claim type (inpatient/outpatient), claim duration in days, hospital stay in days.

**Step 2 — Patient Info**
Enter patient age, gender, race group, number of diagnosis codes, number of procedure codes on the claim.

**Step 3 — Chronic Conditions**
Check any chronic conditions present: Heart Failure, Alzheimer's, Cancer, COPD, Diabetes, Depression, Ischemic Heart, Osteoporosis, Kidney Disease.

**Step 4 — Provider Info**
Enter total claims this provider has submitted, unique doctors used, provider's average claim amount, and provider's historical fraud ratio.

**Step 5 — Click "Analyze for Fraud"**
Instantly see:
- Fraud Probability Score with gauge visualization
- Fraud % vs Legitimate % breakdown
- Risk level badge (Normal / Suspicious / High Risk)
- Feature Risk Profile — which inputs drove the score most
- Claim Summary table — all entered values in one view

---

## 💡 Key Design Decisions & Why They Matter

**Why XGBoost and not a neural network?**
XGBoost handles tabular data better than deep learning in most real-world scenarios with limited data. It's also faster to train, easier to debug, and integrates directly with SHAP for explainability. Insurance companies need to explain every flagged claim to comply with regulations — a black-box neural network fails that requirement.

**Why SMOTE only on training data?**
Applying SMOTE before the train/test split would cause data leakage — the model would have seen information derived from test samples during training, producing artificially inflated performance metrics. SMOTE is strictly fitted and applied to training data only.

**Why drop diagnosis and procedure code columns?**
Raw ICD codes and CPT procedure codes have thousands of unique values (high cardinality). Without careful encoding they cause overfitting and make the model brittle. The key fraud signal they carry is better captured through engineered aggregate features like NumDiagnosisCodes and NumProcedureCodes.

**Why median imputation?**
HospitalStayDays is null for all outpatient claims — outpatients have no hospital admission. Filling with 0 would be semantically wrong. Median imputation preserves the realistic distribution. Crucially, the imputer is fitted only on training data to prevent any test set information from influencing preprocessing.

---

## 🌍 Business Impact

From model evaluation:
```
Frauds detected in test set    →  calculated in notebook
Average fraudulent claim amount →  calculated in notebook  
Estimated financial loss prevented → detected_frauds × avg_claim_amount
```

At production scale — processing 10,000 claims per month — FraudShield can:
- **Reduce manual review workload by ~60–70%** by auto-approving low-risk claims
- **Flag high-risk claims** before payment for targeted human review
- **Catch more fraud** than manual reviewers who miss complex multi-feature patterns
- **Justify every flag** with explainable feature contributions (regulatory requirement)

---

## 🔮 Roadmap

- [ ] SHAP waterfall plot per prediction (per-claim deep explanation)
- [ ] Integrate Medical Document Parser — validate claim documents against structured data
- [ ] Add hospital-level billing anomaly detection (TPA Intelligence Layer)
- [ ] Retrain on Indian insurer data with India-specific features (regional diagnosis patterns, TPA codes)
- [ ] FastAPI backend for direct integration into insurer claim management systems
- [ ] Model performance monitoring and drift detection dashboard
- [ ] Batch processing mode — upload CSV of 1000 claims, get scored output

---

## 👤 About

**[Your Name]**  
Data Scientist building at the intersection of AI and Insurance

| | |
|--|--|
| LinkedIn | [your-linkedin-url] |
| GitHub | [your-github-url] |
| Email | your@email.com |

---

## 📄 License

MIT License — free to use, modify, and build on.

---

## 🙏 Acknowledgements

- **Dataset:** [Rohit Anand Gupta — Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
- **SHAP** by Scott Lundberg — model explainability
- **Streamlit** — rapid ML dashboard development  
- **XGBoost** by Tianqi Chen — gradient boosting framework
- **imbalanced-learn** — SMOTE implementation

---

*"Lower loss ratios → lower premiums → higher penetration → superior customer experience"*  
*Building the AI infrastructure layer that makes insurance work better for everyone.*
