# Corporate Bankruptcy Risk Predictor - MVP Structure Reference

## 1. Project Overview

- **Product Name**: Corporate Bankruptcy Risk Predictor
- **Type**: Web-based MVP (Minimum Viable Product)
- **Purpose**: Predict corporate bankruptcy risk from financial indicators and provide actionable, AI-generated improvement recommendations
- **Interface**: Streamlit web application
- **Target Users**: Banks (credit risk teams), Auditors (Big 4 firms), CFOs/Finance teams, Investors/Analysts

---

## 2. Dataset

- **Source**: American Bankruptcy dataset
- **Structure**: Panel data — multiple years per company (avg ~8.7 years/company, 92% have ≥2 years)
- **Size**: 8,971 unique companies (most recent year per company used to avoid data leakage)
- **Target Variable**: `status_label` - "failed" (bankrupt) vs "alive" (healthy)
- **Class Distribution**: 6.79% bankruptcy rate (highly imbalanced)
- **Raw Features**: 18 financial indicators (X1-X18):

| Feature | Name |
|---------|------|
| X1 | Current Assets |
| X2 | Cost of Goods Sold |
| X3 | Depreciation and Amortization |
| X4 | EBITDA |
| X5 | Inventory |
| X6 | Net Income |
| X7 | Total Receivables |
| X8 | Market Value |
| X9 | Net Sales |
| X10 | Total Assets |
| X11 | Total Long-term Debt |
| X12 | EBIT |
| X13 | Gross Profit |
| X14 | Total Current Liabilities |
| X15 | Retained Earnings |
| X16 | Total Revenue |
| X17 | Total Liabilities |
| X18 | Total Operating Expenses |

---

## 3. Feature Engineering

### 3.1 Problem: Company Size Bias

Raw financial indicators (X1-X18) are absolute dollar amounts. A large company's "Total Assets = $500K" could signal distress, while the same value for a small startup is healthy. Without normalization, models confuse company size with financial health.

### 3.2 Solution: Three-Layer Feature Engineering

The final model uses **31 features** (18 raw + 8 ratios + 5 YoY):

#### Layer 1: StandardScaler on Raw Features (X1-X18)

- **Method**: `sklearn.StandardScaler` — zero-mean, unit-variance normalization
- **Scope**: Applied only to X1-X18 (raw dollar amounts)
- **Purpose**: Remove absolute scale differences so the model compares relative positions within the training distribution
- **Fit on**: Training set only (to prevent data leakage); test set and app inputs use the same fitted scaler

#### Layer 2: Financial Ratios (R1-R8)

8 domain-driven ratios that are inherently scale-invariant (no scaler needed):

| Ratio | Name | Formula | Financial Meaning |
|-------|------|---------|-------------------|
| R1 | Current Ratio | X1 / X14 | Liquidity — ability to cover short-term obligations |
| R2 | Debt-to-Asset Ratio | X17 / X10 | Leverage — what % of assets are financed by debt |
| R3 | Net Profit Margin | X6 / X16 | Profitability — how much of revenue becomes profit |
| R4 | EBITDA Margin | X4 / X16 | Operating efficiency — earnings before non-cash charges |
| R5 | Gross Margin | X13 / X16 | Core profitability — revenue minus cost of goods |
| R6 | Debt-to-Equity Ratio | X17 / (X10-X17) | Capital structure — debt relative to shareholder equity |
| R7 | Asset Turnover | X9 / X10 | Efficiency — how much revenue each dollar of assets generates |
| R8 | Retained Earnings / Assets | X15 / X10 | Cumulative profitability — historical profit retention |

- **Safe division**: Custom `safe_divide()` function handles zero/NaN denominators (returns 0)
- **Clipping**: All ratios clipped to [-10, 10] to cap extreme outliers

#### Layer 3: Year-over-Year (YoY) Change Rates (5 features)

Leverages panel data structure to capture financial trajectory (improving vs deteriorating):

| Feature | Name | Formula |
|---------|------|---------|
| YoY_X4 | YoY EBITDA | (current - prev) / |prev| |
| YoY_X6 | YoY Net Income | (current - prev) / |prev| |
| YoY_X10 | YoY Total Assets | (current - prev) / |prev| |
| YoY_X16 | YoY Total Revenue | (current - prev) / |prev| |
| YoY_X17 | YoY Total Liabilities | (current - prev) / |prev| |

- **Why only 5 (not 18)**: Testing showed full 18-feature YoY hurt performance (ROC-AUC dropped from 0.8186 to 0.8092). These 5 key indicators capture the most important financial trends.
- **Training**: Computed from panel data via `groupby("company_name").shift(1)`. First year per company filled with 0.
- **App (runtime)**: User optionally provides previous year values for these 5 indicators. If not provided, YoY features default to 0.
- **Clipping**: All YoY values clipped to [-10, 10]

---

## 4. System Architecture

### 4.1 Architecture Layers

```
Layer 1: DATA SOURCES
  - American Bankruptcy CSV (8,971 companies, 18 raw features → 31 engineered features)
  - 20 Industry Outlook PDFs (Deloitte + KPMG)
  - User Input (18 financial indicators + 5 optional prev-year indicators + industry selection)
  - OpenAI API (GPT-4o + text-embedding-3-small)

Layer 2: PROCESSING
  - train_model.py: Feature engineering + model training pipeline
    → StandardScaler (X1-X18) + Financial Ratios (R1-R8) + YoY trends (5 features)
  - build_vectordb.py: RAG vector store construction
  - app.py: Real-time prediction + SHAP + ensemble + counterfactual + LLM generation

Layer 3: MODEL & STORAGE
  - ML Models: XGBoost, Random Forest, LightGBM (.pkl files)
  - StandardScaler: Fitted on training data (.pkl)
  - Optimal Threshold: F2-score optimized (.pkl)
  - ChromaDB: Vector store for industry report chunks
  - SHAP TreeExplainer: Feature attribution engine
  - DiCE Engine: Counterfactual scenario generator

Layer 4: APPLICATION
  - Streamlit Web App (app.py)
  - Features: Risk Score, SHAP Analysis (31 features), Model Ensemble, What-If Analysis, AI Recommendations, 8 Languages
```

### 4.2 Data Flow (End-to-End)

1. **Training (offline)**: `train_model.py` loads panel CSV → sorts by company+year → computes YoY change rates → takes most recent year per company → computes financial ratios → StandardScaler on raw features → splits 80/20 (stratified) → trains XGBoost/RF/LightGBM with boosted class weights → F2-score threshold optimization → saves `.pkl` artifacts
2. **RAG Indexing (offline)**: `build_vectordb.py` extracts text from 20 PDFs via PyMuPDF → chunks (1000 chars, 200 overlap) → embeds via OpenAI → stores in ChromaDB with industry metadata
3. **User Input (runtime)**: User enters 18 financial indicators + optionally 5 previous-year indicators + selects industry in Streamlit sidebar
4. **Feature Engineering (runtime)**: App computes 8 financial ratios + 5 YoY features (or fills 0) from user inputs → StandardScaler transforms raw features
5. **Prediction (runtime)**: XGBoost outputs bankruptcy probability → displayed as risk score (0-100)
6. **Explainability (runtime)**: SHAP TreeExplainer computes feature attributions for all 31 features → top 5 risk factors shown
7. **Ensemble (runtime)**: RF and LightGBM also predict → side-by-side comparison + consensus analysis
8. **Counterfactual (runtime, risk >= 40 only)**: DiCE generates 3 scenarios showing minimum changes to reach low risk
9. **AI Recommendations (runtime)**: Top SHAP factors form RAG query → ChromaDB retrieves relevant industry chunks → GPT-4o generates grounded advice in user's language

---

## 5. ML/AI Components

### 5.1 Primary Model: XGBoost Classifier

- **Algorithm**: Gradient Boosted Decision Trees (XGBClassifier)
- **Input Features**: 31 (18 raw scaled + 8 ratios + 5 YoY)
- **Hyperparameters**: max_depth=5, n_estimators=200, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8
- **Class Imbalance Handling**: `scale_pos_weight` = 2× natural class ratio (~27.5) — boosted to prioritize recall (catching bankrupt companies)
- **Threshold Optimization**: F2-score (recall-weighted) grid search over [0.10, 0.60] → optimal threshold = 0.25
- **Performance**: ROC-AUC 0.81, Recall 0.62 (at optimized threshold) on 20% held-out test set
- **Output**: Bankruptcy probability [0, 1], displayed as risk score [0, 100]
- **Risk Thresholds (display)**: High Risk >= 70, Medium Risk >= 40, Low Risk < 40

### 5.2 Baseline Model: Logistic Regression

- **Purpose**: Simple baseline for comparison during training
- **Config**: max_iter=1000, class_weight="balanced"
- **Performance**: ROC-AUC 0.74
- **Note**: Not deployed in the app, used only for training evaluation

### 5.3 Ensemble Models

- **Random Forest**: n_estimators=200, max_depth=10, class_weight="balanced"
- **LightGBM**: n_estimators=200, max_depth=5, learning_rate=0.1, scale_pos_weight (boosted 2×)
- **Consensus Analysis**: Counts how many of the 3 models agree on the risk classification (above/below optimized threshold). Reports agreement ratio and standard deviation across model predictions.
- **Ensemble Average**: Mean probability across all 3 models

### 5.4 SHAP Explainability

- **Method**: TreeExplainer (exact Shapley values for tree-based models, fast computation)
- **Output**: One SHAP value per feature per prediction (31 features total)
- **Interpretation**: Positive SHAP = increases bankruptcy risk, Negative SHAP = decreases risk
- **UI Display**: Top 5 risk factors with direction indicators + full 31-feature impact table
- **Integration**: SHAP values are passed into the GPT-4o prompt so AI recommendations are grounded in the model's actual reasoning

### 5.5 RAG (Retrieval-Augmented Generation)

- **Knowledge Base**: 20 industry outlook PDFs from Deloitte and KPMG
  - Deloitte: Manufacturing, Renewable Energy, Technology (2025); Chemical, Engineering & Construction, Oil & Gas, Power & Utilities, Renewable Energy (2026)
  - KPMG: Retail, Consumer Products, Travel (2026); plus reports on AI in asset management, M&A trends, automotive retailing, cyber resilience, etc.
- **PDF Processing**: PyMuPDF text extraction → clean whitespace → chunk at 1000 characters with 200-character overlap
- **Embedding Model**: OpenAI `text-embedding-3-small` (batch size 20)
- **Vector Store**: ChromaDB PersistentClient, cosine similarity, industry metadata tags
- **Total Chunks Indexed**: ~371+ (varies with PDF count)
- **Retrieval**: Query embedding generated from top 5 SHAP risk factor names + industry keyword → ChromaDB returns top 5 most relevant chunks filtered by industry
- **Strict RAG Usage Rules** (enforced in prompt):
  1. Company's own financial data and SHAP values are PRIMARY
  2. Industry report context is SECONDARY/supplementary only
  3. No recommendation should be based solely on report content
  4. Company financial evidence always overrides industry context if conflicting
  5. Report-only suggestions labeled separately as "Sector-Context Suggestions"

### 5.6 LLM: GPT-4o

- **Role**: Generate personalized financial improvement recommendations
- **Input**: Risk score, top 5 SHAP features (from 31 total) with values and directions, RAG industry context (if available)
- **System Prompt**: "Professional financial advisor specializing in corporate distress and turnaround strategies. Always ground analysis in company-specific financial data first."
- **Output**: Structured response with financial health assessment + 3-5 actionable recommendations + optional sector suggestions
- **Language**: Responds in user's selected language via prompt instruction

### 5.7 Counterfactual What-If Analysis (DiCE)

- **Library**: Microsoft DiCE (Diverse Counterfactual Explanations)
- **Method**: Random sampling from training data distribution
- **Trigger**: Only when risk score >= 40 (medium or high risk)
- **Output**: 3 counterfactual scenarios showing minimum feature changes to flip prediction to "non-bankrupt"
- **Display**: Table per scenario with columns: Feature, Current Value, Target Value, Change (with direction arrow)
- **Training Data**: Uses saved X_train/y_train from model training for realistic counterfactual generation

### 5.8 Multilingual Support (i18n)

- **Languages**: English, Traditional Chinese, Simplified Chinese, Japanese, Korean, Spanish, French, German
- **Implementation**: Dictionary-based lookup (`UI_TEXT` dict with all UI strings per language)
- **AI Output**: GPT-4o prompt includes language instruction, so recommendations are generated natively in the selected language
- **Selector**: Sidebar dropdown, instantly switches all UI labels

---

## 6. Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Frontend | Streamlit | Interactive web UI |
| Primary ML Model | XGBoost | Bankruptcy classification |
| Ensemble Models | Random Forest, LightGBM | Multi-model comparison |
| Feature Scaling | scikit-learn StandardScaler | Normalize raw features (X1-X18) |
| Explainability | SHAP (TreeExplainer) | Feature attribution |
| Counterfactual | DiCE (Microsoft) | What-if scenario generation |
| LLM | OpenAI GPT-4o | AI recommendation generation |
| Embeddings | OpenAI text-embedding-3-small | Document & query embedding for RAG |
| Vector Database | ChromaDB | Persistent vector store for RAG retrieval |
| PDF Parsing | PyMuPDF (fitz) | Extract text from industry report PDFs |
| Data Processing | pandas, NumPy | Data manipulation |
| ML Framework | scikit-learn | Preprocessing, metrics, Random Forest |
| Config | python-dotenv | Environment variable management (.env) |
| Visualization | matplotlib | Chart generation |
| Language | Python 3.10+ | Runtime |

---

## 7. Project File Structure

```
Bankruptcy-Risk/
|-- data/
|   |-- american_bankruptcy.csv       # Training dataset (8,971 companies, panel data)
|   |-- reports/                      # 20 Deloitte & KPMG industry outlook PDFs
|   |-- vectordb/                     # ChromaDB persistent vector store (generated)
|
|-- models/                           # All generated by train_model.py
|   |-- xgb_bankruptcy.pkl            # XGBoost model (31 features)
|   |-- rf_bankruptcy.pkl             # Random Forest model
|   |-- lgbm_bankruptcy.pkl           # LightGBM model
|   |-- scaler.pkl                    # StandardScaler fitted on training X1-X18
|   |-- threshold.pkl                 # Optimal F2-score threshold (0.25)
|   |-- features.pkl                  # Feature name list (31 features)
|   |-- feature_labels.pkl            # Feature label mapping (human-readable names)
|   |-- train_stats.pkl               # Training data for DiCE counterfactuals
|
|-- src/
|   |-- app.py                        # Main Streamlit application (all features)
|   |-- train_model.py                # Feature engineering + model training pipeline
|   |-- build_vectordb.py             # RAG: PDF -> chunks -> embeddings -> ChromaDB
|   |-- generate_report.py            # Product report PDF generator
|   |-- generate_slides.py            # Demo slides PDF generator
|
|-- docs/                             # Generated PDF outputs + reference docs
|-- .env                              # OPENAI_API_KEY (not in git)
|-- requirements.txt                  # Python dependencies
|-- .gitignore
```

---

## 8. Key Design Decisions

1. **XGBoost as primary model**: Best balance of performance (ROC-AUC 0.81) and SHAP compatibility (TreeExplainer provides exact, fast Shapley values for tree models)
2. **Three-layer feature engineering**: Raw features (StandardScaler) + financial ratios (scale-invariant) + YoY trends (temporal) — addresses company size bias while capturing financial health from multiple angles
3. **StandardScaler on raw features only**: Ratios and YoY features are already scale-invariant; applying scaler only where needed
4. **Boosted scale_pos_weight (2× natural ratio)**: Prioritizes recall — in bankruptcy prediction, missing a bankrupt company (false negative) is far costlier than a false alarm (false positive)
5. **F2-score threshold optimization**: Instead of default 0.5, searches for the threshold that maximizes F2-score (recall-weighted), found at 0.25. This shifts the model toward catching more bankrupt companies.
6. **YoY limited to 5 key indicators**: Full 18-feature YoY hurt performance. EBITDA, Net Income, Total Assets, Total Revenue, and Total Liabilities capture the most important financial trajectory signals.
7. **Most recent year per company**: To prevent data leakage from having multiple years of the same company in train/test splits
8. **Strict RAG guardrails**: Prompt engineering ensures AI recommendations are driven by the company's actual financial data, not just report summaries
9. **Counterfactual only for medium/high risk**: No actionable "what to change" is needed for already low-risk companies
10. **Dictionary-based i18n over translation API**: Instant switching, no API latency, consistent UI terminology
11. **ChromaDB with industry metadata filter**: Ensures retrieved chunks are from the correct industry, not just semantically similar text from unrelated sectors

---

## 9. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Train models (generates .pkl files including scaler and threshold)
python src/train_model.py

# 4. Build RAG vector store (requires API key)
python src/build_vectordb.py

# 5. Launch the app
streamlit run src/app.py
```

---

## 10. User Flow

1. User opens the Streamlit app in browser
2. Selects language (sidebar dropdown, 8 options)
3. Selects industry sector (sidebar dropdown, 7 industries + Other)
4. Enters 18 financial indicators (sidebar number inputs)
5. *(Optional)* Expands "Previous Year Data" and enters 5 key indicators (EBITDA, Net Income, Total Assets, Total Revenue, Total Liabilities) for trend analysis
6. Clicks "Predict Bankruptcy Risk"
7. **Risk Score**: Large colored number (0-100) with High/Medium/Low label
8. **SHAP Analysis**: Top 5 risk factors from 31 features + full feature impact table
9. **Model Ensemble**: Side-by-side XGBoost/RF/LightGBM comparison + consensus
10. **What-If Analysis** (if risk >= 40): 3 counterfactual scenarios with minimum changes needed
11. **AI Recommendations**: GPT-4o personalized advice grounded in company data + industry context
12. **Industry Insights Referenced**: Expandable section showing RAG source excerpts

---

## 11. Model Performance Summary

| Model | Role | ROC-AUC | Recall (bankrupt) | Precision (bankrupt) | Threshold |
|-------|------|---------|-------------------|---------------------|-----------|
| Logistic Regression | Baseline (training only) | 0.74 | 0.75 | 0.13 | 0.50 |
| XGBoost | Primary model | 0.81 | 0.62 | 0.21 | 0.25 (F2-optimized) |
| Random Forest | Ensemble member | 0.80 | 0.65 | 0.18 | 0.25 |
| LightGBM | Ensemble member | 0.80 | 0.63 | 0.20 | 0.25 |

- **Test set**: 20% stratified hold-out (random_state=42)
- **Class imbalance handling**: scale_pos_weight = 2× natural ratio (~27.5) for XGBoost/LightGBM; class_weight="balanced" for RF/LR
- **Threshold**: Optimized via F2-score grid search (beta=2, recall-weighted) — prioritizes catching bankrupt companies over precision
- **Feature count**: 31 (18 raw scaled + 8 financial ratios + 5 YoY change rates)
