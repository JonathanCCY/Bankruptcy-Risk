# 📉 Corporate Bankruptcy Risk Predictor

An end-to-end, explainable machine-learning system that predicts corporate bankruptcy risk from financial statements — and goes a step further by **explaining *why*** (SHAP), showing **what to change** to reduce risk (counterfactuals), and generating **industry-grounded improvement advice** via Retrieval-Augmented Generation (RAG).

> Built as a Machine Learning for Business group project, engineered to production-app standards: imbalanced-learning best practices, train/serve feature parity, model explainability, and a multilingual Streamlit front end.

---

## 🎯 Headline Results

On a held-out test set of **1,795 companies** (122 bankruptcies, ~6.8% base rate), iterative modeling lifted the minority-class metrics that actually matter on imbalanced data:

| Stage | PR-AUC ⭐ | ROC-AUC | Recall | Precision | F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| ① Baseline — most-recent-year snapshot (31 features) | 0.321 | 0.806 | 0.623 | 0.213 | 0.317 |
| ② + **Panel time-series feature engineering** (62 features) | 0.382 | **0.841** | 0.631 | 0.252 | 0.360 |
| ③ + **BorderlineSMOTE** oversampling (final) | **0.401** | 0.840 | 0.557 | 0.366 | 0.442 |
| **Cumulative improvement** | **+25%** | +4% | — | **+72%** | **+39%** |

**Why PR-AUC, not accuracy?** With a 93/7 class split, a naïve "always healthy" classifier scores 93% accuracy while catching *zero* bankruptcies. PR-AUC and minority-class recall/precision are the honest metrics here — and they are what we optimized.

The operating threshold (0.31) is tuned via **F2-score** to deliberately favor **recall**: in credit-risk screening, missing a company that fails is far costlier than a false alarm.

---

## 🧠 What Makes This More Than a Notebook

### 1. Panel (multi-year) feature engineering
The raw dataset is **panel data** — 8,971 US companies tracked across 1999–2018 (median 7 years each). A naïve model collapses each company to its latest year and throws the history away.

Instead, each company's full time series is summarized into **trend, volatility, and momentum** features for 11 key indicators, plus company-level meta features:

- `__slope` — multi-year OLS trend (size-normalized for raw $ figures)
- `__vol` — volatility / coefficient of variation
- `__yoy` — most-recent year-over-year change
- `n_years`, `frac_loss_years`, `frac_rev_decline` — survival & deterioration signals

The intuition — *a healthy company and a soon-to-fail one can look similar in their last snapshot, but their trajectories differ* — is borne out: **4 of the top-12 most important features are panel-derived** (years of history, profit-margin trend, revenue-decline frequency, retained-earnings trend).

### 2. Principled class-imbalance handling
- **BorderlineSMOTE** synthesizes minority examples near the decision boundary — empirically validated to beat naïve `scale_pos_weight` boosting on PR-AUC *and* precision.
- Applied **only to the scaled training fold**; the test set and the data used for counterfactual generation stay un-resampled to avoid leakage and unrealistic synthetic ranges.

### 3. Explainability, two ways
- **SHAP** (`TreeExplainer`) — per-prediction feature attributions, surfacing the top risk drivers for each company.
- **DiCE counterfactuals** — "what-if" analysis showing the *minimum changes* to the company's current-year financials that would flip the prediction to low-risk (panel-derived features like "years of history" are correctly held fixed, since they aren't actionable).

### 4. RAG-augmented, industry-aware advice
Selecting an industry retrieves relevant context from a **ChromaDB vector store** built over real Deloitte & KPMG sector outlook reports (`text-embedding-3-small` embeddings), which grounds **GPT-4o**-generated improvement recommendations in current industry trends rather than generic boilerplate.

### 5. Train/serve consistency by design
The Streamlit app **imports the exact feature-engineering functions from the training module** (single source of truth), eliminating the train/serve skew that silently degrades real ML systems. The inference path accepts multi-year input and reconstructs the identical 62-feature vector the model was trained on — verified end-to-end.

### 6. Production-app polish
Multilingual UI (**8 languages**), risk gauge, model-ensemble consensus view, and graceful degradation when no API key is configured.

---

## 🏗️ Architecture

```
                         ┌────────────────────────────────────────┐
   american_bankruptcy   │  train_model.py                        │
   (panel, 1999–2018) ──▶│  ratios → panel features (trend/vol)    │
                         │  → BorderlineSMOTE → XGBoost / RF / LGBM │
                         │  → F2-tuned threshold                    │──▶ models/*.pkl
                         └────────────────────────────────────────┘
   Deloitte / KPMG       ┌────────────────────────────────────────┐
   sector PDFs ─────────▶│  build_vectordb.py → ChromaDB embeddings │──▶ data/vectordb/
                         └────────────────────────────────────────┘
                                              │
                                              ▼
                         ┌────────────────────────────────────────┐
   user: N years of  ───▶│  app.py (Streamlit)                     │
   X1–X18 financials     │  reuse training feature fns → predict   │
                         │  → SHAP + DiCE counterfactuals          │
                         │  → RAG retrieval → GPT-4o advice         │──▶ risk score + explanation
                         └────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

**ML / Data:** Python, scikit-learn, XGBoost, LightGBM, imbalanced-learn (BorderlineSMOTE), pandas, NumPy
**Explainability:** SHAP, DiCE-ML
**LLM / RAG:** OpenAI (GPT-4o, text-embedding-3-small), ChromaDB, PyMuPDF
**App:** Streamlit (multilingual)

---

## 📂 Project Structure

```
.
├── src/
│   ├── train_model.py     # Feature engineering + training pipeline (panel features, SMOTE)
│   ├── build_vectordb.py  # PDF → chunks → embeddings → ChromaDB for RAG
│   └── app.py             # Streamlit app: predict, explain (SHAP), counterfactuals, RAG advice
├── data/
│   ├── american_bankruptcy.csv   # Panel dataset (8,971 companies, 1999–2018)
│   └── reports/                  # Deloitte/KPMG sector outlooks (RAG knowledge base)
├── docs/                  # Product report & demo slides
├── models/                # Trained artifacts (generated by train_model.py — git-ignored)
└── requirements.txt
```

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the models (generates models/*.pkl)
python src/train_model.py

# 3. (Optional) Build the RAG vector store — requires OPENAI_API_KEY in .env
python src/build_vectordb.py

# 4. Launch the app
streamlit run src/app.py
```

> An `OPENAI_API_KEY` in a `.env` file enables RAG-grounded GPT-4o advice. Without it, prediction, SHAP, counterfactuals, and the ensemble view still work — the app degrades gracefully.

---

## 🔬 Modeling Notes & Honest Limitations

- **Operating point is a business choice.** The model is tuned for recall (catch failures) at the cost of precision (~0.37). Raising the threshold trades recall for precision along the same PR curve — the right point depends on the cost asymmetry of the use case.
- **Label is company-level**, not event-level (a failed company is labeled "failed" in every year it appears), so the task is framed as *"will this company eventually fail?"* and features are aggregated per company accordingly.
- **Financial data only.** No macro, market, or textual signals — adding them is the most promising avenue to push ROC-AUC toward the 0.85–0.95 range seen in top bankruptcy-prediction literature.
- **Intended use:** a first-line risk *screening / early-warning* tool that flags companies for human review — not a standalone credit or investment decision.

---

## 📊 Dataset

[American Bankruptcy Dataset](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset) — 18 financial-statement features (X1–X18) for US public companies, 1999–2018, with eventual bankruptcy status.
