import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "data" / "american_bankruptcy.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Feature name mapping (X1-X18 from the American Bankruptcy dataset)
FEATURE_LABELS = {
    "X1": "Current Assets",
    "X2": "Cost of Goods Sold",
    "X3": "Depreciation and Amortization",
    "X4": "EBITDA",
    "X5": "Inventory",
    "X6": "Net Income",
    "X7": "Total Receivables",
    "X8": "Market Value",
    "X9": "Net Sales",
    "X10": "Total Assets",
    "X11": "Total Long-term Debt",
    "X12": "EBIT",
    "X13": "Gross Profit",
    "X14": "Total Current Liabilities",
    "X15": "Retained Earnings",
    "X16": "Total Revenue",
    "X17": "Total Liabilities",
    "X18": "Total Operating Expenses",
}

FEATURES = [f"X{i}" for i in range(1, 19)]


def load_and_prepare_data():
    """Load CSV and prepare features/target."""
    df = pd.read_csv(DATA_PATH)

    # Encode target: failed=1, alive=0
    df["target"] = (df["status_label"] == "failed").astype(int)

    # Use the most recent year per company to avoid data leakage
    df = df.sort_values("year").groupby("company_name").last().reset_index()

    X = df[FEATURES]
    y = df["target"]

    print(f"Dataset: {len(df)} companies")
    print(f"Class distribution:\n{y.value_counts().to_string()}")
    print(f"Bankruptcy rate: {y.mean():.2%}\n")

    return X, y


def train():
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_ratio = n_neg / n_pos if n_pos > 0 else 1
    print(f"Class imbalance ratio (alive/failed): {scale_ratio:.2f}")

    # --- Baseline: Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    print("=" * 60)
    print("[Baseline] Logistic Regression")
    print("=" * 60)
    print(classification_report(y_test, y_pred_lr))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}\n")

    # --- Main: XGBoost ---
    xgb = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_ratio,
        max_depth=5,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("[Main] XGBoost")
    print("=" * 60)
    print(classification_report(y_test, y_pred_xgb))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}\n")

    # Save model artifacts
    joblib.dump(xgb, MODEL_DIR / "xgb_bankruptcy.pkl")
    joblib.dump(FEATURES, MODEL_DIR / "features.pkl")
    joblib.dump(FEATURE_LABELS, MODEL_DIR / "feature_labels.pkl")
    print("Model saved to models/xgb_bankruptcy.pkl")


if __name__ == "__main__":
    train()
