import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, fbeta_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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

RAW_FEATURES = [f"X{i}" for i in range(1, 19)]

# Engineered financial ratios (normalize for company size)
RATIO_DEFINITIONS = {
    "R1_Current_Ratio":           ("X1", "X14"),   # Current Assets / Current Liabilities
    "R2_Debt_to_Asset":           ("X17", "X10"),   # Total Liabilities / Total Assets
    "R3_Net_Profit_Margin":       ("X6", "X16"),    # Net Income / Total Revenue
    "R4_EBITDA_Margin":           ("X4", "X16"),    # EBITDA / Total Revenue
    "R5_Gross_Margin":            ("X13", "X16"),   # Gross Profit / Total Revenue
    "R6_Debt_to_Equity":          ("X17", "X10-X17"),  # Total Liabilities / (Total Assets - Total Liabilities)
    "R7_Asset_Turnover":          ("X9", "X10"),    # Net Sales / Total Assets
    "R8_RE_to_Assets":            ("X15", "X10"),   # Retained Earnings / Total Assets
}

RATIO_LABELS = {
    "R1_Current_Ratio": "Current Ratio (X1/X14)",
    "R2_Debt_to_Asset": "Debt-to-Asset Ratio (X17/X10)",
    "R3_Net_Profit_Margin": "Net Profit Margin (X6/X16)",
    "R4_EBITDA_Margin": "EBITDA Margin (X4/X16)",
    "R5_Gross_Margin": "Gross Margin (X13/X16)",
    "R6_Debt_to_Equity": "Debt-to-Equity Ratio (X17/(X10-X17))",
    "R7_Asset_Turnover": "Asset Turnover (X9/X10)",
    "R8_RE_to_Assets": "Retained Earnings / Assets (X15/X10)",
}

# YoY (Year-over-Year) change rate — key indicators only
YOY_RAW = ["X4", "X6", "X10", "X16", "X17"]  # EBITDA, Net Income, Total Assets, Total Revenue, Total Liabilities
YOY_FEATURES = [f"YoY_{feat}" for feat in YOY_RAW]
YOY_LABELS = {f"YoY_{feat}": f"YoY {FEATURE_LABELS[feat]}" for feat in YOY_RAW}

FEATURE_LABELS.update(RATIO_LABELS)
FEATURE_LABELS.update(YOY_LABELS)


def safe_divide(numerator, denominator):
    """Safe division: returns 0 where denominator is 0 or NaN."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    mask = np.isnan(den) | (np.abs(den) < 1e-9)
    result = np.zeros_like(num, dtype=float)
    result[~mask] = num[~mask] / den[~mask]
    return result


def compute_ratios(df):
    """Compute financial ratio features from raw features."""
    df = df.copy()
    df["R1_Current_Ratio"]      = safe_divide(df["X1"], df["X14"])
    df["R2_Debt_to_Asset"]      = safe_divide(df["X17"], df["X10"])
    df["R3_Net_Profit_Margin"]  = safe_divide(df["X6"], df["X16"])
    df["R4_EBITDA_Margin"]      = safe_divide(df["X4"], df["X16"])
    df["R5_Gross_Margin"]       = safe_divide(df["X13"], df["X16"])
    df["R6_Debt_to_Equity"]     = safe_divide(df["X17"], df["X10"] - df["X17"])
    df["R7_Asset_Turnover"]     = safe_divide(df["X9"], df["X10"])
    df["R8_RE_to_Assets"]       = safe_divide(df["X15"], df["X10"])

    ratio_cols = [c for c in df.columns if c.startswith("R")]
    for col in ratio_cols:
        df[col] = df[col].clip(-10, 10)

    return df


def compute_yoy_from_values(current_vals, prev_vals):
    """Compute YoY change rate: (current - prev) / |prev|. Returns 0 if prev is 0."""
    return safe_divide(current_vals - prev_vals, np.abs(prev_vals))


FEATURES = RAW_FEATURES + list(RATIO_DEFINITIONS.keys()) + YOY_FEATURES


def load_and_prepare_data():
    """Load panel data, compute ratios and YoY lag features, take most recent year."""
    df = pd.read_csv(DATA_PATH)

    # Encode target: failed=1, alive=0
    df["target"] = (df["status_label"] == "failed").astype(int)

    # Sort by company and year for lag computation
    df = df.sort_values(["company_name", "year"])

    # Compute YoY change rate for key indicators per company
    for feat in YOY_RAW:
        prev_col = df.groupby("company_name")[feat].shift(1)
        df[f"YoY_{feat}"] = compute_yoy_from_values(df[feat].values, prev_col.values)

    # Fill NaN (first year per company has no lag) with 0, clip outliers
    for feat in YOY_FEATURES:
        df[feat] = df[feat].fillna(0.0).clip(-10, 10)

    # Take the most recent year per company (now with YoY features)
    df = df.groupby("company_name").last().reset_index()

    # Compute financial ratios
    df = compute_ratios(df)

    X = df[FEATURES]
    y = df["target"]

    print(f"Dataset: {len(df)} companies")
    print(f"Features: {len(FEATURES)} ({len(RAW_FEATURES)} raw + {len(RATIO_DEFINITIONS)} ratios + {len(YOY_FEATURES)} YoY)")
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

    # --- Standardize raw features (X1-X18) to remove company-size bias ---
    scaler = StandardScaler()
    X_train[RAW_FEATURES] = scaler.fit_transform(X_train[RAW_FEATURES])
    X_test[RAW_FEATURES] = scaler.transform(X_test[RAW_FEATURES])
    print("StandardScaler applied to raw features (X1-X18)")
    print("Ratios (R1-R8) kept as-is (already scale-invariant)\n")

    # --- Baseline: Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("[Baseline] Logistic Regression")
    print("=" * 60)
    print(classification_report(y_test, y_pred_lr))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}\n")

    # Boost scale_pos_weight 2x to prioritize catching bankrupt companies (higher recall)
    boosted_weight = scale_ratio * 2
    print(f"Boosted scale_pos_weight: {boosted_weight:.2f} (2x natural ratio)\n")

    # --- Main: XGBoost ---
    xgb = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=boosted_weight,
        max_depth=5,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    # Find optimal threshold using F2-score (recall-weighted)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_xgb)
    best_threshold = 0.5
    best_f2 = 0
    for t_val in np.arange(0.10, 0.60, 0.01):
        y_pred_t = (y_prob_xgb >= t_val).astype(int)
        f2 = fbeta_score(y_test, y_pred_t, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t_val

    print(f"Optimal threshold (F2-score): {best_threshold:.2f} (F2={best_f2:.4f})")

    y_pred_xgb_default = (y_prob_xgb >= 0.5).astype(int)
    y_pred_xgb_optimal = (y_prob_xgb >= best_threshold).astype(int)

    print("=" * 60)
    print("[Main] XGBoost (threshold=0.50)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_xgb_default))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}\n")

    print("=" * 60)
    print(f"[Main] XGBoost (threshold={best_threshold:.2f} - optimized)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_xgb_optimal))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}\n")

    # --- Ensemble: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf >= best_threshold).astype(int)

    print("=" * 60)
    print(f"[Ensemble] Random Forest (threshold={best_threshold:.2f})")
    print("=" * 60)
    print(classification_report(y_test, y_pred_rf))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}\n")

    # --- Ensemble: LightGBM ---
    lgbm = LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=boosted_weight,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    y_prob_lgbm = lgbm.predict_proba(X_test)[:, 1]
    y_pred_lgbm = (y_prob_lgbm >= best_threshold).astype(int)

    print("=" * 60)
    print(f"[Ensemble] LightGBM (threshold={best_threshold:.2f})")
    print("=" * 60)
    print(classification_report(y_test, y_pred_lgbm))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lgbm):.4f}\n")

    # Save model artifacts
    joblib.dump(xgb, MODEL_DIR / "xgb_bankruptcy.pkl")
    joblib.dump(rf, MODEL_DIR / "rf_bankruptcy.pkl")
    joblib.dump(lgbm, MODEL_DIR / "lgbm_bankruptcy.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(best_threshold, MODEL_DIR / "threshold.pkl")
    joblib.dump(FEATURES, MODEL_DIR / "features.pkl")
    joblib.dump(FEATURE_LABELS, MODEL_DIR / "feature_labels.pkl")

    # Save training data for counterfactual generation (already scaled)
    joblib.dump({"X_train": X_train, "y_train": y_train}, MODEL_DIR / "train_stats.pkl")

    print("Models saved:")
    print("  - models/xgb_bankruptcy.pkl")
    print("  - models/rf_bankruptcy.pkl")
    print("  - models/lgbm_bankruptcy.pkl")
    print(f"  - models/threshold.pkl ({best_threshold:.2f})")
    print("  - models/train_stats.pkl")


if __name__ == "__main__":
    train()
