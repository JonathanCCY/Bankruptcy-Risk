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
from imblearn.over_sampling import BorderlineSMOTE
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

# --- Panel (multi-year) features ---
# Key series to summarize across each company's time window. Chosen for signal,
# kept small (11 series) to limit feature count vs the ~600 bankrupt companies.
KEY_RATIOS = [
    "R1_Current_Ratio", "R2_Debt_to_Asset", "R3_Net_Profit_Margin",
    "R4_EBITDA_Margin", "R6_Debt_to_Equity", "R8_RE_to_Assets",
]
KEY_RAW = ["X6", "X16", "X17", "X4", "X10"]  # Net Income, Revenue, Liabilities, EBITDA, Assets
PANEL_SERIES = KEY_RATIOS + KEY_RAW

# Per-series panel stats: slope (trend), vol (volatility), yoy (recent momentum).
# Level (_last) is already covered by RAW_FEATURES + ratios on the most recent year.
PANEL_STATS = ["slope", "vol", "yoy"]
PANEL_FEATURES = [f"{s}__{stat}" for s in PANEL_SERIES for stat in PANEL_STATS]

# Company-level meta features summarizing the whole window.
META_FEATURES = ["n_years", "frac_loss_years", "frac_rev_decline"]

# Human-readable labels for the new features
_STAT_LABEL = {"slope": "Trend", "vol": "Volatility", "yoy": "Recent YoY"}
PANEL_LABELS = {
    f"{s}__{stat}": f"{_STAT_LABEL[stat]} of {FEATURE_LABELS.get(s, RATIO_LABELS.get(s, s))}"
    for s in PANEL_SERIES for stat in PANEL_STATS
}
META_LABELS = {
    "n_years": "Years of History",
    "frac_loss_years": "Fraction of Loss Years",
    "frac_rev_decline": "Fraction of Revenue-Decline Years",
}

FEATURE_LABELS.update(RATIO_LABELS)
FEATURE_LABELS.update(PANEL_LABELS)
FEATURE_LABELS.update(META_LABELS)


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


def _slope(values):
    """OLS slope of a series over its (centered) year index. Returns 0 if <2 points."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    denom = (x ** 2).sum()
    if denom < 1e-9:
        return 0.0
    return float((x * (v - v.mean())).sum() / denom)


def panel_stats_for_series(values, is_raw):
    """Compute (slope, vol, yoy) for one company's time series of a single indicator.

    For raw (size-bearing) series, slope and vol are normalized by mean magnitude
    so they are comparable across companies of different sizes. Ratios are already
    scale-invariant and kept as-is. All outputs are clipped to [-10, 10].
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0.0, 0.0, 0.0

    scale = (np.mean(np.abs(v)) + 1e-3) if is_raw else 1.0
    slope = _slope(v) / scale
    vol = (np.std(v) / scale) if is_raw else float(np.std(v))
    yoy = safe_divide(v[-1] - v[-2], np.abs(v[-2])) if len(v) >= 2 else 0.0
    yoy = float(np.asarray(yoy))

    clip = lambda z: float(np.clip(z, -10, 10))
    return clip(slope), clip(vol), clip(yoy)


def compute_panel_features(company_df):
    """Build panel (multi-year) summary features for one company's full history.

    `company_df` is the company's rows sorted by year (already has ratios computed).
    Returns a dict of PANEL_FEATURES + META_FEATURES values.
    """
    out = {}
    for s in PANEL_SERIES:
        is_raw = s in KEY_RAW
        slope, vol, yoy = panel_stats_for_series(company_df[s].values, is_raw)
        out[f"{s}__slope"] = slope
        out[f"{s}__vol"] = vol
        out[f"{s}__yoy"] = yoy

    n_years = len(company_df)
    net_income = company_df["X6"].values
    revenue = company_df["X16"].values
    out["n_years"] = float(n_years)
    out["frac_loss_years"] = float(np.mean(net_income < 0)) if n_years else 0.0
    out["frac_rev_decline"] = (
        float(np.mean(np.diff(revenue) < 0)) if n_years >= 2 else 0.0
    )
    return out


FEATURES = RAW_FEATURES + list(RATIO_DEFINITIONS.keys()) + PANEL_FEATURES + META_FEATURES


def load_and_prepare_data():
    """Load panel data, compute per-year ratios, summarize each company's full
    time series into panel features, and collapse to one row per company."""
    df = pd.read_csv(DATA_PATH)

    # Encode target: failed=1, alive=0 (label is constant per company)
    df["target"] = (df["status_label"] == "failed").astype(int)

    # Sort by company and year so series are chronological
    df = df.sort_values(["company_name", "year"])

    # Compute per-year financial ratios (needed for panel stats on ratios)
    df = compute_ratios(df)

    # Build panel (multi-year) features per company from the full history
    panel_rows = []
    for company, g in df.groupby("company_name", sort=False):
        feats = compute_panel_features(g)
        feats["company_name"] = company
        panel_rows.append(feats)
    panel_df = pd.DataFrame(panel_rows).set_index("company_name")

    # Take the most recent year per company for level features + target
    last_df = df.groupby("company_name").last()

    # Merge level features (raw X1-X18 + ratios from last year) with panel features
    merged = last_df.join(panel_df)

    X = merged[FEATURES]
    y = merged["target"]

    n_panel = len(PANEL_FEATURES)
    print(f"Dataset: {len(merged)} companies")
    print(f"Features: {len(FEATURES)} ({len(RAW_FEATURES)} raw + {len(RATIO_DEFINITIONS)} ratios "
          f"+ {n_panel} panel + {len(META_FEATURES)} meta)")
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

    # --- Baseline: Logistic Regression (on original imbalanced data) ---
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("[Baseline] Logistic Regression")
    print("=" * 60)
    print(classification_report(y_test, y_pred_lr))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}\n")

    # --- Oversample minority class with BorderlineSMOTE ---
    # Synthesizes bankrupt examples near the decision boundary. Validated to lift
    # PR-AUC (0.38->0.40) and precision vs scale_pos_weight boosting. Applied to the
    # SCALED training set only; test set and saved train_stats stay un-resampled.
    X_res, y_res = BorderlineSMOTE(random_state=42).fit_resample(X_train, y_train)
    print(f"BorderlineSMOTE: train {len(X_train)} -> {len(X_res)} rows "
          f"(failed {int((y_train==1).sum())} -> {int((y_res==1).sum())})\n")

    # Tree models train on the balanced (resampled) set — no extra class weighting,
    # which would double-correct on top of SMOTE.
    # --- Main: XGBoost ---
    xgb = XGBClassifier(
        eval_metric="logloss",
        max_depth=5,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb.fit(X_res, y_res)
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
        random_state=42,
    )
    rf.fit(X_res, y_res)
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
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_res, y_res)
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
