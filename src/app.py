import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from openai import OpenAI
import chromadb
import dice_ml

# Reuse the exact panel feature logic from training (single source of truth)
import train_model as tm


def safe_divide(numerator, denominator):
    """Safe division: returns 0 where denominator is 0 or NaN."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    mask = np.isnan(den) | (np.abs(den) < 1e-9)
    result = np.zeros_like(num, dtype=float)
    result[~mask] = num[~mask] / den[~mask]
    return result


def compute_ratios(df):
    """Compute financial ratio features from raw features (must match train_model.py)."""
    df = df.copy()
    df["R1_Current_Ratio"]      = safe_divide(df["X1"].values, df["X14"].values)
    df["R2_Debt_to_Asset"]      = safe_divide(df["X17"].values, df["X10"].values)
    df["R3_Net_Profit_Margin"]  = safe_divide(df["X6"].values, df["X16"].values)
    df["R4_EBITDA_Margin"]      = safe_divide(df["X4"].values, df["X16"].values)
    df["R5_Gross_Margin"]       = safe_divide(df["X13"].values, df["X16"].values)
    df["R6_Debt_to_Equity"]     = safe_divide(df["X17"].values, df["X10"].values - df["X17"].values)
    df["R7_Asset_Turnover"]     = safe_divide(df["X9"].values, df["X10"].values)
    df["R8_RE_to_Assets"]       = safe_divide(df["X15"].values, df["X10"].values)

    ratio_cols = [c for c in df.columns if c.startswith("R")]
    for col in ratio_cols:
        df[col] = df[col].clip(-10, 10)

    return df


def build_inference_features(years_df):
    """Build the full FEATURES vector from a company's multi-year history.

    `years_df`: one row per fiscal year (oldest first, newest last), columns X1-X18.
    Mirrors train_model.load_and_prepare_data: per-year ratios -> panel summary of
    the full window -> level features from the most recent year. Returns a single-row
    DataFrame aligned to tm.FEATURES (unscaled; caller applies the scaler).
    """
    df = years_df.copy().reset_index(drop=True)

    # Per-year financial ratios (needed for panel stats on ratios)
    df = compute_ratios(df)

    # Panel (multi-year) summary features from the full window
    panel = tm.compute_panel_features(df)

    # Level features = most recent year's raw + ratio values
    last = df.iloc[[-1]].copy().reset_index(drop=True)
    for k, v in panel.items():
        last[k] = v

    return last.reindex(columns=tm.FEATURES, fill_value=0.0)


# --- Paths ---
ROOT_DIR = Path(__file__).parent.parent
MODEL_DIR = ROOT_DIR / "models"
CHROMA_DIR = ROOT_DIR / "data" / "vectordb"

# --- Industry options ---
INDUSTRIES = [
    "Manufacturing",
    "Technology",
    "Chemical",
    "Engineering & Construction",
    "Oil & Gas",
    "Power & Utilities",
    "Renewable Energy",
    "Other",
]

# --- i18n ---
LANGUAGES = {
    "English": "en",
    "繁體中文": "zh-TW",
    "简体中文": "zh-CN",
    "日本語": "ja",
    "한국어": "ko",
    "Español": "es",
    "Français": "fr",
    "Deutsch": "de",
}

UI_TEXT = {
    "en": {
        "page_title": "Bankruptcy Risk Predictor",
        "title": "📉 Corporate Bankruptcy Risk Predictor",
        "caption": "Enter financial indicators to predict bankruptcy risk, with AI-powered improvement suggestions",
        "sidebar_header": "📥 Financial Indicators",
        "sidebar_desc": "Enter the company's financial data below:",
        "predict_btn": "🚀 Predict Bankruptcy Risk",
        "risk_score_title": "### 🎯 Bankruptcy Risk Score",
        "high_risk": "High Risk",
        "medium_risk": "Medium Risk",
        "low_risk": "Low Risk",
        "shap_title": "### 📊 Risk Factor Analysis (SHAP)",
        "top_factors": "#### 🔍 Top Risk Factors",
        "all_impacts": "#### 📋 All Feature Impacts",
        "increases_risk": "🔴 Increases Risk",
        "decreases_risk": "🟢 Decreases Risk",
        "shap_info": (
            "💡 **How to read SHAP values:**\n"
            "- **Positive (+)**: This feature increases bankruptcy risk\n"
            "- **Negative (-)**: This feature decreases bankruptcy risk\n"
            "- **Larger absolute value**: Stronger influence on prediction"
        ),
        "ai_title": "### 💡 AI Improvement Recommendations",
        "ai_spinner": "Generating personalized recommendations...",
        "ai_success": "Recommendations generated successfully!",
        "ai_fail": "Failed to generate AI recommendations",
        "ai_no_key": (
            "💡 AI recommendations require an OpenAI API key. "
            "Set `OPENAI_API_KEY` in the `.env` file to enable this feature."
        ),
        "footer": "📁 Corporate Bankruptcy Risk Predictor | Model: XGBoost + SHAP",
        "model_not_found": "Model files not found. Please run `python src/train_model.py` first.",
        "col_feature": "Feature",
        "col_input": "Input Value",
        "col_shap": "SHAP Value",
        "col_impact": "Impact",
        "language": "🌐 Language",
        "industry_label": "🏭 Industry Sector",
        "industry_desc": "Select the company's industry for tailored recommendations:",
        "rag_context_title": "#### 📄 Industry Insights Referenced",
    },
    "zh-TW": {
        "page_title": "企業破產風險預測器",
        "title": "📉 企業破產風險預測器",
        "caption": "輸入財務指標以預測破產風險，並獲得 AI 改善建議",
        "sidebar_header": "📥 財務指標",
        "sidebar_desc": "請輸入公司的財務數據：",
        "predict_btn": "🚀 預測破產風險",
        "risk_score_title": "### 🎯 破產風險評分",
        "high_risk": "高風險",
        "medium_risk": "中風險",
        "low_risk": "低風險",
        "shap_title": "### 📊 風險因素分析 (SHAP)",
        "top_factors": "#### 🔍 主要風險因素",
        "all_impacts": "#### 📋 所有特徵影響",
        "increases_risk": "🔴 增加風險",
        "decreases_risk": "🟢 降低風險",
        "shap_info": (
            "💡 **如何解讀 SHAP 值：**\n"
            "- **正值 (+)**：此特徵增加破產風險\n"
            "- **負值 (-)**：此特徵降低破產風險\n"
            "- **絕對值越大**：對預測的影響越強"
        ),
        "ai_title": "### 💡 AI 改善建議",
        "ai_spinner": "正在生成個性化建議...",
        "ai_success": "建議生成成功！",
        "ai_fail": "生成 AI 建議失敗",
        "ai_no_key": (
            "💡 AI 建議需要 OpenAI API 金鑰。"
            "請在 `.env` 檔案中設定 `OPENAI_API_KEY` 以啟用此功能。"
        ),
        "footer": "📁 企業破產風險預測器 | 模型：XGBoost + SHAP",
        "model_not_found": "找不到模型檔案。請先執行 `python src/train_model.py`。",
        "col_feature": "特徵",
        "col_input": "輸入值",
        "col_shap": "SHAP 值",
        "col_impact": "影響",
        "language": "🌐 語言",
        "industry_label": "🏭 產業類別",
        "industry_desc": "選擇公司所屬產業以獲取客製化建議：",
        "rag_context_title": "#### 📄 參考的產業分析",
    },
    "zh-CN": {
        "page_title": "企业破产风险预测器",
        "title": "📉 企业破产风险预测器",
        "caption": "输入财务指标以预测破产风险，并获得 AI 改善建议",
        "sidebar_header": "📥 财务指标",
        "sidebar_desc": "请输入公司的财务数据：",
        "predict_btn": "🚀 预测破产风险",
        "risk_score_title": "### 🎯 破产风险评分",
        "high_risk": "高风险",
        "medium_risk": "中风险",
        "low_risk": "低风险",
        "shap_title": "### 📊 风险因素分析 (SHAP)",
        "top_factors": "#### 🔍 主要风险因素",
        "all_impacts": "#### 📋 所有特征影响",
        "increases_risk": "🔴 增加风险",
        "decreases_risk": "🟢 降低风险",
        "shap_info": (
            "💡 **如何解读 SHAP 值：**\n"
            "- **正值 (+)**：此特征增加破产风险\n"
            "- **负值 (-)**：此特征降低破产风险\n"
            "- **绝对值越大**：对预测的影响越强"
        ),
        "ai_title": "### 💡 AI 改善建议",
        "ai_spinner": "正在生成个性化建议...",
        "ai_success": "建议生成成功！",
        "ai_fail": "生成 AI 建议失败",
        "ai_no_key": (
            "💡 AI 建议需要 OpenAI API 密钥。"
            "请在 `.env` 文件中设置 `OPENAI_API_KEY` 以启用此功能。"
        ),
        "footer": "📁 企业破产风险预测器 | 模型：XGBoost + SHAP",
        "model_not_found": "找不到模型文件。请先运行 `python src/train_model.py`。",
        "col_feature": "特征",
        "col_input": "输入值",
        "col_shap": "SHAP 值",
        "col_impact": "影响",
        "language": "🌐 语言",
        "industry_label": "🏭 行业类别",
        "industry_desc": "选择公司所属行业以获取定制化建议：",
        "rag_context_title": "#### 📄 参考的行业分析",
    },
    "ja": {
        "page_title": "企業倒産リスク予測",
        "title": "📉 企業倒産リスク予測",
        "caption": "財務指標を入力して倒産リスクを予測し、AIによる改善提案を受け取れます",
        "sidebar_header": "📥 財務指標",
        "sidebar_desc": "企業の財務データを入力してください：",
        "predict_btn": "🚀 倒産リスクを予測",
        "risk_score_title": "### 🎯 倒産リスクスコア",
        "high_risk": "高リスク",
        "medium_risk": "中リスク",
        "low_risk": "低リスク",
        "shap_title": "### 📊 リスク要因分析 (SHAP)",
        "top_factors": "#### 🔍 主要リスク要因",
        "all_impacts": "#### 📋 全特徴の影響",
        "increases_risk": "🔴 リスク増加",
        "decreases_risk": "🟢 リスク低減",
        "shap_info": (
            "💡 **SHAP値の読み方：**\n"
            "- **正の値 (+)**：この特徴は倒産リスクを増加させます\n"
            "- **負の値 (-)**：この特徴は倒産リスクを低減させます\n"
            "- **絶対値が大きいほど**：予測への影響が強い"
        ),
        "ai_title": "### 💡 AI改善提案",
        "ai_spinner": "パーソナライズされた提案を生成中...",
        "ai_success": "提案が正常に生成されました！",
        "ai_fail": "AI提案の生成に失敗しました",
        "ai_no_key": (
            "💡 AI提案にはOpenAI APIキーが必要です。"
            "`.env`ファイルに`OPENAI_API_KEY`を設定してください。"
        ),
        "footer": "📁 企業倒産リスク予測 | モデル：XGBoost + SHAP",
        "model_not_found": "モデルファイルが見つかりません。先に `python src/train_model.py` を実行してください。",
        "col_feature": "特徴",
        "col_input": "入力値",
        "col_shap": "SHAP値",
        "col_impact": "影響",
        "language": "🌐 言語",
        "industry_label": "🏭 業種",
        "industry_desc": "企業の業種を選択してカスタマイズされた提案を取得：",
        "rag_context_title": "#### 📄 参照した業界分析",
    },
    "ko": {
        "page_title": "기업 파산 위험 예측기",
        "title": "📉 기업 파산 위험 예측기",
        "caption": "재무 지표를 입력하여 파산 위험을 예측하고 AI 기반 개선 제안을 받으세요",
        "sidebar_header": "📥 재무 지표",
        "sidebar_desc": "회사의 재무 데이터를 입력하세요:",
        "predict_btn": "🚀 파산 위험 예측",
        "risk_score_title": "### 🎯 파산 위험 점수",
        "high_risk": "고위험",
        "medium_risk": "중위험",
        "low_risk": "저위험",
        "shap_title": "### 📊 위험 요인 분석 (SHAP)",
        "top_factors": "#### 🔍 주요 위험 요인",
        "all_impacts": "#### 📋 모든 특성 영향",
        "increases_risk": "🔴 위험 증가",
        "decreases_risk": "🟢 위험 감소",
        "shap_info": (
            "💡 **SHAP 값 읽는 방법:**\n"
            "- **양수 (+)**: 이 특성은 파산 위험을 증가시킵니다\n"
            "- **음수 (-)**: 이 특성은 파산 위험을 감소시킵니다\n"
            "- **절대값이 클수록**: 예측에 대한 영향이 강합니다"
        ),
        "ai_title": "### 💡 AI 개선 제안",
        "ai_spinner": "맞춤형 제안을 생성하는 중...",
        "ai_success": "제안이 성공적으로 생성되었습니다!",
        "ai_fail": "AI 제안 생성에 실패했습니다",
        "ai_no_key": (
            "💡 AI 제안에는 OpenAI API 키가 필요합니다. "
            "`.env` 파일에 `OPENAI_API_KEY`를 설정하세요."
        ),
        "footer": "📁 기업 파산 위험 예측기 | 모델: XGBoost + SHAP",
        "model_not_found": "모델 파일을 찾을 수 없습니다. 먼저 `python src/train_model.py`를 실행하세요.",
        "col_feature": "특성",
        "col_input": "입력값",
        "col_shap": "SHAP 값",
        "col_impact": "영향",
        "language": "🌐 언어",
        "industry_label": "🏭 산업 분야",
        "industry_desc": "맞춤형 제안을 위해 회사의 산업을 선택하세요:",
        "rag_context_title": "#### 📄 참고한 산업 분석",
    },
    "es": {
        "page_title": "Predictor de Riesgo de Quiebra",
        "title": "📉 Predictor de Riesgo de Quiebra Corporativa",
        "caption": "Ingrese indicadores financieros para predecir el riesgo de quiebra, con sugerencias de mejora impulsadas por IA",
        "sidebar_header": "📥 Indicadores Financieros",
        "sidebar_desc": "Ingrese los datos financieros de la empresa:",
        "predict_btn": "🚀 Predecir Riesgo de Quiebra",
        "risk_score_title": "### 🎯 Puntuación de Riesgo de Quiebra",
        "high_risk": "Alto Riesgo",
        "medium_risk": "Riesgo Medio",
        "low_risk": "Bajo Riesgo",
        "shap_title": "### 📊 Análisis de Factores de Riesgo (SHAP)",
        "top_factors": "#### 🔍 Principales Factores de Riesgo",
        "all_impacts": "#### 📋 Impacto de Todas las Características",
        "increases_risk": "🔴 Aumenta el Riesgo",
        "decreases_risk": "🟢 Disminuye el Riesgo",
        "shap_info": (
            "💡 **Cómo leer los valores SHAP:**\n"
            "- **Positivo (+)**: Esta característica aumenta el riesgo de quiebra\n"
            "- **Negativo (-)**: Esta característica disminuye el riesgo de quiebra\n"
            "- **Mayor valor absoluto**: Mayor influencia en la predicción"
        ),
        "ai_title": "### 💡 Recomendaciones de Mejora con IA",
        "ai_spinner": "Generando recomendaciones personalizadas...",
        "ai_success": "¡Recomendaciones generadas con éxito!",
        "ai_fail": "Error al generar recomendaciones de IA",
        "ai_no_key": (
            "💡 Las recomendaciones de IA requieren una clave API de OpenAI. "
            "Configure `OPENAI_API_KEY` en el archivo `.env` para habilitar esta función."
        ),
        "footer": "📁 Predictor de Riesgo de Quiebra | Modelo: XGBoost + SHAP",
        "model_not_found": "Archivos de modelo no encontrados. Ejecute `python src/train_model.py` primero.",
        "col_feature": "Característica",
        "col_input": "Valor de Entrada",
        "col_shap": "Valor SHAP",
        "col_impact": "Impacto",
        "language": "🌐 Idioma",
        "industry_label": "🏭 Sector Industrial",
        "industry_desc": "Seleccione la industria de la empresa para recomendaciones personalizadas:",
        "rag_context_title": "#### 📄 Análisis Industrial Referenciado",
    },
    "fr": {
        "page_title": "Prédicteur de Risque de Faillite",
        "title": "📉 Prédicteur de Risque de Faillite d'Entreprise",
        "caption": "Saisissez les indicateurs financiers pour prédire le risque de faillite, avec des suggestions d'amélioration par IA",
        "sidebar_header": "📥 Indicateurs Financiers",
        "sidebar_desc": "Saisissez les données financières de l'entreprise :",
        "predict_btn": "🚀 Prédire le Risque de Faillite",
        "risk_score_title": "### 🎯 Score de Risque de Faillite",
        "high_risk": "Risque Élevé",
        "medium_risk": "Risque Moyen",
        "low_risk": "Risque Faible",
        "shap_title": "### 📊 Analyse des Facteurs de Risque (SHAP)",
        "top_factors": "#### 🔍 Principaux Facteurs de Risque",
        "all_impacts": "#### 📋 Impact de Toutes les Caractéristiques",
        "increases_risk": "🔴 Augmente le Risque",
        "decreases_risk": "🟢 Diminue le Risque",
        "shap_info": (
            "💡 **Comment lire les valeurs SHAP :**\n"
            "- **Positif (+)** : Cette caractéristique augmente le risque de faillite\n"
            "- **Négatif (-)** : Cette caractéristique diminue le risque de faillite\n"
            "- **Plus la valeur absolue est grande** : Plus l'influence sur la prédiction est forte"
        ),
        "ai_title": "### 💡 Recommandations d'Amélioration par IA",
        "ai_spinner": "Génération de recommandations personnalisées...",
        "ai_success": "Recommandations générées avec succès !",
        "ai_fail": "Échec de la génération des recommandations IA",
        "ai_no_key": (
            "💡 Les recommandations IA nécessitent une clé API OpenAI. "
            "Configurez `OPENAI_API_KEY` dans le fichier `.env` pour activer cette fonctionnalité."
        ),
        "footer": "📁 Prédicteur de Risque de Faillite | Modèle : XGBoost + SHAP",
        "model_not_found": "Fichiers de modèle introuvables. Veuillez d'abord exécuter `python src/train_model.py`.",
        "col_feature": "Caractéristique",
        "col_input": "Valeur d'Entrée",
        "col_shap": "Valeur SHAP",
        "col_impact": "Impact",
        "language": "🌐 Langue",
        "industry_label": "🏭 Secteur Industriel",
        "industry_desc": "Sélectionnez le secteur de l'entreprise pour des recommandations personnalisées :",
        "rag_context_title": "#### 📄 Analyses Industrielles Référencées",
    },
    "de": {
        "page_title": "Insolvenzrisiko-Vorhersage",
        "title": "📉 Unternehmens-Insolvenzrisiko-Vorhersage",
        "caption": "Geben Sie Finanzkennzahlen ein, um das Insolvenzrisiko vorherzusagen, mit KI-gestützten Verbesserungsvorschlägen",
        "sidebar_header": "📥 Finanzkennzahlen",
        "sidebar_desc": "Geben Sie die Finanzdaten des Unternehmens ein:",
        "predict_btn": "🚀 Insolvenzrisiko vorhersagen",
        "risk_score_title": "### 🎯 Insolvenzrisiko-Score",
        "high_risk": "Hohes Risiko",
        "medium_risk": "Mittleres Risiko",
        "low_risk": "Niedriges Risiko",
        "shap_title": "### 📊 Risikofaktor-Analyse (SHAP)",
        "top_factors": "#### 🔍 Wichtigste Risikofaktoren",
        "all_impacts": "#### 📋 Alle Merkmal-Auswirkungen",
        "increases_risk": "🔴 Erhöht das Risiko",
        "decreases_risk": "🟢 Verringert das Risiko",
        "shap_info": (
            "💡 **So lesen Sie SHAP-Werte:**\n"
            "- **Positiv (+)**: Dieses Merkmal erhöht das Insolvenzrisiko\n"
            "- **Negativ (-)**: Dieses Merkmal verringert das Insolvenzrisiko\n"
            "- **Größerer Absolutwert**: Stärkerer Einfluss auf die Vorhersage"
        ),
        "ai_title": "### 💡 KI-Verbesserungsempfehlungen",
        "ai_spinner": "Personalisierte Empfehlungen werden generiert...",
        "ai_success": "Empfehlungen erfolgreich generiert!",
        "ai_fail": "KI-Empfehlungen konnten nicht generiert werden",
        "ai_no_key": (
            "💡 KI-Empfehlungen erfordern einen OpenAI-API-Schlüssel. "
            "Setzen Sie `OPENAI_API_KEY` in der `.env`-Datei, um diese Funktion zu aktivieren."
        ),
        "footer": "📁 Insolvenzrisiko-Vorhersage | Modell: XGBoost + SHAP",
        "model_not_found": "Modelldateien nicht gefunden. Bitte führen Sie zuerst `python src/train_model.py` aus.",
        "col_feature": "Merkmal",
        "col_input": "Eingabewert",
        "col_shap": "SHAP-Wert",
        "col_impact": "Auswirkung",
        "language": "🌐 Sprache",
        "industry_label": "🏭 Branche",
        "industry_desc": "Wählen Sie die Branche des Unternehmens für maßgeschneiderte Empfehlungen:",
        "rag_context_title": "#### 📄 Referenzierte Branchenanalysen",
    },
}

# Language name for GPT prompt
LANG_NAMES = {
    "en": "English", "zh-TW": "Traditional Chinese (繁體中文)",
    "zh-CN": "Simplified Chinese (简体中文)", "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)", "es": "Spanish (Español)",
    "fr": "French (Français)", "de": "German (Deutsch)",
}


# --- Load vector store ---
@st.cache_resource
def load_vectordb():
    """Load ChromaDB vector store for RAG."""
    if not CHROMA_DIR.exists():
        return None
    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = chroma_client.get_collection("industry_reports")
        return collection
    except Exception:
        return None


def retrieve_industry_context(collection, openai_client, industry: str, query: str, n_results: int = 5) -> list[str]:
    """Retrieve relevant chunks from vector store based on industry and query."""
    if collection is None or openai_client is None:
        return []

    # Generate query embedding
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_embedding = response.data[0].embedding

    # Query with industry filter
    where_filter = {"industry": industry} if industry != "Other" else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
    )

    return results["documents"][0] if results["documents"] else []


# Raw feature names (X1-X18) for scaler
RAW_FEATURES = [f"X{i}" for i in range(1, 19)]


# --- Load model artifacts ---
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "xgb_bankruptcy.pkl")
    features = joblib.load(MODEL_DIR / "features.pkl")
    feature_labels = joblib.load(MODEL_DIR / "feature_labels.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    threshold = joblib.load(MODEL_DIR / "threshold.pkl")
    explainer = shap.TreeExplainer(model)
    return model, features, feature_labels, scaler, threshold, explainer


@st.cache_resource
def load_ensemble_models():
    """Load RF and LightGBM models for ensemble comparison."""
    models = {}
    for name, fname in [("Random Forest", "rf_bankruptcy.pkl"), ("LightGBM", "lgbm_bankruptcy.pkl")]:
        path = MODEL_DIR / fname
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_train_data():
    """Load training data for counterfactual generation."""
    path = MODEL_DIR / "train_stats.pkl"
    if path.exists():
        return joblib.load(path)
    return None


# --- OpenAI client ---
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def generate_advice(client, shap_array, feature_names, feature_labels, input_values, risk_score, lang_code, industry="Other", industry_context=None):
    """Generate AI improvement suggestions based on SHAP analysis and industry context."""
    top_indices = sorted(
        range(len(shap_array)), key=lambda i: abs(shap_array[i]), reverse=True
    )[:5]

    details = []
    for i in top_indices:
        fname = feature_names[i]
        label = feature_labels.get(fname, fname)
        val = input_values[i]
        sv = shap_array[i]
        impact = "increases bankruptcy risk" if sv > 0 else "decreases bankruptcy risk"
        details.append(f"- {label} ({fname}): value={val:.2f}, SHAP={sv:.4f} ({impact})")

    lang_name = LANG_NAMES.get(lang_code, "English")

    # Build industry context section
    context_section = ""
    if industry_context and industry != "Other":
        context_text = "\n\n---\n\n".join(industry_context)
        context_section = f"""

--- SUPPLEMENTARY INDUSTRY CONTEXT (for reference only) ---
The company operates in the **{industry}** industry. Below are excerpts from Deloitte's industry outlook reports.
These are SUPPLEMENTARY context only — do NOT let them override the company's own financial data above.

{context_text}
--- END OF SUPPLEMENTARY CONTEXT ---
"""

    prompt = f"""You are a professional financial advisor specializing in corporate bankruptcy risk analysis.

A company has been assessed with a Bankruptcy Risk Score of {risk_score:.0f}/100.

The top factors influencing this prediction (from a SHAP-based ML model) are:

{chr(10).join(details)}
{context_section}
**IMPORTANT — RAG Usage Rules (you must follow these strictly):**
1. Your analysis and recommendations must be PRIMARILY driven by the company's bankruptcy risk score, financial indicators, and SHAP feature contributions above. These are the ground truth for this company.
2. The retrieved industry context (if provided) may ONLY be used to:
   a. Explain why a SHAP-identified risk driver is particularly relevant in the {industry} industry.
   b. Suggest realistic mitigation practices commonly used in this sector — but only for risks already identified by the company's own financial data.
   c. Strengthen or refine a recommendation that is already supported by company-specific financial evidence.
3. Do NOT introduce a recommendation solely because it appears in the industry report excerpts.
4. If a recommendation is supported mainly by industry context but NOT by the company's financial indicators or SHAP drivers, you may mention it separately as a "Sector-Context Suggestion" clearly labeled as secondary.
5. If there is any conflict between the industry context and the company's financial evidence, ALWAYS prioritize the company's financial evidence.

**Your response structure:**
1. Briefly explain what the company's financial indicators and SHAP values suggest about its financial health.
2. Provide 3-5 specific, actionable core recommendations to reduce bankruptcy risk, grounded in the company's own data.{f" Where a recommendation aligns with {industry} industry trends, note the connection briefly." if industry != "Other" else ""}
3. Prioritize the core recommendations by impact.
{f"4. (Optional) If relevant, add 1-2 secondary sector-context suggestions clearly labeled as supplementary." if industry != "Other" else ""}

Use a professional but accessible tone. Write in {lang_name}."""

    system_msg = (
        "You are a professional financial advisor specializing in corporate distress and turnaround strategies. "
        "You always ground your analysis in company-specific financial data first. "
        "Industry report data is secondary supporting context — never the primary basis for recommendations."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        timeout=60.0,
    )
    return response.choices[0].message.content


# --- Page config ---
st.set_page_config(page_title="Bankruptcy Risk Predictor", page_icon="📉", layout="wide")

# --- Language selector (top of sidebar) ---
lang_display = st.sidebar.selectbox("🌐 Language", list(LANGUAGES.keys()), index=0)
lang_code = LANGUAGES[lang_display]
t = UI_TEXT[lang_code]

# --- Load resources ---
try:
    model, features, feature_labels, scaler, threshold, explainer = load_model()
except FileNotFoundError:
    st.error(t["model_not_found"])
    st.stop()

client = get_openai_client()
vectordb = load_vectordb()
ensemble_models = load_ensemble_models()
train_data = load_train_data()

# --- Header ---
st.title(t["title"])
st.caption(t["caption"])

# --- Sidebar: Input ---
st.sidebar.header(t["sidebar_header"])
st.sidebar.markdown(t["industry_desc"])
selected_industry = st.sidebar.selectbox(t["industry_label"], INDUSTRIES, index=0)
st.sidebar.markdown("---")
st.sidebar.markdown(t["sidebar_desc"])

# Multi-year financials input — one row per fiscal year (oldest first, newest last).
# Panel features (trend / volatility / momentum) are computed from the full window.
n_years = st.sidebar.number_input(
    "Years of history", min_value=1, max_value=20, value=5, step=1,
    help="Enter X1-X18 for each year. More years = stronger trend/volatility signals.",
)
n_years = int(n_years)

year_labels = [f"Year -{n}" if n > 0 else "Year 0 (latest)" for n in range(n_years - 1, -1, -1)]
default_years_df = pd.DataFrame(0.0, index=year_labels, columns=RAW_FEATURES)

st.sidebar.caption("Rows = fiscal years (top = oldest, bottom = latest). Columns X1-X18 are raw financials.")
with st.sidebar.expander("ℹ️ X1-X18 meaning"):
    st.caption("\n".join(f"- **{f}**: {feature_labels.get(f, f)}" for f in RAW_FEATURES))

years_input = st.sidebar.data_editor(
    default_years_df,
    use_container_width=True,
    num_rows="fixed",
    key="years_editor",
)

# --- Main area ---
predict_btn = st.sidebar.button(t["predict_btn"], use_container_width=True)

if predict_btn:
    # Build the full feature vector (level + panel) from the multi-year input
    input_df = build_inference_features(years_input)

    input_df = input_df.reindex(columns=features, fill_value=0)
    # Standardize raw features (X1-X18) using training scaler
    input_df[RAW_FEATURES] = scaler.transform(input_df[RAW_FEATURES])

    # Prediction
    prob = model.predict_proba(input_df)[0][1]
    risk_score = prob * 100

    # --- Risk Score Display ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(t["risk_score_title"])

        # Color based on risk level
        if risk_score >= 70:
            color = "#FF4B4B"
            level = t["high_risk"]
        elif risk_score >= 40:
            color = "#FFA500"
            level = t["medium_risk"]
        else:
            color = "#00CC66"
            level = t["low_risk"]

        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 72px; font-weight: bold; color: {color};">{risk_score:.1f}</div>
                <div style="font-size: 24px; color: {color}; margin-top: -10px;">/ 100</div>
                <div style="font-size: 28px; font-weight: bold; color: {color}; margin-top: 10px;">{level}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Progress bar
        st.progress(float(min(prob, 1.0)))

    st.markdown("---")

    # --- SHAP Explanation ---
    st.markdown(t["shap_title"])

    shap_values = explainer.shap_values(input_df)
    shap_array = np.array(shap_values[0]).flatten() if isinstance(shap_values, list) else np.array(shap_values).flatten()

    # Build analysis table
    analysis = []
    for i, feat in enumerate(features):
        label = feature_labels.get(feat, feat)
        sv = float(shap_array[i])
        val = float(input_df.iloc[0][feat])
        analysis.append({
            t["col_feature"]: f"{label} ({feat})",
            t["col_input"]: val,
            t["col_shap"]: sv,
            "Abs SHAP": abs(sv),
            t["col_impact"]: t["increases_risk"] if sv > 0 else t["decreases_risk"],
        })

    analysis.sort(key=lambda x: x["Abs SHAP"], reverse=True)

    # Top risk factors
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(t["top_factors"])
        for idx, item in enumerate(analysis[:5]):
            direction_icon = "🔴" if item[t["col_shap"]] > 0 else "🟢"
            st.markdown(
                f"**{idx+1}. {item[t['col_feature']]}**  \n"
                f"   Value: `{item[t['col_input']]:.2f}` | SHAP: `{item[t['col_shap']]:.4f}` {direction_icon}"
            )

    with col_right:
        st.markdown(t["all_impacts"])
        display_df = pd.DataFrame([
            {
                t["col_feature"]: a[t["col_feature"]],
                t["col_input"]: f"{a[t['col_input']]:.2f}",
                t["col_shap"]: f"{a[t['col_shap']]:.4f}",
                t["col_impact"]: a[t["col_impact"]],
            }
            for a in analysis
        ])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.info(t["shap_info"])

    st.markdown("---")

    # --- Model Ensemble Comparison ---
    if ensemble_models:
        st.markdown("### 🤖 Model Ensemble Comparison")

        ensemble_results = {"XGBoost": prob}
        for name, m in ensemble_models.items():
            ensemble_results[name] = float(m.predict_proba(input_df)[0][1])

        # Consensus analysis
        all_probs = list(ensemble_results.values())
        avg_prob = np.mean(all_probs)
        std_prob = np.std(all_probs)
        agree_count = sum(1 for p in all_probs if (p >= threshold) == (avg_prob >= threshold))

        col_ens1, col_ens2 = st.columns([2, 1])

        with col_ens1:
            ens_rows = []
            for name, p in ensemble_results.items():
                risk_pct = p * 100
                if risk_pct >= 70:
                    verdict = t["high_risk"]
                elif risk_pct >= 40:
                    verdict = t["medium_risk"]
                else:
                    verdict = t["low_risk"]
                ens_rows.append({
                    "Model": name,
                    "Risk Score": f"{risk_pct:.1f}",
                    "Risk Level": verdict,
                })
            st.dataframe(pd.DataFrame(ens_rows), use_container_width=True, hide_index=True)

        with col_ens2:
            consensus = agree_count == len(all_probs)
            if consensus:
                st.success(f"**Consensus: {agree_count}/{len(all_probs)} models agree**\nHigh prediction confidence.")
            else:
                st.warning(f"**Split: {agree_count}/{len(all_probs)} models agree**\nModerate prediction uncertainty (std: {std_prob:.3f}).")

            st.metric("Ensemble Average", f"{avg_prob * 100:.1f} / 100")

        st.markdown("---")

    # --- Counterfactual Explanation ---
    if train_data is not None and risk_score >= 40:
        st.markdown("### 🔄 What-If Analysis (Counterfactual)")
        st.caption("Minimum changes needed to move this company to **Low Risk** (below 40/100)")

        with st.spinner("Computing counterfactual scenarios..."):
            try:
                # Prepare DiCE data
                X_train_cf = train_data["X_train"].copy()
                y_train_cf = train_data["y_train"].copy()
                train_df_cf = X_train_cf.copy()
                train_df_cf["target"] = y_train_cf

                d = dice_ml.Data(
                    dataframe=train_df_cf,
                    continuous_features=features,
                    outcome_name="target",
                )
                m = dice_ml.Model(model=model, backend="sklearn")
                exp = dice_ml.Dice(d, m, method="random")

                cf_result = exp.generate_counterfactuals(
                    input_df,
                    total_CFs=3,
                    desired_class="opposite",
                    random_seed=42,
                    # Only suggest changes to actionable current-year financials,
                    # not panel-derived features (trend/volatility/years of history).
                    features_to_vary=RAW_FEATURES,
                )

                cf_df = cf_result.cf_examples_list[0].final_cfs_df
                if cf_df is not None and len(cf_df) > 0:
                    for cf_idx in range(len(cf_df)):
                        cf_row = cf_df.iloc[cf_idx]
                        changes = []
                        for feat in features:
                            orig_val = float(input_df.iloc[0][feat])
                            cf_val = float(cf_row[feat])
                            if abs(cf_val - orig_val) > 1e-6:
                                label = feature_labels.get(feat, feat)
                                diff = cf_val - orig_val
                                arrow = "↑" if diff > 0 else "↓"
                                changes.append({
                                    "Feature": f"{label} ({feat})",
                                    "Current": f"{orig_val:.2f}",
                                    "Target": f"{cf_val:.2f}",
                                    "Change": f"{arrow} {abs(diff):.2f}",
                                })

                        if changes:
                            st.markdown(f"**Scenario {cf_idx + 1}** — {len(changes)} change(s):")
                            st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)

                    st.info(
                        "💡 These scenarios show the **smallest changes** to your financial indicators "
                        "that would shift the prediction from bankrupt to non-bankrupt. "
                        "They are model-generated suggestions, not financial advice."
                    )
                else:
                    st.info("No counterfactual scenarios could be generated for this input.")

            except Exception as e:
                st.warning(f"Counterfactual generation failed: {str(e)}")

        st.markdown("---")

    # --- AI Recommendations ---
    st.markdown(t["ai_title"])

    if client is not None:
        with st.spinner(t["ai_spinner"]):
            try:
                input_values = [float(input_df.iloc[0][f]) for f in features]

                # RAG: retrieve industry context
                industry_context = []
                if selected_industry != "Other" and vectordb is not None:
                    # Build query from top risk factors
                    top_factors_query = " ".join([
                        feature_labels.get(features[i], features[i])
                        for i in sorted(range(len(shap_array)), key=lambda i: abs(shap_array[i]), reverse=True)[:5]
                    ])
                    rag_query = f"{selected_industry} industry risks challenges {top_factors_query}"
                    industry_context = retrieve_industry_context(
                        vectordb, client, selected_industry, rag_query, n_results=5
                    )

                advice = generate_advice(
                    client, shap_array, features, feature_labels, input_values, risk_score, lang_code,
                    industry=selected_industry, industry_context=industry_context,
                )
                st.success(t["ai_success"])
                st.markdown(advice)

                # Show referenced context in expander
                if industry_context:
                    with st.expander(t["rag_context_title"]):
                        for i, chunk in enumerate(industry_context, 1):
                            st.markdown(f"**Excerpt {i}:**")
                            st.caption(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                            st.markdown("---")

            except Exception as e:
                st.warning(f"{t['ai_fail']}: {str(e)}")
    else:
        st.info(t["ai_no_key"])

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption(t["footer"])
