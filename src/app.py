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

# --- Paths ---
ROOT_DIR = Path(__file__).parent.parent
MODEL_DIR = ROOT_DIR / "models"

# --- Load model artifacts ---
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "xgb_bankruptcy.pkl")
    features = joblib.load(MODEL_DIR / "features.pkl")
    feature_labels = joblib.load(MODEL_DIR / "feature_labels.pkl")
    explainer = shap.TreeExplainer(model)
    return model, features, feature_labels, explainer


# --- OpenAI client ---
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def generate_advice(client, shap_array, feature_names, feature_labels, input_values, risk_score):
    """Generate AI improvement suggestions based on SHAP analysis."""
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

    prompt = f"""You are a professional financial advisor specializing in corporate bankruptcy risk analysis.

A company has been assessed with a Bankruptcy Risk Score of {risk_score:.0f}/100.

The top factors influencing this prediction (from a SHAP-based ML model) are:

{chr(10).join(details)}

Based on these findings:
1. Briefly explain what these indicators suggest about the company's financial health.
2. Provide 3-5 specific, actionable recommendations to reduce bankruptcy risk.
3. Prioritize the recommendations by impact.

Use a professional but accessible tone. Write in Traditional Chinese (繁體中文)."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional financial advisor specializing in corporate distress and turnaround strategies."},
            {"role": "user", "content": prompt},
        ],
        timeout=30.0,
    )
    return response.choices[0].message.content


# --- Page config ---
st.set_page_config(page_title="Bankruptcy Risk Predictor", page_icon="📉", layout="wide")

# --- Load resources ---
try:
    model, features, feature_labels, explainer = load_model()
except FileNotFoundError:
    st.error("Model files not found. Please run `python src/train_model.py` first.")
    st.stop()

client = get_openai_client()

# --- Header ---
st.title("📉 Corporate Bankruptcy Risk Predictor")
st.caption("Enter financial indicators to predict bankruptcy risk, with AI-powered improvement suggestions")

# --- Sidebar: Input ---
st.sidebar.header("📥 Financial Indicators")
st.sidebar.markdown("Enter the company's financial data below:")

input_data = {}
for feat in features:
    label = feature_labels.get(feat, feat)
    input_data[feat] = st.sidebar.number_input(
        f"{label} ({feat})",
        value=0.0,
        format="%.2f",
        key=feat,
    )

# --- Main area ---
predict_btn = st.sidebar.button("🚀 Predict Bankruptcy Risk", use_container_width=True)

if predict_btn:
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Prediction
    prob = model.predict_proba(input_df)[0][1]
    risk_score = prob * 100

    # --- Risk Score Display ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🎯 Bankruptcy Risk Score")

        # Color based on risk level
        if risk_score >= 70:
            color = "#FF4B4B"
            level = "High Risk"
            level_zh = "高風險"
        elif risk_score >= 40:
            color = "#FFA500"
            level = "Medium Risk"
            level_zh = "中風險"
        else:
            color = "#00CC66"
            level = "Low Risk"
            level_zh = "低風險"

        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 72px; font-weight: bold; color: {color};">{risk_score:.1f}</div>
                <div style="font-size: 24px; color: {color}; margin-top: -10px;">/ 100</div>
                <div style="font-size: 28px; font-weight: bold; color: {color}; margin-top: 10px;">{level} ({level_zh})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Progress bar
        st.progress(float(min(prob, 1.0)))

    st.markdown("---")

    # --- SHAP Explanation ---
    st.markdown("### 📊 Risk Factor Analysis (SHAP)")

    shap_values = explainer.shap_values(input_df)
    shap_array = np.array(shap_values[0]).flatten() if isinstance(shap_values, list) else np.array(shap_values).flatten()

    # Build analysis table
    analysis = []
    for i, feat in enumerate(features):
        label = feature_labels.get(feat, feat)
        sv = float(shap_array[i])
        val = float(input_df.iloc[0][feat])
        analysis.append({
            "Feature": f"{label} ({feat})",
            "Input Value": val,
            "SHAP Value": sv,
            "Abs SHAP": abs(sv),
            "Impact": "🔴 Increases Risk" if sv > 0 else "🟢 Decreases Risk",
        })

    analysis.sort(key=lambda x: x["Abs SHAP"], reverse=True)

    # Top risk factors
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🔍 Top Risk Factors")
        for idx, item in enumerate(analysis[:5]):
            direction_icon = "🔴" if item["SHAP Value"] > 0 else "🟢"
            st.markdown(
                f"**{idx+1}. {item['Feature']}**  \n"
                f"   Value: `{item['Input Value']:.2f}` | SHAP: `{item['SHAP Value']:.4f}` {direction_icon}"
            )

    with col_right:
        st.markdown("#### 📋 All Feature Impacts")
        display_df = pd.DataFrame([
            {
                "Feature": a["Feature"],
                "Input Value": f"{a['Input Value']:.2f}",
                "SHAP Value": f"{a['SHAP Value']:.4f}",
                "Impact": a["Impact"],
            }
            for a in analysis
        ])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.info(
        "💡 **How to read SHAP values:**\n"
        "- **Positive (+)**: This feature increases bankruptcy risk\n"
        "- **Negative (-)**: This feature decreases bankruptcy risk\n"
        "- **Larger absolute value**: Stronger influence on prediction"
    )

    st.markdown("---")

    # --- AI Recommendations ---
    st.markdown("### 💡 AI Improvement Recommendations")

    if client is not None:
        with st.spinner("Generating personalized recommendations..."):
            try:
                input_values = [float(input_df.iloc[0][f]) for f in features]
                advice = generate_advice(
                    client, shap_array, features, feature_labels, input_values, risk_score
                )
                st.success("Recommendations generated successfully!")
                st.markdown(advice)
            except Exception as e:
                st.warning(f"Failed to generate AI recommendations: {str(e)}")
    else:
        st.info(
            "💡 AI recommendations require an OpenAI API key. "
            "Set `OPENAI_API_KEY` in the `.env` file to enable this feature."
        )

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("📁 Corporate Bankruptcy Risk Predictor | Model: XGBoost + SHAP")
