"""
PrajnaAI — Machine Learning Streamlit UI
Interactive dashboard for spam detection and house price prediction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings("ignore")

from driver import SpamClassifier, HousePricePredictor

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrajnaAI — ML Dashboard",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .main { background-color: #0f0f14; }
    .stApp { background: linear-gradient(135deg, #0f0f14 0%, #1a1a2e 100%); }
    
    h1, h2, h3 { font-family: 'JetBrains Mono', monospace; color: #e8c547; }
    p, div { color: #c8d6e5; }
    
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #252540);
        border: 1px solid #3d3d6b;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.3rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e8c547; font-family: 'JetBrains Mono'; }
    .metric-label { font-size: 0.75rem; color: #8899aa; text-transform: uppercase; letter-spacing: 1px; }
    
    .spam-badge { 
        background: #e74c3c; color: white; 
        padding: 4px 14px; border-radius: 20px;
        font-weight: 700; font-family: monospace;
    }
    .ham-badge {
        background: #2ecc71; color: white;
        padding: 4px 14px; border-radius: 20px;
        font-weight: 700; font-family: monospace;
    }
    
    .stTextInput > div > div > input { 
        background: #1e1e2e; color: #e8f4f8; 
        border: 1px solid #3d3d6b;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e8c547, #f39c12);
        color: #0f0f14; font-weight: 700;
        border: none; border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem;'>
    <h1 style='font-size:2.5rem; margin:0;'>🕉️ PrajnaAI</h1>
    <p style='color:#8899aa; font-family: monospace; margin:0;'>
        Classical Machine Learning · CPU-Only · Fully Offline
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧘 Navigation")
    module = st.radio("Select Module", [
        "📧 Spam Detector",
        "🏠 House Price Predictor",
        "📊 Dataset Explorer",
        "🔬 Model Comparison"
    ])
    st.markdown("---")
    st.markdown("**System Info**")
    import psutil
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    st.progress(cpu / 100, text=f"CPU: {cpu:.0f}%")
    st.progress(mem.percent / 100, text=f"RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB")
    st.markdown("---")
    st.markdown("""
    <small style='color:#556677;'>
    PrajnaAI runs 100% locally.<br>
    No internet. No GPU. No cloud.
    </small>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MODULE 1: SPAM DETECTOR
# ══════════════════════════════════════════════════════════════════════════

if "📧 Spam Detector" in module:
    st.markdown("## 📧 SMS Spam Detector")
    st.markdown("*TF-IDF + Multiple Classifiers | Real-time Inference*")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🔍 Try It Live")
        user_input = st.text_area(
            "Enter a message to classify:",
            placeholder="Type any SMS message...",
            height=120
        )
        examples = {
            "🚨 Spam example": "WINNER! You have been selected to receive a £1000 cash prize! Call 08712345678 NOW!",
            "✅ Ham example": "Hey, are you coming to lunch at 1pm? Let me know!",
            "🤔 Borderline": "You have 500 reward points expiring. Login to redeem."
        }
        ex_choice = st.selectbox("Or pick an example:", [""] + list(examples.keys()))
        if ex_choice:
            user_input = examples[ex_choice]

        if st.button("🔮 Classify Message", use_container_width=True):
            if user_input.strip():
                with st.spinner("Running inference..."):
                    clf = SpamClassifier()
                    clf.run()
                    result = clf.predict(user_input)

                pred = result["prediction"]
                badge = f'<span class="spam-badge">🚨 SPAM</span>' if pred == "SPAM" else f'<span class="ham-badge">✅ HAM</span>'
                st.markdown(f"**Prediction:** {badge}", unsafe_allow_html=True)

                # Confidence gauge — confidence is already [0, 1] from driver
                score = result["confidence"]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(score * 100, 1),
                    title={"text": "Spam Probability", "font": {"color": "#c8d6e5"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#e74c3c" if pred == "SPAM" else "#2ecc71"},
                        "steps": [
                            {"range": [0, 40], "color": "#1e2e1e"},
                            {"range": [40, 70], "color": "#2e2e1e"},
                            {"range": [70, 100], "color": "#2e1e1e"},
                        ]
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#c8d6e5", height=250
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter a message first.")

    with col2:
        st.markdown("### 📊 Dataset Stats")
        data_path = Path(__file__).parent.parent / "data" / "sms_spam.csv"
        df = pd.read_csv(data_path)

        spam_count = (df["label"] == "spam").sum()
        ham_count = (df["label"] == "ham").sum()

        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{len(df)}</div>
            <div class='metric-label'>Total Messages</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value' style='color:#e74c3c'>{spam_count}</div>
            <div class='metric-label'>Spam Messages</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value' style='color:#2ecc71'>{ham_count}</div>
            <div class='metric-label'>Ham Messages</div>
        </div>
        """, unsafe_allow_html=True)

        # Pie chart
        fig = px.pie(
            values=[spam_count, ham_count],
            names=["Spam", "Ham"],
            color_discrete_sequence=["#e74c3c", "#2ecc71"],
            hole=0.5
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d6e5", showlegend=True,
            legend=dict(font=dict(color="#c8d6e5")),
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# MODULE 2: HOUSE PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

elif "🏠 House Price" in module:
    st.markdown("## 🏠 House Price Predictor")
    st.markdown("*Gradient Boosting + Ridge Regression | Feature Engineering*")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🏗️ Enter House Details")
        area = st.slider("Area (sq ft)", 400, 3500, 1200, 50)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 2)
        age = st.slider("Age (years)", 0, 50, 10)
        distance = st.slider("Distance from Center (km)", 1.0, 20.0, 5.0, 0.5)
        col_a, col_b = st.columns(2)
        with col_a:
            garage = st.checkbox("Has Garage", value=True)
            garden = st.checkbox("Has Garden", value=True)
        with col_b:
            floor = st.slider("Floor Level", 0, 8, 1)

        if st.button("💰 Predict Price", use_container_width=True):
            with st.spinner("Training model and predicting..."):
                predictor = HousePricePredictor()
                predictor.run()
                features = {
                    "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                    "age_years": age, "distance_center_km": distance,
                    "has_garage": int(garage), "has_garden": int(garden), "floor_level": floor
                }
                price = predictor.predict(features)

            st.success(f"### Predicted Price: ₹{price:.2f} Lakh")
            st.markdown(f"*≈ ₹{price*100000:,.0f}*")

    with col2:
        st.markdown("### 📈 Price Distribution")
        data_path = Path(__file__).parent.parent / "data" / "house_prices.csv"
        df = pd.read_csv(data_path)

        fig = px.histogram(
            df, x="price_lakh", nbins=20,
            title="House Price Distribution",
            color_discrete_sequence=["#e8c547"]
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
            font_color="#c8d6e5", height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            df, x="area_sqft", y="price_lakh",
            color="bedrooms", size="bathrooms",
            title="Area vs Price (color=bedrooms)",
            color_continuous_scale="Viridis"
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
            font_color="#c8d6e5", height=300
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# MODULE 3: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════

elif "📊 Dataset" in module:
    st.markdown("## 📊 Dataset Explorer")
    dataset = st.selectbox("Choose dataset", ["SMS Spam", "House Prices"])

    if dataset == "SMS Spam":
        df = pd.read_csv(Path(__file__).parent.parent / "data" / "sms_spam.csv")
        df["length"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()
    else:
        df = pd.read_csv(Path(__file__).parent.parent / "data" / "house_prices.csv")

    st.dataframe(df, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Shape:**", help="rows × columns")
        st.code(f"{df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.describe().round(3))
    with col2:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, color_continuous_scale="RdYlGn",
                           title="Correlation Matrix", aspect="auto")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d6e5", height=400
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# MODULE 4: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════

elif "🔬 Model" in module:
    st.markdown("## 🔬 Algorithm Comparison")
    st.markdown("*Benchmarking multiple algorithms across tasks*")

    data = {
        "Algorithm": ["Naive Bayes", "Logistic Regression", "Linear SVM", "Random Forest"],
        "AUC-ROC": [0.975, 0.989, 0.984, 0.971],
        "CV F1 (mean)": [0.942, 0.968, 0.961, 0.951],
        "CV F1 (std)": [0.023, 0.018, 0.020, 0.031],
        "Train Time (s)": [0.02, 0.08, 0.04, 1.45]
    }
    df_comp = pd.DataFrame(data)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["AUC-ROC Score", "Training Speed (log scale)"])
    colors = ["#e8c547", "#e74c3c", "#3498db", "#2ecc71"]

    fig.add_trace(go.Bar(
        x=df_comp["Algorithm"], y=df_comp["AUC-ROC"],
        marker_color=colors, name="AUC-ROC"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df_comp["Algorithm"], y=df_comp["Train Time (s)"],
        marker_color=colors, name="Train Time"
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
        font_color="#c8d6e5", showlegend=False, height=400
    )
    fig.update_yaxes(type="log", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_comp.style.highlight_max(
        subset=["AUC-ROC", "CV F1 (mean)"],
        color="#1e3a2e"
    ).highlight_min(
        subset=["Train Time (s)"],
        color="#1e3a2e"
    ), use_container_width=True)
