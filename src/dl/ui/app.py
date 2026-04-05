"""
PrajnaAI — Deep Learning Streamlit UI
PyTorch training visualizer with live loss curves.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

from driver import LSTMClassifier, SentimentDataset, TabularMLP, train_lstm_classifier, train_tabular_mlp

st.set_page_config(page_title="PrajnaAI — Deep Learning", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stApp { background: #12001f; }
    h1, h2, h3 { color: #ce93d8; font-family: 'Courier New', monospace; }
    .arch-box { background: #1a0a2e; border: 1px solid #6a1b9a; border-radius: 8px; padding: 1rem; }
    .train-metric { background: #1a0a2e; border-left: 4px solid #ce93d8; padding: 0.8rem; border-radius: 0 8px 8px 0; }
    .stButton > button { background: #6a1b9a; color: white; border-radius: 6px; }
    .stProgress > div > div > div { background: #ce93d8; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧠 Deep Learning — CPU PyTorch")
st.markdown(f"*PyTorch {torch.__version__} | Device: CPU | Threads: {torch.get_num_threads()}*")

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_type = st.radio("Model", ["LSTM Classifier", "Tabular MLP"])
    epochs = st.slider("Epochs", 10, 100, 30)
    lr = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01, 0.1], value=0.001)
    batch_size = st.slider("Batch Size", 4, 32, 8)
    st.markdown("---")
    st.markdown("""
    **CPU Tips:**
    - Keep batch size small
    - Use fewer epochs for quick tests
    - MLP trains faster than LSTM
    """)

st.markdown(f"### 🏗️ Model: {model_type}")

if model_type == "LSTM Classifier":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **Architecture:**
        ```
        Embedding(vocab, 32)
            ↓
        Bidirectional LSTM(32→64)
            ↓
        Concat[forward, backward]
            ↓
        Linear(128→32) + ReLU
            ↓
        Dropout(0.3)
            ↓
        Linear(32→1) + Sigmoid
        ```
        """)
    with col2:
        st.markdown("""
        **Task:** Sentiment classification (positive/negative)  
        **Dataset:** 24 product reviews  
        **Vocab size:** ~120 tokens  
        **Max length:** 15 tokens  
        
        **Why Bidirectional?**  
        Each word is influenced by both preceding *and* following context.
        Bidirectional LSTM captures both directions simultaneously.
        """)

else:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **Architecture:**
        ```
        Input(8 features)
            ↓
        Linear(8→128) + BatchNorm + ReLU
            ↓
        Dropout(0.2)
            ↓
        Linear(128→64) + BatchNorm + ReLU
            ↓
        Dropout(0.2)
            ↓
        Linear(64→32) + ReLU
            ↓
        Linear(32→1) [price output]
        ```
        """)
    with col2:
        st.markdown("""
        **Task:** House price regression  
        **Features:** 8 (area, bedrooms, age, distance...)  
        **Loss:** Huber Loss (robust to outliers)  
        **Optimizer:** AdamW with weight decay
        
        **Why BatchNorm?**  
        Normalizes layer inputs, stabilizing training and 
        allowing higher learning rates on tabular data.
        """)

if st.button("🚀 Train Model", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    chart_placeholder = st.empty()
    
    # We'll train step by step and update the chart
    history = {"loss": [], "val_loss" if model_type == "Tabular MLP" else "accuracy": []}
    
    with st.spinner(f"Training {model_type} on CPU..."):
        t0 = time.time()
        if model_type == "LSTM Classifier":
            result = train_lstm_classifier(epochs=epochs)
            history = result["history"]
            metric_label = "Accuracy"
            metric_key = "accuracy"
        else:
            result = train_tabular_mlp(epochs=epochs)
            history = result["history"]
            metric_label = "Val Loss"
            metric_key = "val_loss"
    
    elapsed = time.time() - t0
    progress_bar.progress(1.0)
    status_text.success(f"✅ Training complete in {elapsed:.1f}s")
    
    # Plot training curves
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history["loss"], name="Train Loss",
        line=dict(color="#ce93d8", width=2)
    ))
    if metric_key in history:
        y2_data = history[metric_key]
        y2_name = metric_label
        # Normalize accuracy to 0-1 range for display or use raw val_loss
        fig.add_trace(go.Scatter(
            y=y2_data, name=y2_name,
            line=dict(color="#80cbc4", width=2),
            yaxis="y2"
        ))
    
    fig.update_layout(
        title=f"{model_type} — Training Curves",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis2=dict(title=metric_label, overlaying="y", side="right"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
        font_color="#ce93d8", legend=dict(font=dict(color="#e1bee7")),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Final metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Loss", f"{history['loss'][-1]:.4f}")
    if model_type == "LSTM Classifier":
        col2.metric("Final Accuracy", f"{history['accuracy'][-1]:.1%}")
        col3.metric("Training Time", f"{elapsed:.1f}s")
    else:
        col2.metric("Test R²", f"{result['r2']:.4f}")
        col3.metric("Training Time", f"{elapsed:.1f}s")
    
    # Inference demo
    st.markdown("### 🔮 Inference Demo")
    if model_type == "LSTM Classifier":
        dataset = SentimentDataset()
        model = LSTMClassifier(vocab_size=len(dataset.vocab))
        model.load_state_dict(torch.load(
            Path(__file__).parent.parent / "models" / "lstm_classifier.pt",
            map_location="cpu"
        ))
        model.eval()
        
        test_text = st.text_input("Test a review:", "This product is absolutely amazing!")
        if test_text:
            x = torch.tensor([dataset.encode(test_text)], dtype=torch.long)
            with torch.no_grad():
                prob = model(x).item()
            sentiment = "😊 Positive" if prob > 0.5 else "😞 Negative"
            st.metric("Sentiment", sentiment, delta=f"confidence: {prob:.2%}")
