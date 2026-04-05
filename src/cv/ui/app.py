"""
PrajnaAI — Computer Vision Streamlit UI
Face detection + motion analysis dashboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time

from driver import HaarFaceDetector, MotionDetector, ImageAnalyzer

st.set_page_config(page_title="PrajnaAI — CV", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0a1628; }
    h1, h2, h3 { color: #4fc3f7; font-family: monospace; }
    .cv-metric { background: #0d2137; border: 1px solid #1565c0; border-radius: 8px; padding: 1rem; text-align: center; }
    .cv-value { font-size: 1.8rem; color: #4fc3f7; font-weight: bold; font-family: monospace; }
    .cv-label { font-size: 0.7rem; color: #546e7a; text-transform: uppercase; letter-spacing: 1px; }
    .stButton > button { background: #1565c0; color: white; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 👁️ Computer Vision — Classical CPU Methods")
st.markdown("*Haar Cascades + Background Subtraction. No deep learning. Runs at 15+ FPS on i3.*")

with st.sidebar:
    st.markdown("### 🎛️ Module")
    module = st.radio("", ["🎭 Face Detection", "🎬 Motion Detection", "🔍 Image Analysis"])
    st.markdown("---")
    st.markdown("""
    **Tech Stack**
    - OpenCV 4.8+
    - Haar Cascade XML
    - MOG2 Background Subtractor
    - Canny Edge Detector
    - Laplacian Sharpness
    """)


# ── FACE DETECTION ─────────────────────────────────────────────────────────
if "🎭 Face" in module:
    st.markdown("### 🎭 Haar Cascade Face Detection")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**How it works:**")
        st.markdown("""
        Haar Cascades use a sliding window approach trained on thousands of face/non-face 
        images. The cascade architecture rejects non-face regions early, making it very 
        fast on CPU. Originally proposed by Viola & Jones (2001).
        
        - Scale factor: 1.1 (10% size steps)
        - Min neighbors: 5 (reduces false positives)
        - Min face size: 30×30 pixels
        """)
    
    with col2:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        use_synthetic = st.checkbox("Use synthetic test image", value=True)
    
    if st.button("🔍 Detect Faces", use_container_width=True):
        try:
            detector = HaarFaceDetector()
            
            if uploaded and not use_synthetic:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                # Synthetic test image
                img = np.ones((480, 640, 3), dtype=np.uint8) * 190
                # Simulated face shapes
                cv2.ellipse(img, (200, 200), (70, 90), 0, 0, 360, (210, 180, 160), -1)
                cv2.circle(img, (175, 185), 12, (60, 40, 30), -1)
                cv2.circle(img, (225, 185), 12, (60, 40, 30), -1)
                cv2.ellipse(img, (450, 220), (60, 75), 0, 0, 360, (205, 175, 155), -1)
                cv2.circle(img, (430, 208), 10, (50, 35, 25), -1)
                cv2.circle(img, (470, 208), 10, (50, 35, 25), -1)
                st.info("Using synthetic face-like image (Haar cascades detect real photos better)")
            
            t0 = time.time()
            faces, eyes = detector.detect(img)
            elapsed = (time.time() - t0) * 1000
            
            annotated = detector.annotate(img, faces, eyes)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
            with c2:
                st.image(annotated_rgb, caption=f"Detected ({len(faces)} faces)", use_column_width=True)
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(f'<div class="cv-metric"><div class="cv-value">{len(faces)}</div><div class="cv-label">Faces Found</div></div>', unsafe_allow_html=True)
            mc2.markdown(f'<div class="cv-metric"><div class="cv-value">{elapsed:.0f}ms</div><div class="cv-label">Detection Time</div></div>', unsafe_allow_html=True)
            mc3.markdown(f'<div class="cv-metric"><div class="cv-value">{1000/elapsed:.0f}</div><div class="cv-label">Est. FPS</div></div>', unsafe_allow_html=True)
        
        except FileNotFoundError as e:
            st.error(f"Haar cascade not found: {e}")
            st.info("Run `bash scripts/setup.sh` to download cascade files.")


# ── MOTION DETECTION ───────────────────────────────────────────────────────
elif "🎬 Motion" in module:
    st.markdown("### 🎬 Motion Detection — MOG2 Background Subtraction")
    
    st.markdown("""
    **MOG2 (Mixture of Gaussians v2):** Models each pixel as a mixture of Gaussians.  
    Pixels deviating from the learned background are flagged as foreground (motion).
    Shadows are detected and filtered separately.
    """)
    
    n_frames = st.slider("Simulation frames", 30, 100, 60)
    
    if st.button("▶️ Run Motion Simulation", use_container_width=True):
        with st.spinner("Simulating video sequence..."):
            detector = MotionDetector()
            events = detector.simulate_video(n_frames=n_frames)
        
        df_events = {"Frame": [e["frame"] for e in events],
                     "Motion Ratio": [e["motion_ratio"] for e in events],
                     "Objects": [e["objects"] for e in events]}
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_events["Frame"], y=df_events["Motion Ratio"],
            fill="tozeroy", line=dict(color="#e74c3c", width=2),
            name="Motion Intensity"
        ))
        fig.add_hline(y=0.01, line_dash="dash", line_color="#f39c12",
                     annotation_text="Motion Threshold")
        fig.update_layout(
            title="Motion Intensity Over Time",
            xaxis_title="Frame", yaxis_title="Motion Area Ratio",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.2)",
            font_color="#b0bec5", height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        motion_frames = sum(1 for e in events if e["motion_ratio"] > 0.01)
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames Processed", n_frames)
        c2.metric("Motion Frames", motion_frames)
        c3.metric("Motion Events", len(detector.motion_events))


# ── IMAGE ANALYSIS ─────────────────────────────────────────────────────────
elif "🔍 Image Analysis" in module:
    st.markdown("### 🔍 Classical Image Analysis Pipeline")
    
    uploaded = st.file_uploader("Upload image for analysis", type=["jpg", "jpeg", "png"])
    
    if st.button("🔬 Analyze", use_container_width=True):
        analyzer = ImageAnalyzer()
        
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.rectangle(img, (50, 50), (150, 150), (100, 200, 100), -1)
            cv2.circle(img, (300, 150), 80, (200, 100, 100), -1)
            cv2.line(img, (0, 200), (400, 200), (200, 200, 50), 3)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            st.info("Using synthetic test image")
        
        analysis = analyzer.analyze(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
        with c2:
            st.image(edges, caption="Canny Edges")
        
        fig = px.bar(x=list(range(256)), y=hist.tolist(),
                    title="Grayscale Histogram",
                    labels={"x": "Pixel Intensity", "y": "Count"},
                    color_discrete_sequence=["#4fc3f7"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#b0bec5", height=280)
        st.plotly_chart(fig, use_container_width=True)
        
        cols = st.columns(3)
        cols[0].metric("Brightness", f"{analysis['mean_brightness']:.0f}/255")
        cols[1].metric("Sharpness", f"{analysis['sharpness_score']:.0f}", 
                      delta="Sharp" if not analysis['is_blurry'] else "Blurry")
        cols[2].metric("Contours", analysis["contour_count"])
        
        st.json(analysis)
