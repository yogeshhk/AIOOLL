"""
PrajnaAI — LLM Inference Streamlit UI
Full chat interface + model benchmarking dashboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json

from driver import OllamaClient, ChatSession, ModelBenchmarker, BENCHMARK_PROMPTS

st.set_page_config(
    page_title="PrajnaAI — LLM Chat",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    .stApp { background: #0d1117; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #58a6ff; }
    .chat-bubble-user {
        background: #1f3a5f; border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1.2rem; margin: 0.5rem 0; max-width: 80%;
        margin-left: auto; color: #e6edf3;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .chat-bubble-ai {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1.2rem; margin: 0.5rem 0; max-width: 85%;
        color: #e6edf3; font-family: 'IBM Plex Sans', sans-serif;
    }
    .model-badge {
        background: #238636; color: white; padding: 2px 10px;
        border-radius: 12px; font-size: 0.75rem; font-family: monospace;
    }
    .metric-box {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 1rem; text-align: center;
    }
    .stButton > button { background: #238636; color: white; border: none; border-radius: 6px; }
    .stTextInput > div > div > input { background: #161b22; color: #e6edf3; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦙 PrajnaAI LLM")
    
    client = OllamaClient()
    ollama_ok = client.is_running()
    
    if ollama_ok:
        st.success("✅ Ollama Connected")
        models = [m["name"] for m in client.list_models()]
    else:
        st.error("❌ Ollama Offline")
        st.code("ollama serve")
        models = ["gemma2:2b", "qwen2:1.5b"]  # fallback for UI demo
    
    selected_model = st.selectbox("🤖 Model", models if models else ["No models found"])
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("📏 Max Tokens", 64, 1024, 512, 64)
    
    st.markdown("---")
    module = st.radio("📍 Module", ["💬 Chat", "🏎️ Benchmark", "🎯 Prompt Patterns"])
    
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.session_state.session = None
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# CHAT MODULE
# ══════════════════════════════════════════════════════════════════════════

if "💬 Chat" in module:
    st.markdown("## 💬 Chat with Local LLM")
    st.markdown(f'<span class="model-badge">🦙 {selected_model}</span> Running 100% locally on CPU', 
                unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session" not in st.session_state or st.session_state.session is None:
        st.session_state.session = ChatSession(model=selected_model)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                meta = msg.get("meta", "")
                st.markdown(
                    f'<div class="chat-bubble-ai">🤖 {msg["content"]}'
                    f'<br><small style="color:#58a6ff">{meta}</small></div>',
                    unsafe_allow_html=True
                )
    
    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("Message", placeholder="Ask anything...", label_visibility="collapsed", key="chat_input")
    with col2:
        send_btn = st.button("Send →", use_container_width=True)
    
    if send_btn and user_input.strip() and ollama_ok:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner(f"🦙 {selected_model} thinking..."):
            t0 = time.time()
            response = st.session_state.session.chat(user_input)
            latency = time.time() - t0
        
        words = len(response.split())
        meta = f"⚡ {latency:.1f}s | ~{words/latency*0.75:.0f} tok/s | {words} words"
        st.session_state.messages.append({"role": "assistant", "content": response, "meta": meta})
        st.rerun()
    elif send_btn and not ollama_ok:
        st.error("Start Ollama first: `ollama serve`")
    
    # Quick prompts
    st.markdown("**Quick prompts:**")
    quick_cols = st.columns(3)
    quick_prompts = [
        "Explain CPU quantization in 2 sentences",
        "Write a Python hello world",
        "What Linux command shows RAM usage?",
    ]
    for i, prompt in enumerate(quick_prompts):
        if quick_cols[i].button(prompt[:30] + "...", key=f"quick_{i}"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            if ollama_ok:
                response = st.session_state.session.chat(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# BENCHMARK MODULE
# ══════════════════════════════════════════════════════════════════════════

elif "🏎️ Benchmark" in module:
    st.markdown("## 🏎️ Model Speed Benchmark")
    
    if not ollama_ok:
        st.error("Ollama must be running for benchmarks.")
    else:
        all_models = [m["name"] for m in client.list_models()]
        bench_models = st.multiselect("Select models to benchmark", all_models, default=all_models[:2])
        num_prompts = st.slider("Number of prompts", 1, 5, 2)
        
        if st.button("🚀 Run Benchmark", use_container_width=True) and bench_models:
            results = []
            progress = st.progress(0)
            total = len(bench_models) * num_prompts
            done = 0
            
            for model in bench_models:
                for i, prompt in enumerate(BENCHMARK_PROMPTS[:num_prompts]):
                    with st.spinner(f"Testing {model} — Prompt {i+1}..."):
                        try:
                            r = client.generate(model, prompt, max_tokens=200)
                            results.append({
                                "Model": model,
                                "Prompt": f"P{i+1}",
                                "Tokens/sec": round(r["tokens_per_second"], 1),
                                "Latency (s)": round(r["total_duration_s"], 2),
                                "Response Words": len(r["response"].split())
                            })
                        except Exception as e:
                            st.error(f"Error with {model}: {e}")
                    done += 1
                    progress.progress(done / total)
            
            if results:
                df = pd.DataFrame(results)
                
                # Leaderboard
                leaderboard = df.groupby("Model").agg({
                    "Tokens/sec": "mean",
                    "Latency (s)": "mean"
                }).sort_values("Tokens/sec", ascending=False).round(2)
                
                st.markdown("### 🏆 Leaderboard")
                st.dataframe(leaderboard, use_container_width=True)
                
                fig = px.bar(leaderboard.reset_index(), x="Model", y="Tokens/sec",
                            color="Tokens/sec", color_continuous_scale="Viridis",
                            title="Average Tokens per Second by Model")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Raw Results")
                st.dataframe(df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PROMPT PATTERNS MODULE
# ══════════════════════════════════════════════════════════════════════════

elif "🎯 Prompt" in module:
    st.markdown("## 🎯 Prompt Engineering Patterns")
    
    pattern = st.selectbox("Pattern", ["Zero-Shot", "Few-Shot", "Chain-of-Thought", "Structured Output"])
    
    if pattern == "Zero-Shot":
        task = st.text_area("Task", "Explain what GGUF quantization is in simple terms.")
        if st.button("Run") and ollama_ok:
            with st.spinner("Running..."):
                result = client.generate(selected_model, task, max_tokens=300)
            st.markdown(f"**Response:**\n\n{result['response']}")
            st.caption(f"⚡ {result['total_duration_s']:.1f}s | {result['tokens_per_second']:.1f} tok/s")
    
    elif pattern == "Chain-of-Thought":
        problem = st.text_area("Problem", "If Ollama generates 8 tokens/sec and I need a 500-token response, how long does it take?")
        if st.button("Solve Step by Step") and ollama_ok:
            prompt = f"Solve this step by step:\n\nProblem: {problem}\n\nLet me think step by step:\nStep 1:"
            with st.spinner("Reasoning..."):
                result = client.generate(selected_model, prompt, temperature=0.3, max_tokens=400)
            st.markdown(f"**Chain of Thought:**\n\n{result['response']}")
    
    elif pattern == "Structured Output":
        task = st.text_area("Task", "List 3 CPU-friendly AI models with their RAM requirements.")
        schema = '{"models": [{"name": str, "ram_gb": int, "use_case": str}]}'
        st.code(schema, language="json")
        if st.button("Generate JSON") and ollama_ok:
            prompt = f"{task}\n\nRespond ONLY with valid JSON:\n{schema}"
            with st.spinner("Generating..."):
                result = client.generate(selected_model, prompt, temperature=0.1, max_tokens=300)
            try:
                parsed = json.loads(result["response"].strip())
                st.json(parsed)
            except:
                st.code(result["response"])
