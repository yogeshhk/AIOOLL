"""
PrajnaAI — RAG Pipeline Streamlit UI
Local document Q&A with ChromaDB + Ollama.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import time

from driver import LocalRAGPipeline, KB_DIR

st.set_page_config(page_title="PrajnaAI — RAG", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0f1e2e 100%); }
    h1, h2, h3 { color: #64b5f6; font-family: 'Courier New', monospace; }
    .source-card { background: #0d1f2d; border-left: 3px solid #64b5f6; padding: 0.8rem; margin: 0.4rem 0; border-radius: 0 8px 8px 0; color: #b0bec5; font-size: 0.85rem; }
    .answer-box { background: #0d2a1a; border: 1px solid #2e7d32; border-radius: 8px; padding: 1.2rem; color: #e8f5e9; margin: 0.5rem 0; }
    .stButton > button { background: #1565c0; color: white; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📚 Local RAG — Document Q&A")
st.markdown("*Ask questions answered from your local knowledge base. No internet. No cloud.*")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    llm_model = st.selectbox("LLM Model", ["gemma2:2b", "qwen2:1.5b", "phi3:mini"])
    embed_model = st.selectbox("Embedding Model", ["nomic-embed-text"])
    force_rebuild = st.checkbox("Force rebuild index")
    
    st.markdown("---")
    st.markdown("### 📁 Knowledge Base")
    kb_files = list(KB_DIR.glob("*.txt")) + list(KB_DIR.glob("*.pdf"))
    for f in kb_files:
        st.markdown(f"📄 `{f.name}` ({f.stat().st_size // 1024}KB)")
    
    st.markdown("---")
    st.markdown("### ➕ Add Document")
    new_doc = st.text_area("Paste text to add:", height=100)
    doc_source = st.text_input("Source name:", "user_input")
    if st.button("Add to KB"):
        if "pipeline" in st.session_state and new_doc.strip():
            chunks = st.session_state.pipeline.add_document(new_doc, doc_source)
            st.success(f"Added {chunks} chunks!")

# ── Initialize Pipeline ────────────────────────────────────────────────────
if "pipeline" not in st.session_state or st.session_state.get("pipeline_model") != llm_model:
    with st.spinner("Initializing RAG pipeline..."):
        try:
            pipeline = LocalRAGPipeline(llm_model=llm_model, embed_model=embed_model)
            pipeline.build_index(force_rebuild=force_rebuild)
            pipeline.build_chain()
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_model = llm_model
            st.session_state.rag_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.info("Make sure Ollama is running and models are pulled.")
            st.session_state.rag_initialized = False

# ── Q&A Interface ─────────────────────────────────────────────────────────
if st.session_state.get("rag_initialized"):
    st.success("✅ RAG Pipeline Ready")
    
    # Example questions
    st.markdown("**Example questions:**")
    examples = [
        "What models are recommended for CPU inference?",
        "How does GGUF quantization work?",
        "What is Ollama and how do I start it?",
        "What are performance optimization tips for CPU AI?",
    ]
    cols = st.columns(2)
    if "selected_q" not in st.session_state:
        st.session_state.selected_q = ""
    
    for i, q in enumerate(examples):
        if cols[i % 2].button(q, key=f"ex_{i}"):
            st.session_state.selected_q = q
    
    question = st.text_input(
        "Your question:",
        value=st.session_state.get("selected_q", ""),
        placeholder="Ask anything about the knowledge base..."
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_btn = st.button("🔍 Ask", use_container_width=True)
    
    if ask_btn and question.strip():
        with st.spinner(f"🤔 Searching and generating with {llm_model}..."):
            t0 = time.time()
            result = st.session_state.pipeline.query(question)
            elapsed = time.time() - t0
        
        st.markdown(f"### 💡 Answer")
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
        st.caption(f"⚡ Generated in {elapsed:.2f}s | Model: {llm_model}")
        
        if result.get("contexts"):
            st.markdown("### 📎 Source Contexts")
            for i, ctx in enumerate(result["contexts"]):
                src = result["sources"][i] if i < len(result["sources"]) else "unknown"
                src_name = Path(src).name if src != "unknown" else "unknown"
                st.markdown(f'<div class="source-card"><strong>📄 {src_name}</strong><br>{ctx}</div>',
                           unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []
    
    if ask_btn and question.strip():
        st.session_state.history.append({"q": question, "a": result["answer"][:100] + "..."})
    
    if st.session_state.history:
        with st.expander("📜 Query History"):
            for item in reversed(st.session_state.history[-5:]):
                st.markdown(f"**Q:** {item['q']}")
                st.markdown(f"**A:** {item['a']}")
                st.divider()
else:
    st.warning("Pipeline not initialized. Check Ollama connection.")
    st.code("ollama serve\nollama pull nomic-embed-text\nollama pull gemma2:2b")
