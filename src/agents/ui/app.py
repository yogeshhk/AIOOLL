"""
PrajnaAI — Agents Streamlit UI
LangGraph agent workflow visualizer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import time
import json

from driver import ResearchAgent, CodeReviewAgent, calculator, word_counter, python_syntax_checker

st.set_page_config(page_title="PrajnaAI — Agents", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp { background: #1a0a2e; }
    h1, h2, h3 { color: #ce93d8; font-family: monospace; }
    .step-card { background: #2d1b4e; border: 1px solid #7b1fa2; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .step-label { color: #ce93d8; font-size: 0.75rem; text-transform: uppercase; font-family: monospace; }
    .step-content { color: #e1bee7; margin-top: 0.5rem; }
    .tool-result { background: #1a2e1a; border: 1px solid #2e7d32; border-radius: 6px; padding: 0.6rem; color: #a5d6a7; font-family: monospace; font-size: 0.9rem; }
    .stButton > button { background: #7b1fa2; color: white; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🤖 AI Agents — LangGraph Workflows")
st.markdown("*Multi-step autonomous agents running locally. No cloud, no API keys.*")

with st.sidebar:
    st.markdown("### ⚙️ Agent Config")
    model = st.selectbox("LLM Model", ["gemma2:2b", "qwen2:1.5b", "tinyllama:1.1b"])
    st.markdown("---")
    agent_type = st.radio("Agent", ["🔬 Research Agent", "💻 Code Review Agent", "🔧 Tool Demo"])

# ── Research Agent ─────────────────────────────────────────────────────────
if "🔬 Research" in agent_type:
    st.markdown("### 🔬 Research Agent")
    st.markdown("*LangGraph linear pipeline: Plan → Research → Synthesize*")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("Research Topic:", "CPU-optimized AI inference techniques")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Start Research", use_container_width=True)
    
    # Visualize the graph
    st.markdown("**Agent Graph:**")
    st.markdown("""
    ```
    START → [Plan] → [Research] → [Synthesize] → END
    ```
    """)
    
    if run_btn and topic:
        agent = ResearchAgent(model=model)
        
        progress = st.progress(0)
        steps_container = st.container()
        
        with st.spinner("Agent working..."):
            t0 = time.time()
            result = agent.run(topic)
            elapsed = time.time() - t0
        
        progress.progress(1.0)
        
        with steps_container:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-label">Step 1 — Research Plan</div>
                <div class="step-content">{result.get('research_plan', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="step-card">
                <div class="step-label">Step 2 — Gathered Information</div>
                <div class="step-content">{result.get('gathered_info', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="step-card">
                <div class="step-label">Step 3 — Final Report</div>
                <div class="step-content">{result.get('final_report', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Steps Completed", result.get("step_count", 3))
        col2.metric("Total Time", f"{elapsed:.1f}s")
        col3.metric("Model", model)


# ── Code Review Agent ──────────────────────────────────────────────────────
elif "💻 Code Review" in agent_type:
    st.markdown("### 💻 Code Review Agent")
    st.markdown("*LangGraph conditional routing: syntax check → branch by severity*")
    
    st.markdown("""
    **Agent Graph:**
    ```
    START → [Syntax Check] →─ PASS  → [Brief Review] → END
                            ├─ MINOR → [Suggestions]  → END
                            └─ MAJOR → [Fix Guide]    → END
    ```
    """)
    
    sample_codes = {
        "✅ Good Code": """def fibonacci(n: int) -> list[int]:
    \"\"\"Generate Fibonacci sequence.\"\"\"
    if n <= 0:
        return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]""",
        "⚠️ Minor Issues": """def calculate_avg(nums):
    total = 0
    for n in nums:
        total = total + n
    return total / len(nums)""",
        "❌ Syntax Error": """def broken_function(x
    return x * 2
    extra_line = True""",
    }
    
    code_template = st.selectbox("Load example:", list(sample_codes.keys()))
    code_input = st.text_area("Python Code:", value=sample_codes[code_template], height=200)
    
    if st.button("🔍 Review Code", use_container_width=True):
        agent = CodeReviewAgent(model=model)
        
        with st.spinner("Reviewing..."):
            result = agent.run(code_input)
        
        severity = result.get("severity", "unknown")
        colors = {"pass": "🟢", "minor": "🟡", "major": "🔴"}
        
        st.markdown(f"**Verdict:** {colors.get(severity, '⚪')} {severity.upper()}")
        
        # Syntax check result
        syntax_data = result.get("syntax_check", "{}")
        try:
            syntax_parsed = json.loads(syntax_data)
            st.markdown(f'<div class="tool-result">{json.dumps(syntax_parsed, indent=2)}</div>',
                       unsafe_allow_html=True)
        except:
            st.code(syntax_data)
        
        st.markdown("**Review Feedback:**")
        st.info(result.get("review_feedback", "No feedback"))
        
        col1, col2 = st.columns(2)
        col1.metric("Severity", severity.upper())
        col2.metric("Agent Steps", result.get("step_count", 0))


# ── Tool Demo ──────────────────────────────────────────────────────────────
elif "🔧 Tool Demo" in agent_type:
    st.markdown("### 🔧 Agent Tools Demo")
    st.markdown("*Standalone tools used by agents during task execution*")
    
    tool_choice = st.selectbox("Tool", ["Calculator", "Word Counter", "Syntax Checker"])
    
    if tool_choice == "Calculator":
        expr = st.text_input("Math expression:", "sqrt(144) + pow(3, 2)")
        if st.button("Calculate"):
            result = calculator.invoke(expr)
            st.markdown(f'<div class="tool-result">{result}</div>', unsafe_allow_html=True)
    
    elif tool_choice == "Word Counter":
        text = st.text_area("Text to analyze:", "The quick brown fox jumps over the lazy dog. AI on CPU is real!")
        if st.button("Count"):
            result = word_counter.invoke(text)
            st.json(json.loads(result))
    
    elif tool_choice == "Syntax Checker":
        code = st.text_area("Python code:", "def hello():\n    print('world')")
        if st.button("Check"):
            result = python_syntax_checker.invoke(code)
            st.json(json.loads(result))
