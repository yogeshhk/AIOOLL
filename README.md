# 👀 AIOOLL
### *AI On Old Linux Laptop*

> **Full-stack AI workflows running entirely on a CPU-only Linux laptop.**
> No GPU. No cloud. No excuses.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-green.svg)](https://ollama.ai)
[![Offline](https://img.shields.io/badge/Internet-Not%20Required-red.svg)](#)
[![Stars](https://img.shields.io/github/stars/yourusername/AIOOLL?style=social)](#)

---

## 👀 What is AIOOLL?

**AIOOLL** = **A**I **O**n **O**ld **L**inux **L**aptop

That dusty i3 machine under your desk? It's a complete AI development environment.
This repository proves it, with working code, tests, Streamlit UIs, and real benchmark numbers.

The double-O in AIOOLL isn't just letters, it's a pair of eyes, wide open at what old hardware can do.

---

## 🗂️ Repository Structure

```
AIOOLL/
├── src/
│   ├── ml/              # Classical Machine Learning (scikit-learn)
│   ├── dl/              # Deep Learning (PyTorch, CPU optimized)
│   ├── rag/             # Retrieval-Augmented Generation (LangChain)
│   ├── agents/          # AI Agents & Workflows (LangGraph)
│   ├── cv/              # Computer Vision Lite (OpenCV)
│   └── llm_inference/   # Local LLM Inference (Ollama / llama.cpp)
├── datasets/            # Bundled small datasets (all offline)
├── docs/                # Architecture diagrams, guides
├── scripts/             # Setup, model download, env bootstrap
└── tests/               # Integration tests
    ├── conftest.py          # Shared fixtures
    └── test_integration.py  # Structure, imports, offline-readiness checks
```

---

## 🚀 Quick Start

### 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU       | Intel i3 (dual-core) | Intel i5/i7 (quad-core) |
| RAM       | 8 GB    | 16 GB |
| Storage   | 20 GB free (SSD!) | 50 GB SSD |
| OS        | Xubuntu 22.04 LTS | Lubuntu / Xubuntu |
| Python    | 3.10    | 3.11+ |
| Conda     | Miniconda | Miniconda / Mambaforge |

### 2. One-Line Bootstrap

```bash
git clone https://github.com/yogeshhk/AIOOLL.git
cd AIOOLL
conda create -n aiooll python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aiooll
pip install -r requirements.txt
bash scripts/setup.sh
```

### 3. Install Ollama & Pull Models

```bash
# Install Ollama
bash scripts/install_ollama.sh

# Pull quantized CPU-friendly models
ollama pull gemma2:2b
ollama pull qwen2:1.5b
ollama pull nomic-embed-text    # for RAG embeddings
```

### 4. Run Any Module

```bash
conda activate aiooll

# Driver (headless)
cd src/ml && python driver.py

# Interactive Streamlit UI
streamlit run src/ml/ui/app.py

# Or use make
make run-ml
make ui-rag
```

---

## 🧩 Modules at a Glance

### 🔬 1. Classical Machine Learning (`src/ml/`)
Academic-grade scikit-learn implementations.
- **Spam/Sentiment Classifier**: Naive Bayes, Logistic Regression, SVM with TF-IDF
- **Predictive Modeling**: House price regression, cross-validation, SHAP explainability
- **Streamlit UI**: Interactive model comparison dashboard

### 🧠 2. Deep Learning (`src/dl/`)
PyTorch models optimized for CPU inference.
- **Text Classification**: Bidirectional LSTM trained from scratch
- **Tabular Learning**: MLP with batch normalization and Huber loss
- **Streamlit UI**: Training visualizer with live loss curves

### 📚 3. RAG Pipeline (`src/rag/`)
LangChain + local embeddings + Ollama LLMs.
- **Document Q&A**: PDF/TXT ingestion with ChromaDB vector store
- **Semantic Search**: nomic-embed-text embeddings (CPU)
- **Streamlit UI**: Chat interface with source highlighting

### 🤖 4. AI Agents (`src/agents/`)
LangGraph multi-agent workflows: no OpenAI required.
- **Research Agent**: Plan → Search local KB → Synthesize (linear graph)
- **Code Review Agent**: Conditional routing by severity (branching graph)
- **Streamlit UI**: Agent thought-process visualizer

### 👁️ 5. Computer Vision (`src/cv/`)
OpenCV classical CV: zero deep learning required.
- **Face Detection**: Haar Cascade real-time detection
- **Motion Detection**: MOG2 background subtraction
- **Streamlit UI**: Live webcam feed with annotations

### 🦙 6. Local LLM Inference (`src/llm_inference/`)
Direct Ollama integration and benchmarking.
- **Model Benchmarking**: Tokens/sec, latency leaderboard across models
- **Chat Interface**: Multi-turn conversation with history
- **Prompt Patterns**: Zero-shot, few-shot, chain-of-thought, structured output
- **Streamlit UI**: Model comparison dashboard

---

## ⚙️ Hardware Optimization

### Quick Wins (no hardware needed)

```bash
# Set CPU to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check free RAM before inference
free -h

# Monitor during inference
btop
```

| Tweak | Impact |
|-------|--------|
| Xubuntu instead of Ubuntu | Saves ~400MB RAM |
| CPU performance governor | 15–40% faster inference |
| GGUF Q4_K_M quantization | 4x smaller than FP16 |
| SSD instead of HDD | 13x faster model load |
| Close browser tabs | Frees 300–800MB RAM |

---

## 🦙 Supported Local Models

| Model | Size (GGUF Q4) | RAM Needed | Best For |
|-------|----------------|------------|----------|
| Gemma 2 2B | ~1.5 GB | 4 GB | Fast chat, RAG |
| Qwen 2 1.5B | ~1.0 GB | 4 GB | Multilingual |
| TinyLlama 1.1B | ~0.7 GB | 3 GB | Agents, tools |
| Phi-3 Mini 3.8B | ~2.4 GB | 6 GB | Reasoning |
| nomic-embed-text | ~270 MB | 2 GB | Embeddings only |

---

## 📊 Benchmark Results (Intel i3-8130U, 8GB RAM, SSD)

| Task | Tool/Model | Result |
|------|-----------|--------|
| Chat inference | Gemma2:2b | ~8 tok/s |
| RAG query | nomic-embed-text | ~50ms |
| ML training | Logistic Regression | <1s |
| DL training | PyTorch MLP, 100 epochs | ~30s |
| CV face detection | Haar Cascade | ~15 FPS |

Share your own numbers → [docs/community_benchmarks.md](docs/community_benchmarks.md)

---

## 🛠️ Tech Stack

```
LLM Runtime:    Ollama (llama.cpp backend)
ML:             scikit-learn, pandas, numpy, SHAP
DL:             PyTorch (CPU build)
RAG:            LangChain, ChromaDB, sentence-transformers
Agents:         LangGraph
CV:             OpenCV, Pillow
UI:             Streamlit + Plotly
Testing:        pytest
```

---

## 🤝 Contributing

All contributions must respect the **offline-first, CPU-only** constraint.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the full checklist.

---

## 📜 License

MIT: see [LICENSE](LICENSE)

---

## 👀 Star AIOOLL

If AIOOLL helps you build AI without the cloud tax, ⭐ star this repo.
Every star tells the community: **you don't need a GPU to do real AI work.**

*Those two O's are watching. Don't let them down.*
