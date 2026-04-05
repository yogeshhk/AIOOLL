# 🕉️ PrajnaAI — Artificial Intelligence on Ancient Iron

> *Prajna (प्रज्ञा) — Sanskrit for "transcendent wisdom"*
> **Full-stack AI workflows running entirely on a CPU-only Linux laptop. No GPU. No cloud. No excuses.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-green.svg)](https://ollama.ai)
[![Offline](https://img.shields.io/badge/Internet-Not%20Required-red.svg)](#)
[![Stars](https://img.shields.io/github/stars/yourusername/PrajnaAI?style=social)](#)

---

## 🧘 Philosophy

Modern AI culture worships GPUs and cloud credits.  
**PrajnaAI disagrees.**

This repository demonstrates that a dusty i3 laptop with 8GB RAM running Xubuntu can serve as a complete, self-contained AI research and development environment — covering everything from classical ML to RAG pipelines to multi-agent systems, all powered by quantized local LLMs via Ollama.

If wisdom (*prajna*) is the goal, you don't need a $10,000 GPU cluster. You need the right tools, the right mindset, and maybe an SSD upgrade.

---

## 🗂️ Repository Structure

```
PrajnaAI/
├── src/
│   ├── ml/              # Classical Machine Learning (scikit-learn)
│   ├── dl/              # Deep Learning (PyTorch — CPU optimized)
│   ├── rag/             # Retrieval-Augmented Generation (LangChain)
│   ├── agents/          # AI Agents & Workflows (LangGraph)
│   ├── cv/              # Computer Vision Lite (OpenCV)
│   └── llm_inference/   # Local LLM Inference (Ollama / llama.cpp)
├── datasets/            # Bundled small datasets (all offline)
├── docs/                # Architecture diagrams, guides
├── scripts/             # Setup, model download, env bootstrap
└── tests/               # Integration tests
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

### 2. One-Line Bootstrap

```bash
git clone https://github.com/yourusername/PrajnaAI.git
cd PrajnaAI
bash scripts/setup.sh
```

### 3. Install Ollama & Pull Models

```bash
# Install Ollama (offline installer bundled in scripts/)
bash scripts/install_ollama.sh

# Pull quantized CPU-friendly models
ollama pull gemma2:2b
ollama pull qwen2:1.5b
ollama pull nomic-embed-text    # for RAG embeddings
```

### 4. Activate Environment & Run a Module

```bash
conda activate prajna
# or
source venv/bin/activate

# Try the ML module
cd src/ml && python driver.py
streamlit run ui/app.py
```

---

## 🧩 Modules at a Glance

### 🔬 1. Classical Machine Learning (`src/ml/`)
Academic-grade scikit-learn implementations.
- **Spam/Sentiment Classifier** — Naive Bayes, Logistic Regression, SVM with TF-IDF
- **Predictive Modeling** — House price & student performance regression
- **Pipeline Architecture** — Feature engineering, cross-validation, SHAP explainability
- **Streamlit UI** — Interactive model comparison dashboard

### 🧠 2. Deep Learning (`src/dl/`)
PyTorch models optimized for CPU inference.
- **Text Classification** — LSTM / GRU networks trained from scratch
- **Tabular Learning** — MLP with batch normalization
- **Time Series** — 1D CNN for sensor / stock data
- **Streamlit UI** — Training visualizer with live loss curves

### 📚 3. RAG Pipeline (`src/rag/`)
LangChain + local embeddings + Ollama LLMs.
- **Document Q&A** — PDF/TXT ingestion with ChromaDB vector store
- **Semantic Search** — nomic-embed-text embeddings (CPU)
- **Streaming Answers** — Real-time response generation
- **Streamlit UI** — Chat interface with source highlighting

### 🤖 4. AI Agents (`src/agents/`)
LangGraph multi-agent workflows — no OpenAI required.
- **Research Agent** — Plan → Search local KB → Synthesize
- **Code Review Agent** — Static analysis + LLM feedback loop
- **Tool-Use Agent** — Calculator, file reader, web scraper (offline)
- **Streamlit UI** — Agent thought-process visualizer

### 👁️ 5. Computer Vision (`src/cv/`)
OpenCV classical CV — zero deep learning required.
- **Face Detection** — Haar Cascade real-time detection
- **Motion Detection** — Background subtraction for security use
- **Image Pipeline** — Preprocessing, augmentation, analysis
- **Streamlit UI** — Live webcam feed with annotations

### 🦙 6. Local LLM Inference (`src/llm_inference/`)
Direct Ollama integration and benchmarking.
- **Model Benchmarking** — Tokens/sec, memory, latency across models
- **Chat Interface** — Multi-turn conversation with history
- **Prompt Engineering** — Few-shot, chain-of-thought templates
- **Streamlit UI** — Model comparison leaderboard

---

## ⚙️ Hardware Optimization Guide

### Make Your Old Laptop Fly

```bash
# Check CPU governor (set to performance)
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check available RAM
free -h

# Monitor during inference
htop  # or btop for prettier output
```

### Recommended System Tweaks

| Tweak | Command / Action |
|-------|-----------------|
| Use Xubuntu/Lubuntu | Saves ~400MB RAM vs GNOME |
| Disable swap (if SSD) | `sudo swapoff -a` during inference |
| Set CPU performance mode | See above |
| Use GGUF Q4_K_M quantization | Best quality/speed tradeoff on CPU |
| Close browser tabs | Browsers eat RAM voraciously |

---

## 🦙 Supported Local Models

| Model | Size (GGUF Q4) | RAM Needed | Best For |
|-------|----------------|------------|----------|
| Gemma 2 2B | ~1.5 GB | 4 GB | Fast chat, RAG |
| Qwen 2 1.5B | ~1.0 GB | 4 GB | Multilingual |
| TinyLlama 1.1B | ~0.7 GB | 3 GB | Agents, tools |
| Phi-3 Mini 3.8B | ~2.4 GB | 6 GB | Reasoning |
| nomic-embed-text | ~270 MB | 2 GB | Embeddings |

---

## 📊 Benchmark Results (Intel i3-8130U, 8GB RAM, SSD)

| Task | Model | Speed |
|------|-------|-------|
| Chat inference | Gemma2:2b | ~8 tok/s |
| RAG retrieval | nomic-embed | ~50ms/query |
| ML training | Logistic Reg | <1s |
| DL training (small) | PyTorch MLP | ~30s/epoch |
| CV face detection | Haar Cascade | 15 FPS |

---

## 🛠️ Tech Stack

```
Language:     Python 3.11
LLM Runtime:  Ollama (llama.cpp backend)
ML:           scikit-learn, pandas, numpy
DL:           PyTorch (CPU build), torchvision
RAG:          LangChain, ChromaDB, sentence-transformers
Agents:       LangGraph, LangChain tools
CV:           OpenCV, Pillow
UI:           Streamlit
Explainability: SHAP, matplotlib, seaborn
```

---

## 🤝 Contributing

Contributions that respect the **offline-first, CPU-only** philosophy are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-module`
3. Ensure everything works **without internet**
4. Add tests in the module's `tests/` directory
5. Submit a PR with benchmark numbers

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## 🌟 Star This Repository

If PrajnaAI helps you build AI without the cloud tax, please ⭐ star this repo.  
Every star tells the community: **You don't need a GPU to do real AI work.**

---

*"The quieter you become, the more you can hear." — RAM Dass (and also RAM usage)*
