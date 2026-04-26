# Contributing to PrajnaAI 🕉️

First off — thank you for considering a contribution! PrajnaAI is built on a very specific philosophy, and contributions that respect it are warmly welcomed.

## The Prime Directive

> **Everything must work offline, on a CPU-only machine, with ≤ 8GB RAM.**

If your contribution requires an internet connection at runtime, a GPU, or more than 8GB RAM — it doesn't belong here. This constraint is a feature, not a limitation.

## Ways to Contribute

### 🐛 Bug Reports
Open an issue with:
- Your hardware (CPU model, RAM amount)
- OS and Python version
- Exact error message and stack trace
- Steps to reproduce

### 🚀 New Modules / Features
We welcome new additions in these areas:
- New ML algorithms with academic-level implementation
- Additional PyTorch architectures (CPU-optimized)
- New RAG retrieval strategies
- Additional LangGraph agent patterns
- New Ollama model benchmarks
- Computer vision algorithms (OpenCV-based)

### 📊 Benchmark Results
Share benchmark numbers from your hardware! We want to build a community benchmark table. Open a PR adding your results to `docs/community_benchmarks.md`.

## Contribution Guidelines

### Code Standards
- Python 3.10+ with type hints
- Docstrings for all classes and public methods
- Follow the existing driver/tests/ui pattern for each module
- No hardcoded paths — use `Path(__file__).parent` patterns

### Testing
- All new code must have tests in the module's `tests/` directory
- Tests must pass without internet: `pytest src/module/tests/ --timeout=60`
- Include at least: one happy path, one edge case, one error case

### Datasets
- All datasets must be bundled in the repo (no downloads at runtime)
- Maximum dataset size: 5MB per file, 20MB per module
- Synthetic or openly licensed data only (CC0, MIT, Apache 2.0)
- Include a `README.md` in the data directory with source attribution

### Pull Request Checklist
- [ ] Works offline (no runtime internet)
- [ ] Works on CPU-only (no CUDA calls)
- [ ] Tests pass: `pytest src/`
- [ ] Code linted: `flake8 src/` (max line length 100)
- [ ] New dependencies added to `requirements.txt` and `environment.yml`
- [ ] Module has a `driver.py`, `tests/`, and `ui/app.py`

## Getting Started

```bash
git clone https://github.com/yourusername/PrajnaAI.git
cd PrajnaAI
conda env create -f environment.yml
conda activate aiooll
bash scripts/setup.sh

# Run all tests
pytest src/ -v --timeout=120

# Run a specific module
cd src/ml && python driver.py
```

## Code of Conduct

Be respectful. Be constructive. Help others succeed with their old hardware.

The AI community often gatekeeps based on hardware. PrajnaAI exists to push back against that. Every contributor here believes that **wisdom doesn't need a GPU**.

---

*प्रज्ञा — Prajna: May your commits be as clear as your understanding.*
