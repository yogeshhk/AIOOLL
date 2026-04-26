# PrajnaAI (AIOOLL) — Claude Code Project Context

## Project Purpose
Offline, CPU-only AI toolkit for academic learning and demo on budget/old hardware (≤8 GB RAM, no GPU). Six independent modules each with a driver, tests, and Streamlit UI.

## Module Map
```
src/ml/             Classical ML (scikit-learn) — SpamClassifier, HousePricePredictor
src/dl/             Deep Learning (PyTorch CPU) — LSTMClassifier, TabularMLP
src/rag/            RAG pipeline (LangChain + ChromaDB + Ollama)
src/agents/         LangGraph agents — ResearchAgent, CodeReviewAgent
src/cv/             Classical CV (OpenCV) — HaarFaceDetector, MotionDetector, ImageAnalyzer
src/llm_inference/  Ollama benchmarking — OllamaClient, ModelBenchmarker, ChatSession
```

## Architecture Invariants
- Every module has: `driver.py` (runnable demo), `tests/` (pytest), `ui/app.py` (Streamlit)
- No module may require internet at runtime
- No CUDA — all torch code uses `device = "cpu"`
- Datasets bundled in `src/<module>/data/`; max 5 MB per file

## Environment
- Python 3.11, conda — use `environment.yml` to create env (`conda env create -f environment.yml`)
- Conda env name: `aiooll`
- CI: `.github/workflows/ci.yml` (runs ML + DL + agent tool tests; skips Ollama-dependent tests)

## Key Design Decisions
- `SpamClassifier.predict()` always returns confidence in [0, 1] (sigmoid of decision_function for SVM/LR; raw predict_proba for tree models)
- `LSTMClassifier` uses `num_layers=2` so the `dropout` parameter takes effect between layers
- ChromaDB ≥0.4 auto-persists — do not call `.persist()` explicitly
- `TabularMLP` uses BatchNorm1d: always call `model.eval()` for single-sample inference

## Test Strategy
- ML/DL/CV/agent tool tests: fully offline, run in CI
- RAG/LLM tests: require Ollama — mark with `@pytest.mark.ollama`, excluded from CI
- Performance thresholds in tests are real bars (AUC > 0.90, R² > 0.7), not placeholders

## Common Tasks
```bash
make test           # run all offline tests
make run-ml         # headless driver demo
make ui-ml          # Streamlit UI
pytest src/ml/tests/ -v
```
