# PrajnaAI Makefile — convenience commands
# Usage: make <target>

.PHONY: help setup test lint run-ml run-dl run-rag run-agents run-cv run-llm clean

PYTHON = python3
PYTEST = pytest
STREAMLIT = streamlit run

help:
	@echo "🕉️  PrajnaAI — Available Commands"
	@echo "=================================="
	@echo "  make setup      — Install all dependencies"
	@echo "  make models     — Pull Ollama models"
	@echo "  make test       — Run all tests"
	@echo "  make lint       — Lint Python code"
	@echo ""
	@echo "  make run-ml     — Run ML module driver"
	@echo "  make run-dl     — Run DL module driver"
	@echo "  make run-rag    — Run RAG module driver"
	@echo "  make run-agents — Run Agents module driver"
	@echo "  make run-cv     — Run CV module driver"
	@echo "  make run-llm    — Run LLM inference driver"
	@echo ""
	@echo "  make ui-ml      — Launch ML Streamlit UI"
	@echo "  make ui-dl      — Launch DL Streamlit UI"
	@echo "  make ui-rag     — Launch RAG Streamlit UI"
	@echo "  make ui-agents  — Launch Agents Streamlit UI"
	@echo "  make ui-cv      — Launch CV Streamlit UI"
	@echo "  make ui-llm     — Launch LLM Streamlit UI"
	@echo ""
	@echo "  make clean      — Remove generated artifacts"

setup:
	bash scripts/setup.sh

models:
	bash scripts/pull_models.sh

test:
	$(PYTEST) src/ml/tests/ src/dl/tests/ src/agents/tests/ -v --timeout=120

test-ml:
	$(PYTEST) src/ml/tests/ -v --timeout=120

test-dl:
	$(PYTEST) src/dl/tests/ -v --timeout=180

test-agents:
	$(PYTEST) src/agents/tests/ -v --timeout=60

lint:
	flake8 src/ --max-line-length=100 --ignore=E203,W503 --exclude=__pycache__

run-ml:
	cd src/ml && $(PYTHON) driver.py

run-dl:
	cd src/dl && $(PYTHON) driver.py

run-rag:
	cd src/rag && $(PYTHON) driver.py

run-agents:
	cd src/agents && $(PYTHON) driver.py

run-cv:
	cd src/cv && $(PYTHON) driver.py

run-llm:
	cd src/llm_inference && $(PYTHON) driver.py

ui-ml:
	$(STREAMLIT) src/ml/ui/app.py

ui-dl:
	$(STREAMLIT) src/dl/ui/app.py

ui-rag:
	$(STREAMLIT) src/rag/ui/app.py

ui-agents:
	$(STREAMLIT) src/agents/ui/app.py

ui-cv:
	$(STREAMLIT) src/cv/ui/app.py

ui-llm:
	$(STREAMLIT) src/llm_inference/ui/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.log" -delete
	rm -rf src/*/models/*.pkl src/*/models/*.pt
	rm -rf src/*/results/*.png src/*/results/*.json
	rm -rf src/rag/.chroma_db
	@echo "✓ Cleaned"
