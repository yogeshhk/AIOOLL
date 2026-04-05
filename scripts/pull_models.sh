#!/usr/bin/env bash
# PrajnaAI — Pull quantized CPU-friendly models via Ollama
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${CYAN}Pulling CPU-optimized LLMs via Ollama...${NC}"

if ! command -v ollama &>/dev/null; then
  echo "Ollama not installed. Run: bash scripts/install_ollama.sh"
  exit 1
fi

# Ensure Ollama is running
ollama serve &>/dev/null &
sleep 2

MODELS=(
  "gemma2:2b"          # Best quality small model ~1.6GB
  "qwen2:1.5b"         # Fast multilingual ~0.9GB
  "nomic-embed-text"   # Embeddings for RAG ~270MB
)

for model in "${MODELS[@]}"; do
  echo -e "${CYAN}Pulling $model...${NC}"
  ollama pull "$model"
  echo -e "${GREEN}✓ $model ready${NC}"
done

echo ""
echo -e "${GREEN}All models pulled. Run 'ollama list' to verify.${NC}"
