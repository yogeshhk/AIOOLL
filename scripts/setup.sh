#!/usr/bin/env bash
# AIOOLL — Bootstrap Setup Script
# Works fully offline after initial package download
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

banner() {
  echo -e "${CYAN}"
  echo "  ╔═══════════════════════════════════════╗"
  echo "  ║      👀 AIOOLL — Setup Script         ║"
  echo "  ║   AI on CPU. Old laptop. No GPU.      ║"
  echo "  ╚═══════════════════════════════════════╝"
  echo -e "${NC}"
}

check_conda() {
  echo -e "${CYAN}[1/6] Checking Conda...${NC}"
  if ! command -v conda &>/dev/null; then
    echo -e "${RED}Conda not found. Install Miniconda from:${NC}"
    echo "    https://docs.conda.io/en/latest/miniconda.html"
    exit 1
  fi
  echo -e "${GREEN}✓ Conda found: $(conda --version)${NC}"
}

create_conda_env() {
  echo -e "${CYAN}[2/6] Creating conda environment 'aiooll'...${NC}"
  if conda env list | grep -q "^aiooll "; then
    echo -e "${YELLOW}⚠ Environment 'aiooll' already exists, skipping create${NC}"
  else
    conda create -n aiooll python=3.10 -y
    echo -e "${GREEN}✓ Conda env 'aiooll' created${NC}"
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate aiooll
  echo -e "${GREEN}✓ Conda env 'aiooll' activated${NC}"
}

install_deps() {
  echo -e "${CYAN}[3/6] Installing dependencies...${NC}"
  pip install --upgrade pip --quiet

  # CPU-only PyTorch (smaller download)
  pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu --quiet
  echo -e "${GREEN}✓ PyTorch (CPU) installed${NC}"

  pip install -r requirements.txt --quiet
  echo -e "${GREEN}✓ All dependencies installed${NC}"
}

download_nltk_data() {
  echo -e "${CYAN}[4/6] Downloading NLTK data...${NC}"
  python3 -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print('NLTK data ready')
"
  echo -e "${GREEN}✓ NLTK data downloaded${NC}"
}

download_cv_models() {
  echo -e "${CYAN}[5/6] Downloading OpenCV Haar Cascades...${NC}"
  python3 - <<'EOF'
import urllib.request, os
BASE = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
DEST = "src/cv/models"
os.makedirs(DEST, exist_ok=True)
files = ["haarcascade_frontalface_default.xml", "haarcascade_eye.xml"]
for f in files:
    out = os.path.join(DEST, f)
    if not os.path.exists(out):
        print(f"  Downloading {f}...")
        urllib.request.urlretrieve(BASE + f, out)
    else:
        print(f"  {f} already present")
print("CV models ready")
EOF
  echo -e "${GREEN}✓ CV models downloaded${NC}"
}

check_ollama() {
  echo -e "${CYAN}[6/6] Checking Ollama...${NC}"
  if command -v ollama &>/dev/null; then
    echo -e "${GREEN}✓ Ollama found: $(ollama --version)${NC}"
    echo -e "${YELLOW}  Run 'bash scripts/pull_models.sh' to download LLMs${NC}"
  else
    echo -e "${YELLOW}⚠ Ollama not found. Install with:${NC}"
    echo "    curl -fsSL https://ollama.ai/install.sh | sh"
    echo -e "${YELLOW}  Or run: bash scripts/install_ollama.sh${NC}"
  fi
}

main() {
  banner
  check_conda
  create_conda_env
  install_deps
  download_nltk_data
  download_cv_models
  check_ollama

  echo ""
  echo -e "${GREEN}════════════════════════════════════════${NC}"
  echo -e "${GREEN}  ✅ AIOOLL setup complete!             ${NC}"
  echo -e "${GREEN}════════════════════════════════════════${NC}"
  echo ""
  echo "Next steps:"
  echo "  conda activate aiooll"
  echo "  bash scripts/pull_models.sh"
  echo "  cd src/ml && python driver.py"
  echo "  cd src/ml && streamlit run ui/app.py"
  echo ""
}

main "$@"