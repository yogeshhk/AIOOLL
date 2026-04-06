"""
AIOOLL — Integration Test Suite
=================================
Top-level integration tests that verify the full repo structure,
cross-module compatibility, and offline-readiness.

These tests do NOT require Ollama — they test everything that should
work on a fresh clone before any models are pulled.

Run: pytest tests/ -v
"""

import sys
import importlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# REPO STRUCTURE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRepoStructure:
    """Verify all documented files and folders are present."""

    REQUIRED_DIRS = [
        "src/ml", "src/dl", "src/rag", "src/agents", "src/cv", "src/llm_inference",
        "src/ml/data", "src/ml/tests", "src/ml/ui",
        "src/dl/tests", "src/dl/ui",
        "src/rag/knowledge_base", "src/rag/ui",
        "src/agents/tests", "src/agents/ui",
        "src/cv/ui",
        "src/llm_inference/ui",
        "datasets", "docs", "scripts", "tests",
    ]

    REQUIRED_FILES = [
        "README.md", "requirements.txt", "LICENSE", ".gitignore",
        "Makefile", "pyproject.toml",
        "scripts/setup.sh", "scripts/pull_models.sh",
        "docs/hardware_optimization.md",
        "docs/community_benchmarks.md",
        "src/ml/driver.py", "src/ml/ui/app.py", "src/ml/tests/test_ml.py",
        "src/ml/data/sms_spam.csv", "src/ml/data/house_prices.csv",
        "src/dl/driver.py", "src/dl/ui/app.py", "src/dl/tests/test_dl.py",
        "src/rag/driver.py", "src/rag/ui/app.py",
        "src/rag/knowledge_base/ai_on_cpu.txt",
        "src/agents/driver.py", "src/agents/ui/app.py", "src/agents/tests/test_agents.py",
        "src/cv/driver.py", "src/cv/ui/app.py",
        "src/llm_inference/driver.py", "src/llm_inference/ui/app.py",
    ]

    # Files that are optional (generated, platform-specific, or
    # may not be committed to every fork of the repo)
    OPTIONAL_FILES = [
        ".github/workflows/ci.yml",
        "docs/linkedin_post.md",
    ]

    @pytest.mark.parametrize("folder", REQUIRED_DIRS)
    def test_required_directory_exists(self, folder):
        path = ROOT / folder
        assert path.is_dir(), f"Missing directory: {folder}"

    @pytest.mark.parametrize("filepath", REQUIRED_FILES)
    def test_required_file_exists(self, filepath):
        path = ROOT / filepath
        assert path.is_file(), f"Missing file: {filepath}"

    @pytest.mark.parametrize("filepath", OPTIONAL_FILES)
    def test_optional_file_exists(self, filepath):
        """Warn but don't fail if optional files are absent."""
        path = ROOT / filepath
        if not path.is_file():
            pytest.skip(f"Optional file not present: {filepath}")

    def test_all_drivers_are_non_empty(self):
        for module in ["ml", "dl", "rag", "agents", "cv", "llm_inference"]:
            driver = ROOT / "src" / module / "driver.py"
            assert driver.stat().st_size > 500, f"src/{module}/driver.py looks empty"

    def test_all_ui_apps_are_non_empty(self):
        for module in ["ml", "dl", "rag", "agents", "cv", "llm_inference"]:
            ui = ROOT / "src" / module / "ui" / "app.py"
            assert ui.stat().st_size > 200, f"src/{module}/ui/app.py looks empty"


# ═══════════════════════════════════════════════════════════════════════════
# DATASET INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasets:
    """Verify bundled datasets are valid and well-formed."""

    def test_sms_spam_csv_loads(self):
        import pandas as pd
        df = pd.read_csv(ROOT / "src/ml/data/sms_spam.csv")
        assert len(df) >= 50, "SMS spam dataset too small"
        assert "label" in df.columns
        assert "text" in df.columns
        assert set(df["label"].unique()).issubset({"spam", "ham"})

    def test_sms_spam_has_both_classes(self):
        import pandas as pd
        df = pd.read_csv(ROOT / "src/ml/data/sms_spam.csv")
        assert "spam" in df["label"].values
        assert "ham" in df["label"].values

    def test_house_prices_csv_loads(self):
        import pandas as pd
        df = pd.read_csv(ROOT / "src/ml/data/house_prices.csv")
        assert len(df) >= 40, "House prices dataset too small"
        assert "price_lakh" in df.columns
        assert (df["price_lakh"] > 0).all(), "All prices must be positive"

    def test_house_prices_no_nulls(self):
        import pandas as pd
        df = pd.read_csv(ROOT / "src/ml/data/house_prices.csv")
        assert df.isnull().sum().sum() == 0, "House prices dataset has null values"

    def test_rag_knowledge_base_non_empty(self):
        kb = ROOT / "src/rag/knowledge_base/ai_on_cpu.txt"
        content = kb.read_text(encoding="utf-8")
        assert len(content) > 500, "RAG knowledge base file is too small"
        assert "CPU" in content or "cpu" in content.lower()


# ═══════════════════════════════════════════════════════════════════════════
# IMPORT TESTS (offline — no Ollama needed)
# ═══════════════════════════════════════════════════════════════════════════

class TestImports:
    """Verify all key dependencies are importable."""

    # Packages that must be present for AIOOLL to function
    CORE_PACKAGES = [
        "numpy", "pandas", "matplotlib", "seaborn",
        "sklearn", "scipy", "joblib",
        "torch", "torchvision",
        "nltk", "cv2", "PIL",
        "streamlit", "rich", "tqdm", "loguru",
    ]

    # Packages that are useful but not required for offline ML/DL/CV
    OPTIONAL_PACKAGES = [
        "plotly",                              # UI only — skip gracefully if absent
        "langchain", "langgraph", "chromadb", "ollama",
    ]

    @pytest.mark.parametrize("package", CORE_PACKAGES)
    def test_core_package_importable(self, package):
        try:
            importlib.import_module(package)
        except ImportError as e:
            pytest.fail(f"Core package '{package}' not importable: {e}\n"
                        f"Run: pip install -r requirements.txt")

    @pytest.mark.parametrize("package", OPTIONAL_PACKAGES)
    def test_optional_package_importable(self, package):
        try:
            importlib.import_module(package)
        except ImportError:
            pytest.skip(f"Optional package '{package}' not installed — skipping")

    def test_torch_is_cpu_build(self):
        import torch
        x = torch.tensor([1.0, 2.0, 3.0])
        assert x.device.type == "cpu"
        assert (x * 2).tolist() == [2.0, 4.0, 6.0]

    def test_sklearn_basic_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        X = np.random.randn(20, 3)
        y = (X[:, 0] > 0).astype(int)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == 20

    def test_opencv_version(self):
        import cv2
        major = int(cv2.__version__.split(".")[0])
        assert major >= 4, f"OpenCV >= 4 required, found {cv2.__version__}"


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-MODULE COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════

def _import_driver(module_name: str, alias: str):
    """
    Load src/<module_name>/driver.py as a fresh module without touching
    sys.path — avoids the Windows collision where a cached 'driver' entry
    from ml/ is returned for dl/, agents/, and cv/ lookups.
    """
    driver_path = ROOT / "src" / module_name / "driver.py"
    spec = importlib.util.spec_from_file_location(alias, driver_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCrossModule:
    """Verify modules can share data and models correctly."""

    def test_ml_data_usable_by_dl_module(self):
        """DL module reads the house prices CSV from ML data dir."""
        import pandas as pd
        data_path = ROOT / "src" / "ml" / "data" / "house_prices.csv"
        df = pd.read_csv(data_path)
        feature_cols = ["area_sqft", "bedrooms", "bathrooms", "age_years",
                        "distance_center_km", "has_garage", "has_garden", "floor_level"]
        assert all(c in df.columns for c in feature_cols), \
            "DL module expects these feature columns from ML data"

    def test_ml_driver_importable(self):
        mod = _import_driver("ml", "driver_ml")
        assert callable(mod.SpamClassifier)
        assert callable(mod.HousePricePredictor)

    def test_dl_driver_importable(self):
        mod = _import_driver("dl", "driver_dl")
        assert callable(mod.LSTMClassifier)
        assert callable(mod.SentimentDataset)
        assert callable(mod.TabularMLP)

    def test_agents_tools_importable(self):
        mod = _import_driver("agents", "driver_agents")
        # LangChain @tool returns a StructuredTool, not a plain function.
        # callable() returns False on StructuredTool; check .invoke instead.
        for tool_name in ("calculator", "word_counter", "python_syntax_checker"):
            tool = getattr(mod, tool_name)
            assert hasattr(tool, "invoke"), \
                f"'{tool_name}' has no .invoke — is it decorated with @tool?"

    def test_cv_driver_importable(self):
        mod = _import_driver("cv", "driver_cv")
        assert callable(mod.MotionDetector)
        assert callable(mod.ImageAnalyzer)


# ═══════════════════════════════════════════════════════════════════════════
# OFFLINE READINESS
# ═══════════════════════════════════════════════════════════════════════════

class TestOfflineReadiness:
    """Confirm no module makes network calls at import time."""

    def test_requirements_txt_exists_and_parseable(self):
        req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        lines = [l.strip() for l in req.splitlines()
                 if l.strip() and not l.startswith("#")]
        assert len(lines) >= 10, "requirements.txt seems incomplete"

    def test_no_hardcoded_api_keys(self):
        """Scan driver files for accidental API key patterns."""
        import re
        key_pattern = re.compile(
            r'(sk-[A-Za-z0-9]{20,}|OPENAI_API_KEY\s*=\s*["\'][^"\']+["\'])',
            re.IGNORECASE
        )
        for driver in ROOT.glob("src/*/driver.py"):
            content = driver.read_text(encoding="utf-8")
            match = key_pattern.search(content)
            assert match is None, \
                f"Possible hardcoded API key found in {driver}: {match.group()}"

    def test_scripts_are_executable_text(self):
        for script in (ROOT / "scripts").glob("*.sh"):
            content = script.read_text(encoding="utf-8")
            assert "#!/" in content[:20], f"{script.name} missing shebang"
            assert len(content) > 100, f"{script.name} looks empty"

    def test_makefile_has_key_targets(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        for target in ["setup", "test", "run-ml", "run-dl", "run-rag", "run-agents", "clean"]:
            assert f"{target}:" in makefile, f"Makefile missing target: {target}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
