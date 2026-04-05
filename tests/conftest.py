"""
AIOOLL — Shared pytest fixtures for integration tests.
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def sms_data():
    import pandas as pd
    return pd.read_csv(ROOT / "src/ml/data/sms_spam.csv")


@pytest.fixture(scope="session")
def house_data():
    import pandas as pd
    return pd.read_csv(ROOT / "src/ml/data/house_prices.csv")


def pytest_configure(config):
    config.addinivalue_line("markers", "ollama: requires Ollama to be running")
    config.addinivalue_line("markers", "slow: takes more than 30 seconds")
