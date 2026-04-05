"""
PrajnaAI — Agent Tests (no Ollama required for tool tests)
"""

import sys
import json
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from driver import calculator, word_counter, python_syntax_checker, text_summarizer_tool


class TestCalculatorTool:

    def test_basic_arithmetic(self):
        result = calculator.invoke("2 + 3 * 4")
        assert "14" in result

    def test_square_root(self):
        result = calculator.invoke("sqrt(144)")
        assert "12" in result

    def test_power(self):
        result = calculator.invoke("pow(2, 10)")
        assert "1024" in result

    def test_trig(self):
        result = calculator.invoke("round(sin(pi/2), 4)")
        assert "1.0" in result

    def test_invalid_expression(self):
        result = calculator.invoke("import os")
        assert "Error" in result

    def test_division_by_zero(self):
        result = calculator.invoke("1/0")
        assert "Error" in result or "inf" in result.lower()


class TestWordCounterTool:

    def test_basic_count(self):
        result = word_counter.invoke("Hello world how are you")
        data = json.loads(result)
        assert data["words"] == 5

    def test_sentence_count(self):
        result = word_counter.invoke("Hello. World! How are you?")
        data = json.loads(result)
        assert data["sentences"] == 3

    def test_character_count(self):
        result = word_counter.invoke("abc")
        data = json.loads(result)
        assert data["characters"] == 3

    def test_returns_valid_json(self):
        result = word_counter.invoke("Any text here for testing purposes")
        data = json.loads(result)
        assert "words" in data
        assert "sentences" in data
        assert "characters" in data
        assert "avg_word_length" in data


class TestSyntaxChecker:

    def test_valid_code(self):
        code = "def hello():\n    print('world')"
        result = python_syntax_checker.invoke(code)
        data = json.loads(result)
        assert data["status"] == "valid"

    def test_invalid_code(self):
        code = "def broken(\n    print 'no colon'"
        result = python_syntax_checker.invoke(code)
        data = json.loads(result)
        assert data["status"] == "syntax_error"
        assert "line" in data

    def test_counts_functions(self):
        code = "def a():\n    pass\ndef b():\n    pass"
        result = python_syntax_checker.invoke(code)
        data = json.loads(result)
        assert data["functions"] == 2

    def test_counts_classes(self):
        code = "class MyClass:\n    pass"
        result = python_syntax_checker.invoke(code)
        data = json.loads(result)
        assert data["classes"] == 1

    def test_counts_imports(self):
        code = "import os\nimport sys\nfrom pathlib import Path"
        result = python_syntax_checker.invoke(code)
        data = json.loads(result)
        assert data["imports"] == 3


class TestSummarizerTool:

    def test_short_text_passthrough(self):
        text = "AI is great. CPU is enough."
        result = text_summarizer_tool.invoke(text)
        assert len(result) > 0

    def test_long_text_summarized(self):
        text = ". ".join([f"Sentence number {i} about important things" for i in range(20)])
        result = text_summarizer_tool.invoke(text)
        # Result should be shorter than input
        assert len(result) < len(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
