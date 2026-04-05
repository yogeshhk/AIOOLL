"""
PrajnaAI — Local LLM Inference Benchmarking
============================================
Direct Ollama integration for:
  1. Model benchmarking (tokens/sec, latency, memory)
  2. Multi-turn chat with history
  3. Prompt engineering patterns
  4. Model comparison leaderboard

Zero internet. Zero cloud. All local.
"""

import sys
import time
import json
import httpx
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Generator

from loguru import logger

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OLLAMA_BASE = "http://localhost:11434"


# ═══════════════════════════════════════════════════════════════════════════
# OLLAMA CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class OllamaClient:
    """Thin wrapper around Ollama REST API."""

    def __init__(self, base_url: str = OLLAMA_BASE):
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)

    def is_running(self) -> bool:
        try:
            resp = self.client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[dict]:
        resp = self.client.get(f"{self.base_url}/api/tags")
        return resp.json().get("models", [])

    def generate(self, model: str, prompt: str, system: str = "",
                 temperature: float = 0.7, max_tokens: int = 512) -> dict:
        """Generate with timing metadata."""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        t0 = time.time()
        resp = self.client.post(f"{self.base_url}/api/generate", json=payload)
        elapsed = time.time() - t0
        data = resp.json()
        return {
            "response": data.get("response", ""),
            "total_duration_s": elapsed,
            "eval_count": data.get("eval_count", 0),
            "tokens_per_second": data.get("eval_count", 0) / max(elapsed, 0.001),
            "model": model,
        }

    def chat(self, model: str, messages: list[dict], temperature: float = 0.7) -> dict:
        """Multi-turn chat."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 512}
        }
        t0 = time.time()
        resp = self.client.post(f"{self.base_url}/api/chat", json=payload)
        elapsed = time.time() - t0
        data = resp.json()
        return {
            "message": data.get("message", {}).get("content", ""),
            "latency_s": elapsed,
            "model": model,
        }

    def embed(self, model: str, text: str) -> list[float]:
        """Generate embeddings."""
        resp = self.client.post(f"{self.base_url}/api/embeddings",
                                json={"model": model, "prompt": text})
        return resp.json().get("embedding", [])


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARKER
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK_PROMPTS = [
    "Explain the concept of attention mechanism in transformers in 3 sentences.",
    "Write a Python function to check if a number is prime.",
    "What is the difference between RAM and ROM?",
    "Summarize the main advantages of quantized neural networks.",
    "List 5 Linux commands for system monitoring.",
]


@dataclass
class BenchmarkResult:
    model: str
    prompt_idx: int
    tokens_per_second: float
    latency_s: float
    response_length: int
    quality_chars: int


class ModelBenchmarker:
    """Compare multiple Ollama models on speed and quality."""

    def __init__(self, models: list[str] = None):
        self.client = OllamaClient()
        self.available_models = [m["name"] for m in self.client.list_models()]
        self.models = models or [m for m in self.available_models
                                  if any(x in m for x in ["gemma", "qwen", "llama", "phi", "tiny"])]
        self.results: list[BenchmarkResult] = []

    def run_benchmark(self, num_prompts: int = 3) -> list[BenchmarkResult]:
        logger.info(f"\n🏎️  Benchmarking {len(self.models)} models on {num_prompts} prompts")
        logger.info(f"Models: {self.models}")

        for model in self.models:
            logger.info(f"\n📊 Model: {model}")
            model_times = []
            for i, prompt in enumerate(BENCHMARK_PROMPTS[:num_prompts]):
                logger.info(f"  Prompt {i+1}/{num_prompts}: {prompt[:50]}...")
                try:
                    result = self.client.generate(model, prompt, max_tokens=256)
                    bench = BenchmarkResult(
                        model=model,
                        prompt_idx=i,
                        tokens_per_second=result["tokens_per_second"],
                        latency_s=result["total_duration_s"],
                        response_length=len(result["response"].split()),
                        quality_chars=len(result["response"])
                    )
                    self.results.append(bench)
                    model_times.append(result["tokens_per_second"])
                    logger.info(f"    → {result['tokens_per_second']:.1f} tok/s | {result['total_duration_s']:.1f}s")
                except Exception as e:
                    logger.error(f"    Error: {e}")

            if model_times:
                logger.info(f"  Average: {sum(model_times)/len(model_times):.1f} tok/s")

        self._save_results()
        return self.results

    def _save_results(self):
        data = [{"model": r.model, "tokens_per_second": r.tokens_per_second,
                 "latency_s": r.latency_s, "response_words": r.response_length}
                for r in self.results]
        with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved → {RESULTS_DIR / 'benchmark_results.json'}")

    def leaderboard(self) -> dict:
        """Aggregate results by model."""
        by_model = {}
        for r in self.results:
            if r.model not in by_model:
                by_model[r.model] = {"tps": [], "latency": []}
            by_model[r.model]["tps"].append(r.tokens_per_second)
            by_model[r.model]["latency"].append(r.latency_s)

        return {
            model: {
                "avg_tps": sum(v["tps"]) / len(v["tps"]),
                "avg_latency_s": sum(v["latency"]) / len(v["latency"]),
            }
            for model, v in by_model.items()
        }


# ═══════════════════════════════════════════════════════════════════════════
# CHAT SESSION
# ═══════════════════════════════════════════════════════════════════════════

class ChatSession:
    """Multi-turn chat with conversation history."""

    SYSTEM_PROMPT = """You are a helpful AI assistant running locally on a CPU-only Linux laptop.
You are part of PrajnaAI — a suite of offline AI tools.
Be concise, accurate, and helpful. You have no internet access."""

    def __init__(self, model: str = "gemma2:2b"):
        self.model = model
        self.client = OllamaClient()
        self.history: list[dict] = []

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self.history

        result = self.client.chat(self.model, messages)
        assistant_message = result["message"]
        self.history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def reset(self):
        self.history = []

    def get_history(self) -> list[dict]:
        return self.history.copy()


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT ENGINEERING PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

class PromptEngineer:
    """Demonstrate prompt engineering patterns with local LLMs."""

    def __init__(self, model: str = "gemma2:2b"):
        self.client = OllamaClient()
        self.model = model

    def zero_shot(self, task: str) -> str:
        return self.client.generate(self.model, task)["response"]

    def few_shot(self, examples: list[tuple[str, str]], query: str) -> str:
        prompt = "Examples:\n"
        for inp, out in examples:
            prompt += f"Input: {inp}\nOutput: {out}\n\n"
        prompt += f"Input: {query}\nOutput:"
        return self.client.generate(self.model, prompt)["response"]

    def chain_of_thought(self, problem: str) -> str:
        prompt = f"""Solve this step by step:

Problem: {problem}

Let me think through this step by step:
Step 1:"""
        return self.client.generate(self.model, prompt)["response"]

    def structured_output(self, task: str, schema: str) -> str:
        prompt = f"""{task}

Respond ONLY with valid JSON following this schema:
{schema}

JSON response:"""
        return self.client.generate(self.model, prompt, temperature=0.1)["response"]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("🕉️  PrajnaAI — Local LLM Inference Module")
    logger.info("="*60)

    client = OllamaClient()
    if not client.is_running():
        logger.error("Ollama not running. Start with: ollama serve")
        sys.exit(1)

    models = client.list_models()
    logger.info(f"Available models: {[m['name'] for m in models]}")

    # 1. Benchmark
    benchmarker = ModelBenchmarker()
    if benchmarker.models:
        results = benchmarker.run_benchmark(num_prompts=2)
        board = benchmarker.leaderboard()
        logger.info("\n🏆 Model Leaderboard (by tokens/sec):")
        for model, stats in sorted(board.items(), key=lambda x: -x[1]["avg_tps"]):
            logger.info(f"  {model:30s} | {stats['avg_tps']:5.1f} tok/s | {stats['avg_latency_s']:.1f}s avg")
    else:
        logger.warning("No chat models found. Pull models first: bash scripts/pull_models.sh")

    # 2. Prompt engineering demo
    if benchmarker.models:
        model = benchmarker.models[0]
        engineer = PromptEngineer(model=model)

        logger.info(f"\n🎯 Prompt Engineering Demos (model: {model})")
        cot_result = engineer.chain_of_thought(
            "If a CPU processes 8 tokens per second and I need 500 tokens, how long will it take?"
        )
        logger.info(f"Chain-of-Thought result:\n{cot_result[:300]}...")

    logger.info("\n✅ LLM Inference module complete!")
    logger.info("🎨 Launch UI: streamlit run src/llm_inference/ui/app.py")


if __name__ == "__main__":
    main()
