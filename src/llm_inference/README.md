# LLM Inference Module — Local Model Benchmarking

## Overview

Direct Ollama REST API integration for benchmarking, multi-turn chat, and prompt engineering pattern demos. No LangChain abstraction — raw HTTP calls for maximum transparency.

## Features

| Class | Purpose |
|-------|---------|
| `OllamaClient` | Thin wrapper: `generate`, `chat`, `embed` |
| `ModelBenchmarker` | Tokens/sec + latency leaderboard across models |
| `ChatSession` | Multi-turn conversation with history management |
| `PromptEngineer` | Demonstrates zero-shot, few-shot, chain-of-thought, structured output |

## Files

```
llm_inference/
├── driver.py           # Main entry point — benchmark + prompt demos
├── ui/app.py           # Streamlit model comparison + chat UI
└── results/
    └── benchmark_results.json  # Saved benchmark output (auto-generated)
```

## Setup

```bash
ollama serve &
ollama pull gemma2:2b        # or any model in the table below
python driver.py
```

## Recommended Models

| Model | RAM (Q4_K_M) | Speed (i3) | Best use |
|-------|-------------|-----------|---------|
| `gemma2:2b` | ~1.5 GB | ~8 tok/s | General chat, RAG |
| `qwen2:1.5b` | ~1.0 GB | ~12 tok/s | Multilingual |
| `tinyllama:1.1b` | ~0.7 GB | ~15 tok/s | Lightweight agents |
| `phi3:mini` | ~2.4 GB | ~5 tok/s | Reasoning tasks |

## Key Academic Concepts

- **Tokens/sec vs latency:** `tokens_per_second` measures throughput (generation speed); `latency_s` measures time-to-first-response — both matter for UX
- **GGUF quantization:** 4-bit integer weights replace 16-bit floats, cutting memory ~4x with <5% quality loss on most benchmarks
- **Chain-of-thought prompting:** Instructing the model to reason step-by-step before answering improves accuracy on multi-step problems
- **Structured output:** Constraining the model to respond in a JSON schema makes output machine-parseable — temperature 0.1 reduces hallucinated keys
