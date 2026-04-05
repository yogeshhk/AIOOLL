# Community Benchmark Results

This file collects benchmark results from the community running PrajnaAI on various CPU-only hardware.
Add your results via a Pull Request!

## Format

| Hardware | RAM | OS | Model | Task | Speed | Notes |
|----------|-----|----|-------|------|-------|-------|
| Intel i3-8130U | 8GB DDR4 | Xubuntu 22.04 | gemma2:2b | Chat | 8 tok/s | SSD, Q4_K_M |
| Intel i5-7200U | 16GB DDR4 | Ubuntu 22.04 | gemma2:2b | Chat | 11 tok/s | SSD, Q4_K_M |
| Intel i7-6700HQ | 16GB DDR4 | Debian 12 | phi3:mini | Chat | 6 tok/s | SSD, Q4_K_M |
| AMD Ryzen 5 3500U | 8GB | Fedora 39 | qwen2:1.5b | Chat | 18 tok/s | NVMe SSD |
| Intel Celeron N3060 | 4GB | Lubuntu 22.04 | tinyllama:1.1b | Chat | 3 tok/s | HDD, limited |
| Raspberry Pi 4 | 8GB | Raspberry Pi OS | tinyllama:1.1b | Chat | 1.5 tok/s | SD card |

## ML Module Benchmarks

| Hardware | RAM | Task | Algorithm | Time |
|----------|-----|------|-----------|------|
| Intel i3-8130U | 8GB | Spam Classification | Logistic Regression | 0.08s |
| Intel i3-8130U | 8GB | House Price | Gradient Boosting | 2.1s |
| Intel i5-7200U | 16GB | Spam Classification | Logistic Regression | 0.05s |

## RAG Pipeline Benchmarks

| Hardware | RAM | Embed Model | Doc Count | Index Time | Query Latency |
|----------|-----|-------------|-----------|------------|---------------|
| Intel i3-8130U | 8GB | nomic-embed-text | 5 docs | 12s | 0.8s |
| Intel i5-7200U | 16GB | nomic-embed-text | 10 docs | 8s | 0.5s |

## How to Run Your Own Benchmark

```bash
# LLM inference benchmark
cd src/llm_inference
python driver.py

# Check results
cat results/benchmark_results.json
```

## Submit Your Results

1. Fork the repo
2. Add your row to the table above
3. Include your hardware model from `/proc/cpuinfo` and `free -h`
4. Submit a PR with title: `bench: [CPU model] results`
