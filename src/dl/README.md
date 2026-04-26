# DL Module — Deep Learning with CPU PyTorch

## Overview

Academic-grade PyTorch implementations for CPU-only deep learning. All models train in under 5 minutes on an Intel i3.

## Tasks

| Task | Architecture | Dataset | Metric |
|------|-------------|---------|--------|
| Sentiment Classification | Bidirectional LSTM (2-layer) | Inline (24 samples) | Accuracy |
| House Price Regression | Tabular MLP + BatchNorm | Synthetic (50 samples) | R², Huber loss |

## Files

```
dl/
├── driver.py           # Main entry point — trains both models
├── ui/app.py           # Streamlit training visualizer
├── models/             # Saved model weights (.pt files, auto-generated)
├── results/            # Training curve plots (auto-generated)
└── tests/test_dl.py    # Pytest test suite
```

## Run

```bash
python driver.py
streamlit run ui/app.py
pytest tests/ -v
```

## Key Academic Concepts

- **LSTM (Long Short-Term Memory):** Recurrent network that learns long-range dependencies in text sequences via gating mechanisms
- **Bidirectional LSTM:** Processes sequence forward and backward, then concatenates both final hidden states for richer context
- **Batch Normalization:** Normalizes layer inputs per mini-batch, reducing internal covariate shift — call `model.eval()` for single-sample inference
- **Huber Loss:** Combines MSE (smooth near zero) and MAE (robust to outliers) — better than pure MSE for regression with noisy targets
- **Gradient Clipping:** `clip_grad_norm_` prevents exploding gradients common in RNN training
