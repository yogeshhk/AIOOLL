# CV Module — Classical Computer Vision

## Overview

Classical computer vision with OpenCV — no deep learning, no GPU, no CUDA. Runs at 15+ FPS on an Intel i3.

## Tasks

| Task | Algorithm | Input |
|------|----------|-------|
| Face Detection | Haar Cascade (frontal face + eye) | Image / webcam frame |
| Motion Detection | MOG2 Background Subtraction | Video frame sequence |
| Image Analysis | Canny edges, Laplacian sharpness, histogram | Any image |

## Files

```
cv/
├── driver.py           # Main entry point
├── ui/app.py           # Streamlit UI with webcam support
├── models/             # Haar cascade XML files (auto-downloaded by setup.sh)
│   └── (haarcascade_frontalface_default.xml)
├── data/               # Test images (optional)
├── results/            # Analysis plots (auto-generated)
└── tests/test_cv.py    # Offline tests (synthetic images only)
```

## Run

```bash
python driver.py
streamlit run ui/app.py
pytest tests/ -v
```

Note: If `models/haarcascade_frontalface_default.xml` is missing, the driver automatically falls back to OpenCV's bundled cascade.

## Key Academic Concepts

- **Haar Cascades:** Integral image + Adaboost cascade of weak classifiers; pre-deep-learning but still fast and accurate for frontal faces
- **MOG2 (Mixture of Gaussians v2):** Each pixel's background is modelled as a mixture of Gaussians; pixels that deviate significantly are flagged as foreground (motion)
- **Laplacian variance:** Measures image sharpness — blurry images have low high-frequency content, so the Laplacian response variance is near zero
- **Canny edge detection:** Two-threshold hysteresis filters weak edges (noise) while preserving strong edges (real boundaries)
