"""
PrajnaAI — Computer Vision (Lite) Driver
==========================================
Classical computer vision with OpenCV — no deep learning required.

Implementations:
  1. Face Detection     — Haar Cascade (real-time capable)
  2. Motion Detection   — Background subtraction
  3. Image Analysis     — Histogram, edge detection, morphology
  4. Webcam Pipeline    — Live annotated feed (optional)

CPU-only. No CUDA. Runs on i3 with 4GB RAM.
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import cv2
from loguru import logger

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
for d in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

FACE_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = MODELS_DIR / "haarcascade_eye.xml"


# ═══════════════════════════════════════════════════════════════════════════
# 1. FACE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class HaarFaceDetector:
    """
    Real-time face detector using OpenCV Haar Cascades.
    Works on CPU at 15+ FPS on i3 hardware.
    Haar cascades pre-date deep learning but are remarkably fast
    and accurate for frontal face detection.
    """

    def __init__(self):
        if not FACE_CASCADE_PATH.exists():
            logger.warning(f"Haar cascade not found at {FACE_CASCADE_PATH}")
            logger.warning("Run: bash scripts/setup.sh to download models")
            # Try OpenCV's bundled cascades
            cv2_data = Path(cv2.__file__).parent / "data"
            fallback = cv2_data / "haarcascade_frontalface_default.xml"
            if fallback.exists():
                self.face_cascade = cv2.CascadeClassifier(str(fallback))
                logger.info(f"Using bundled cascade: {fallback}")
            else:
                raise FileNotFoundError("Haar cascade XML not found")
        else:
            self.face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        
        if EYE_CASCADE_PATH.exists():
            self.eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))
        else:
            self.eye_cascade = None

        logger.info("HaarFaceDetector initialized")

    def detect(self, image: np.ndarray) -> tuple[list, list]:
        """
        Detect faces and eyes in an image.
        Returns: (faces, eyes) as lists of (x, y, w, h) tuples.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = cv2.equalizeHist(gray)  # improve contrast

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        faces = list(faces) if len(faces) > 0 else []

        eyes = []
        if self.eye_cascade is not None:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                detected_eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15)
                )
                eyes.extend(detected_eyes)

        return faces, eyes

    def annotate(self, image: np.ndarray, faces: list, eyes: list) -> np.ndarray:
        """Draw detection boxes on image."""
        annotated = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated, "Face", (x, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(annotated, (ex, ey), (ex+ew, ey+eh), (255, 100, 0), 1)
        return annotated

    def benchmark_on_synthetic(self, n_images: int = 10) -> dict:
        """Benchmark detection speed on synthetic images."""
        times = []
        for _ in range(n_images):
            # Generate synthetic face-like image (gradient + circles)
            img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
            # Add face-like oval
            cv2.ellipse(img, (320, 240), (80, 100), 0, 0, 360, (200, 170, 150), -1)
            cv2.circle(img, (290, 220), 15, (80, 60, 40), -1)  # left eye
            cv2.circle(img, (350, 220), 15, (80, 60, 40), -1)  # right eye

            t0 = time.time()
            faces, eyes = self.detect(img)
            times.append(time.time() - t0)

        avg_time = np.mean(times)
        return {
            "avg_detection_time_ms": avg_time * 1000,
            "fps_estimate": 1 / avg_time,
            "n_images": n_images
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. MOTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class MotionDetector:
    """
    Background subtraction-based motion detection.
    Uses Gaussian Mixture Model (MOG2) — a classic algorithm that:
    - Maintains a statistical model of the background
    - Flags pixels that deviate significantly as foreground (motion)
    
    Zero neural networks. Works in real-time on CPU.
    """

    def __init__(self, history: int = 500, threshold: float = 16.0):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.frame_count = 0
        self.motion_events: list[dict] = []

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list, float]:
        """
        Process one frame for motion.
        Returns: (mask, contours, motion_area_ratio)
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (gray pixels → 0)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to reduce noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        # Find contours (moving objects)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = [c for c in contours if cv2.contourArea(c) > 500]

        motion_area = sum(cv2.contourArea(c) for c in significant)
        total_area = frame.shape[0] * frame.shape[1]
        motion_ratio = motion_area / total_area

        self.frame_count += 1
        if significant and motion_ratio > 0.01:
            self.motion_events.append({
                "frame": self.frame_count,
                "objects": len(significant),
                "area_ratio": motion_ratio
            })

        return fg_mask, significant, motion_ratio

    def annotate(self, frame: np.ndarray, contours: list, motion_ratio: float) -> np.ndarray:
        annotated = frame.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        color = (0, 0, 255) if motion_ratio > 0.05 else (0, 255, 0)
        status = "MOTION DETECTED" if motion_ratio > 0.01 else "No Motion"
        cv2.putText(annotated, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated, f"Motion: {motion_ratio:.1%}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return annotated

    def simulate_video(self, n_frames: int = 50) -> list[dict]:
        """Simulate a video sequence with synthetic motion."""
        events = []
        bg = np.ones((480, 640, 3), dtype=np.uint8) * 180
        x_pos = 100

        for frame_idx in range(n_frames):
            frame = bg.copy()
            # Simulated moving rectangle (intruder!)
            if frame_idx > 10:
                x_pos += 10
                cv2.rectangle(frame, (x_pos, 200), (x_pos + 60, 300), (50, 50, 200), -1)

            fg_mask, contours, ratio = self.process_frame(frame)
            events.append({"frame": frame_idx, "motion_ratio": ratio, "objects": len(contours)})

        return events


# ═══════════════════════════════════════════════════════════════════════════
# 3. IMAGE ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class ImageAnalyzer:
    """Classical image analysis: histograms, edges, morphology."""

    def analyze(self, image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection (Canny)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (255 * gray.size)

        # Blur level (Laplacian variance — higher = sharper)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Histogram statistics
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        mean_brightness = float(np.average(np.arange(256), weights=hist))
        
        # Contour count (object-like regions)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return {
            "shape": image.shape,
            "mean_brightness": round(mean_brightness, 1),
            "sharpness_score": round(float(laplacian_var), 2),
            "edge_density": round(edge_density, 4),
            "contour_count": len(contours),
            "is_blurry": laplacian_var < 100,
        }

    def generate_synthetic_analysis_plot(self) -> str:
        """Generate a synthetic demo image and analysis visualization."""
        # Create test image with some structure
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (100, 200, 100), -1)
        cv2.circle(img, (300, 150), 80, (200, 100, 100), -1)
        cv2.line(img, (0, 200), (400, 200), (200, 200, 50), 3)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Image Analysis Pipeline — PrajnaAI CV Module", fontsize=13, fontweight="bold")

        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image (Synthetic)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(edges, cmap="gray")
        axes[0, 1].set_title("Canny Edge Detection")
        axes[0, 1].axis("off")

        axes[1, 0].bar(range(256), hist, color="#3498db", alpha=0.7, width=1)
        axes[1, 0].set_title("Grayscale Histogram")
        axes[1, 0].set_xlabel("Pixel Intensity")
        axes[1, 0].set_ylabel("Frequency")

        analysis = self.analyze(img)
        metrics = [f"{k}: {v}" for k, v in analysis.items() if k != "shape"]
        axes[1, 1].text(0.05, 0.95, "\n".join(metrics),
                       transform=axes[1, 1].transAxes,
                       verticalalignment="top", fontfamily="monospace", fontsize=10,
                       bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
        axes[1, 1].set_title("Analysis Metrics")
        axes[1, 1].axis("off")

        plt.tight_layout()
        out_path = str(RESULTS_DIR / "cv_analysis.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("🕉️  PrajnaAI — Computer Vision Module (OpenCV)")
    logger.info("="*60)

    # 1. Face Detector
    logger.info("\n👁️  Task 1: Face Detection (Haar Cascade)")
    try:
        detector = HaarFaceDetector()
        bench = detector.benchmark_on_synthetic(n_images=20)
        logger.info(f"  Avg detection time: {bench['avg_detection_time_ms']:.1f}ms")
        logger.info(f"  Estimated FPS:      {bench['fps_estimate']:.1f}")
    except FileNotFoundError as e:
        logger.warning(f"Skipping face detection: {e}")

    # 2. Motion Detector
    logger.info("\n🎬  Task 2: Motion Detection (MOG2 Background Subtraction)")
    motion_det = MotionDetector()
    events = motion_det.simulate_video(n_frames=60)
    motion_frames = [e for e in events if e["motion_ratio"] > 0.01]
    logger.info(f"  Frames processed:   {len(events)}")
    logger.info(f"  Motion frames:      {len(motion_frames)}")
    logger.info(f"  Total events:       {len(motion_det.motion_events)}")

    # 3. Image Analysis
    logger.info("\n🔍  Task 3: Image Analysis Pipeline")
    analyzer = ImageAnalyzer()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    analysis = analyzer.analyze(img)
    logger.info(f"  Analysis results: {analysis}")
    plot_path = analyzer.generate_synthetic_analysis_plot()
    logger.info(f"  Analysis plot saved → {plot_path}")

    logger.info("\n✅ Computer Vision module complete!")
    logger.info("🎨 Launch UI: streamlit run src/cv/ui/app.py")


if __name__ == "__main__":
    main()
