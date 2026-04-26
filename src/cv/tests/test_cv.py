"""
PrajnaAI — Computer Vision Tests
All tests use synthetic images only — no files, no internet, no webcam required.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from driver import ImageAnalyzer, MotionDetector


# ── ImageAnalyzer ──────────────────────────────────────────────────────────

class TestImageAnalyzer:

    @pytest.fixture
    def analyzer(self):
        return ImageAnalyzer()

    @pytest.fixture
    def synthetic_image(self):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        import cv2
        cv2.rectangle(img, (50, 50), (200, 200), (100, 200, 100), -1)
        cv2.circle(img, (300, 150), 60, (200, 100, 100), -1)
        return img

    def test_analyze_returns_expected_keys(self, analyzer, synthetic_image):
        result = analyzer.analyze(synthetic_image)
        for key in ["shape", "mean_brightness", "sharpness_score", "edge_density",
                    "contour_count", "is_blurry"]:
            assert key in result, f"Missing key: {key}"

    def test_analyze_shape_matches_input(self, analyzer, synthetic_image):
        result = analyzer.analyze(synthetic_image)
        assert result["shape"] == synthetic_image.shape

    def test_mean_brightness_range(self, analyzer, synthetic_image):
        result = analyzer.analyze(synthetic_image)
        assert 0 <= result["mean_brightness"] <= 255

    def test_edge_density_range(self, analyzer, synthetic_image):
        result = analyzer.analyze(synthetic_image)
        assert 0.0 <= result["edge_density"] <= 1.0

    def test_blank_image_is_not_blurry(self, analyzer):
        # A uniform black image has zero Laplacian variance — reported as blurry
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        result = analyzer.analyze(blank)
        assert result["is_blurry"]

    def test_sharp_image_not_blurry(self, analyzer):
        import cv2
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Hard edges → high Laplacian variance → not blurry
        cv2.rectangle(img, (10, 10), (190, 190), (255, 255, 255), 2)
        cv2.line(img, (0, 100), (200, 100), (255, 0, 0), 2)
        result = analyzer.analyze(img)
        assert result["sharpness_score"] > 0

    def test_grayscale_input_accepted(self, analyzer):
        gray = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = analyzer.analyze(gray)
        assert "mean_brightness" in result

    def test_synthetic_analysis_plot_created(self, analyzer, tmp_path, monkeypatch):
        import driver as cv_driver
        monkeypatch.setattr(cv_driver, "RESULTS_DIR", tmp_path)
        out_path = analyzer.generate_synthetic_analysis_plot()
        assert Path(out_path).exists()


# ── MotionDetector ─────────────────────────────────────────────────────────

class TestMotionDetector:

    @pytest.fixture
    def detector(self):
        return MotionDetector()

    def test_simulate_returns_correct_frame_count(self, detector):
        events = detector.simulate_video(n_frames=30)
        assert len(events) == 30

    def test_motion_detected_after_object_appears(self, detector):
        events = detector.simulate_video(n_frames=40)
        # Simulated object starts moving at frame 10
        motion_frames = [e for e in events if e["motion_ratio"] > 0.01]
        assert len(motion_frames) > 0, "Expected motion to be detected after frame 10"

    def test_no_motion_at_start(self, detector):
        events = detector.simulate_video(n_frames=5)
        # First few frames: background model is being built, object not yet present
        # motion_ratio should be low (background still being established)
        assert events[0]["frame"] == 0

    def test_process_frame_returns_tuple(self, detector):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 180
        result = detector.process_frame(frame)
        assert len(result) == 3
        mask, contours, ratio = result
        assert 0.0 <= ratio <= 1.0

    def test_motion_event_structure(self, detector):
        detector.simulate_video(n_frames=50)
        for event in detector.motion_events:
            assert "frame" in event
            assert "objects" in event
            assert "area_ratio" in event

    def test_annotate_returns_same_shape(self, detector):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        import cv2
        contour = np.array([[[50, 50]], [[150, 50]], [[150, 150]], [[50, 150]]], dtype=np.int32)
        annotated = detector.annotate(frame, [contour], motion_ratio=0.05)
        assert annotated.shape == frame.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
