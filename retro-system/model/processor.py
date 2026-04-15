from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from model.detector import RoadElementDetector
from model.reflectivity import calculate_reflectivity_score, classify_reflectivity


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


class RetroreflectivityPipeline:
    """Detect road elements, estimate reflectivity, and package API-ready output."""

    def __init__(
        self,
        sign_weights: str = "yolov8n.pt",
        lane_weights: Optional[str] = None,
        confidence: float = 0.25,
    ) -> None:
        self.detector = RoadElementDetector(
            sign_weights=sign_weights,
            lane_weights=lane_weights,
            confidence=confidence,
        )

    def process_source(self, source_path: str, frame_step: int = 10) -> List[Dict]:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        suffix = source.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            return self.process_image(source)
        if suffix in VIDEO_SUFFIXES:
            return self.process_video(source, frame_step=frame_step)

        raise ValueError(f"Unsupported file type: {suffix}")

    def process_image(self, source: Path) -> List[Dict]:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Could not read image: {source}")

        detections = self.detector.detect(frame)
        return self._build_results(frame, detections, source.name, frame_index=0)

    def process_video(self, source: Path, frame_step: int = 10) -> List[Dict]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {source}")

        results: List[Dict] = []
        frame_index = 0
        effective_frame_step = max(frame_step, 1)

        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % effective_frame_step == 0:
                detections = self.detector.detect(frame)
                results.extend(
                    self._build_results(
                        frame,
                        detections,
                        source.name,
                        frame_index=frame_index,
                    )
                )

            frame_index += 1

        capture.release()
        return results

    def _build_results(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        source_name: str,
        frame_index: int,
    ) -> List[Dict]:
        height, width = frame.shape[:2]
        timestamp = datetime.now(timezone.utc).isoformat()
        packaged_results: List[Dict] = []

        for detection in detections:
            x1, y1, x2, y2 = self._clip_bbox(detection["bbox"], width=width, height=height)
            region = frame[y1:y2, x1:x2]
            score = calculate_reflectivity_score(region)

            packaged_results.append(
                {
                    "type": detection["type"],
                    "score": score,
                    "status": classify_reflectivity(score),
                    "timestamp": timestamp,
                    "lat": round(random.uniform(23.0, 23.5), 6),
                    "lng": round(random.uniform(72.4, 72.7), 6),
                    "source": source_name,
                    "frame_index": frame_index,
                    "label": detection.get("label", detection["type"]),
                    "confidence": detection.get("confidence", 0.0),
                    "method": detection.get("method", "yolov8"),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

        return packaged_results

    @staticmethod
    def _clip_bbox(bbox: List[int], width: int, height: int) -> List[int]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return [x1, y1, x2, y2]
