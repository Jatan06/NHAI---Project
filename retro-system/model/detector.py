from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


SIGN_KEYWORDS = ("sign",)
LANE_KEYWORDS = ("lane", "marking", "line")


class RoadElementDetector:
    """YOLOv8-based detector with lightweight CV fallbacks for demo stability."""

    def __init__(
        self,
        sign_weights: str = "yolov8n.pt",
        lane_weights: Optional[str] = None,
        confidence: float = 0.25,
    ) -> None:
        self.confidence = confidence
        self.sign_model = self._load_model(sign_weights)
        self.lane_model = self._load_model(lane_weights) if lane_weights else None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        detections: List[Dict] = []

        sign_detections = self._detect_signs_with_yolo(frame) if self.sign_model is not None else []
        if not sign_detections:
            sign_detections = self._detect_signs_with_cv(frame)
        detections.extend(sign_detections)

        lane_detections = (
            self._detect_lanes_with_yolo(frame)
            if self.lane_model is not None
            else self._detect_lanes_with_cv(frame)
        )
        detections.extend(lane_detections)

        return detections

    @staticmethod
    def _load_model(weights: Optional[str]) -> Optional[YOLO]:
        if not weights:
            return None

        try:
            return YOLO(weights)
        except Exception:
            return None

    def _detect_signs_with_yolo(self, frame: np.ndarray) -> List[Dict]:
        detections: List[Dict] = []
        try:
            results = self.sign_model.predict(frame, conf=self.confidence, verbose=False)
        except Exception:
            return []

        for result in results:
            names = result.names
            for box in result.boxes:
                label = names[int(box.cls.item())].lower()
                if not any(keyword in label for keyword in SIGN_KEYWORDS):
                    continue

                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "type": "traffic_sign",
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(float(box.conf.item()), 3),
                        "method": "yolov8",
                    }
                )

        return detections

    def _detect_lanes_with_yolo(self, frame: np.ndarray) -> List[Dict]:
        detections: List[Dict] = []
        try:
            results = self.lane_model.predict(frame, conf=self.confidence, verbose=False)
        except Exception:
            return []

        for result in results:
            names = result.names
            for box in result.boxes:
                label = names[int(box.cls.item())].lower()
                if not any(keyword in label for keyword in LANE_KEYWORDS):
                    continue

                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "type": "lane_marking",
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(float(box.conf.item()), 3),
                        "method": "yolov8",
                    }
                )

        return detections

    def _detect_signs_with_cv(self, frame: np.ndarray) -> List[Dict]:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 90, 90])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([160, 90, 90])
        upper_red_2 = np.array([180, 255, 255])

        mask_1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
        mask_2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
        mask = cv2.bitwise_or(mask_1, mask_2)

        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Dict] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) < 4:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            aspect_ratio = width / float(max(height, 1))
            if not 0.7 <= aspect_ratio <= 1.3:
                continue

            detections.append(
                {
                    "type": "traffic_sign",
                    "label": "synthetic_traffic_sign",
                    "bbox": [x, y, x + width, y + height],
                    "confidence": 0.55,
                    "method": "cv-fallback",
                }
            )

        return detections

    def _detect_lanes_with_cv(self, frame: np.ndarray) -> List[Dict]:
        height, width = frame.shape[:2]
        y_offset = int(height * 0.45)
        region_of_interest = frame[y_offset:, :]

        hls_frame = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(
            hls_frame,
            np.array([0, 180, 0]),
            np.array([255, 255, 255]),
        )
        yellow_mask = cv2.inRange(
            hls_frame,
            np.array([10, 30, 115]),
            np.array([40, 255, 255]),
        )
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Dict] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 450:
                continue

            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 12 or box_height < 20:
                continue

            elongated_ratio = max(box_width, box_height) / float(max(min(box_width, box_height), 1))
            if elongated_ratio < 1.5:
                continue

            global_y = y + y_offset
            candidates.append(
                {
                    "type": "lane_marking",
                    "label": "lane_marking",
                    "bbox": [x, global_y, x + box_width, global_y + box_height],
                    "confidence": round(min(0.9, 0.45 + (area / float(height * width))), 3),
                    "method": "cv-fallback",
                    "area": area,
                }
            )

        candidates.sort(key=lambda item: item["area"], reverse=True)
        return [{key: value for key, value in candidate.items() if key != "area"} for candidate in candidates[:4]]
