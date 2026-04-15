from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_PATH = DATA_DIR / "sample_image.png"
VIDEO_PATH = DATA_DIR / "sample_video.mp4"


def draw_scene(frame: np.ndarray, shift: int = 0) -> np.ndarray:
    height, width = frame.shape[:2]

    frame[:] = (180, 205, 225)
    cv2.rectangle(frame, (0, int(height * 0.35)), (width, height), (72, 72, 72), -1)

    left_lane = np.array(
        [
            [int(width * 0.33) + shift, height],
            [int(width * 0.42) + shift, int(height * 0.4)],
            [int(width * 0.46) + shift, int(height * 0.4)],
            [int(width * 0.38) + shift, height],
        ]
    )
    right_lane = np.array(
        [
            [int(width * 0.62) + shift, height],
            [int(width * 0.54) + shift, int(height * 0.4)],
            [int(width * 0.58) + shift, int(height * 0.4)],
            [int(width * 0.67) + shift, height],
        ]
    )
    center_marking = np.array(
        [
            [int(width * 0.49) + shift, int(height * 0.98)],
            [int(width * 0.51) + shift, int(height * 0.98)],
            [int(width * 0.53) + shift, int(height * 0.55)],
            [int(width * 0.47) + shift, int(height * 0.55)],
        ]
    )

    cv2.fillPoly(frame, [left_lane], (245, 245, 245))
    cv2.fillPoly(frame, [right_lane], (245, 245, 245))
    cv2.fillPoly(frame, [center_marking], (0, 220, 255))

    sign_center = (110 + shift, 130)
    sign_radius = 46
    octagon = []
    for index in range(8):
        angle = np.deg2rad(22.5 + (45 * index))
        x_coordinate = int(sign_center[0] + sign_radius * np.cos(angle))
        y_coordinate = int(sign_center[1] + sign_radius * np.sin(angle))
        octagon.append([x_coordinate, y_coordinate])

    cv2.fillPoly(frame, [np.array(octagon)], (20, 20, 220))
    cv2.polylines(frame, [np.array(octagon)], True, (255, 255, 255), 4)
    cv2.putText(
        frame,
        "STOP",
        (sign_center[0] - 28, sign_center[1] + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def create_sample_image() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image = draw_scene(image)
    cv2.imwrite(str(IMAGE_PATH), image)


def create_sample_video() -> None:
    frame_size = (1280, 720)
    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        frame_size,
    )

    for index in range(24):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = draw_scene(frame, shift=index % 6)
        writer.write(frame)

    writer.release()


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    create_sample_image()
    create_sample_video()
    print(f"Sample image saved to: {IMAGE_PATH}")
    print(f"Sample video saved to: {VIDEO_PATH}")
