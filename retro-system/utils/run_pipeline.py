from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.processor import RetroreflectivityPipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process an image or video, then send detections to the backend API."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an input image or video.",
    )
    parser.add_argument(
        "--backend-url",
        default="http://127.0.0.1:5000",
        help="Base URL of the Flask backend.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=6,
        help="Process every Nth frame for videos.",
    )
    parser.add_argument(
        "--sign-weights",
        default="yolov8n.pt",
        help="YOLOv8 weights used for traffic-sign detection.",
    )
    parser.add_argument(
        "--lane-weights",
        default=None,
        help="Optional custom YOLOv8 weights for lane-marking detection.",
    )
    return parser.parse_args()


def post_results(results: List[dict], backend_url: str) -> dict:
    response = requests.post(
        f"{backend_url.rstrip('/')}/data",
        json=results,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    arguments = parse_arguments()
    source_path = Path(arguments.source).resolve()

    pipeline = RetroreflectivityPipeline(
        sign_weights=arguments.sign_weights,
        lane_weights=arguments.lane_weights,
    )
    results = pipeline.process_source(str(source_path), frame_step=arguments.frame_step)

    if not results:
        print("No road elements detected. Try a clearer input frame or lower the frame step.")
        return

    try:
        response_payload = post_results(results, arguments.backend_url)
    except requests.RequestException as error:
        print(f"Processed source: {source_path.name}")
        print(f"Detections found locally: {len(results)}")
        print(f"Could not reach backend at {arguments.backend_url}: {error}")
        print("Start the Flask backend and rerun this command to store the results.")
        print("Sample output:")
        print(results[0])
        return

    print(f"Processed source: {source_path.name}")
    print(f"Detections sent to backend: {len(results)}")
    print(f"Total records currently stored: {response_payload['total']}")
    print("Sample output:")
    print(results[0])


if __name__ == "__main__":
    main()
