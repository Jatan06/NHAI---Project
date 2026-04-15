# AI-Based Vehicle-Mounted Retroreflectivity Measurement System

This project is a simple end-to-end hackathon prototype that:

- processes an image or video from a vehicle-mounted camera,
- detects traffic signs and lane markings,
- estimates retroreflectivity from image brightness,
- stores results through a Flask API,
- and visualizes detections on a Leaflet dashboard.

## Folder Structure

```text
retro-system/
├── model/
│   ├── __init__.py
│   ├── detector.py
│   ├── processor.py
│   └── reflectivity.py
├── backend/
│   ├── __init__.py
│   ├── app.py
│   └── storage.py
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── data/
│   ├── results.json
│   ├── sample_image.png
│   └── sample_video.mp4
├── utils/
│   ├── generate_sample_assets.py
│   └── run_pipeline.py
├── requirements.txt
└── README.md
```

## How It Works

1. `utils/run_pipeline.py` loads an image or video.
2. `model/detector.py` runs YOLOv8 for traffic signs and supports optional YOLOv8 lane weights.
3. If no custom lane model is supplied, a lightweight computer-vision fallback finds lane markings so the prototype stays runnable locally.
4. `model/reflectivity.py` converts the detected region to grayscale and uses mean brightness as the retroreflectivity score.
5. The result is classified as:
   - `Good` if score > 180
   - `Moderate` if score is between 100 and 180
   - `Poor` if score < 100
6. Results are posted to the Flask backend and saved in `data/results.json`.
7. The frontend dashboard reads `/data` and plots markers on a Leaflet map.

## Setup

Use Python 3.10 to 3.12 for the smoothest dependency installation experience with Ultralytics, OpenCV, and Torch.
If you stay on Python 3.13, this project now uses `numpy>=2.2,<2.3` automatically so `pip` can install a Windows wheel instead of trying to compile NumPy from source.

### 1. Create and activate a virtual environment

```powershell
cd "B:\Work\Coding Projects\Web Projects\NHAI - Project\retro-system"
python -m venv .venv
.venv\Scripts\activate
```

If you open a new PowerShell window later, activate the venv again before running any project commands.
If you prefer not to rely on activation, you can always run the venv interpreter directly with `.venv\Scripts\python`.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### If you already hit the NumPy build error on Python 3.13

Run this inside the existing virtual environment after pulling the latest project changes:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you still want the lowest-friction path on Windows, recreate the venv with Python 3.12 and then run the same install command.

### 3. Generate sample image and video

```powershell
python utils\generate_sample_assets.py
```

### 4. Start the backend and dashboard

```powershell
.venv\Scripts\python -m backend.app
```

This module form is the cleanest way to start the Flask backend. The direct command `python backend\app.py` also works after the latest fix.

Open the dashboard in your browser:

```text
http://127.0.0.1:5000/
```

### 5. Run the image pipeline

```powershell
.venv\Scripts\python utils\run_pipeline.py --source data\sample_image.png
```

### 6. Run the video pipeline

```powershell
.venv\Scripts\python utils\run_pipeline.py --source data\sample_video.mp4 --frame-step 6
```

## API

### `POST /data`

Accepts a JSON object or a JSON array of objects.

Example payload:

```json
[
  {
    "type": "lane_marking",
    "score": 132,
    "status": "Moderate",
    "timestamp": "2026-04-15T10:00:00+00:00",
    "lat": 23.223421,
    "lng": 72.553112,
    "source": "sample_image.png",
    "frame_index": 0
  }
]
```

### `GET /data`

Returns all stored detection results from `data/results.json`.

## Sample Output

```json
{
  "type": "lane_marking",
  "score": 132,
  "status": "Moderate",
  "timestamp": "2026-04-15T10:00:00+00:00",
  "lat": 23.287451,
  "lng": 72.611248,
  "source": "sample_image.png",
  "frame_index": 0,
  "label": "lane_marking",
  "confidence": 0.56,
  "method": "cv-fallback",
  "bbox": {
    "x1": 420,
    "y1": 320,
    "x2": 520,
    "y2": 690
  }
}
```

## Notes

- `yolov8n.pt` is used out of the box for traffic-sign detection.
- If you already have custom YOLOv8 lane-marking weights, pass them with `--lane-weights path\to\best.pt`.
- The dashboard uses color-coded markers:
  - Green for `Good`
  - Yellow for `Moderate`
  - Red for `Poor`
