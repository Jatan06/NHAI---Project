from __future__ import annotations

from pathlib import Path
import sys

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.storage import append_results, ensure_results_file, load_results


FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
ensure_results_file()


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/styles.css")
def styles():
    return send_from_directory(FRONTEND_DIR, "styles.css")


@app.route("/app.js")
def script():
    return send_from_directory(FRONTEND_DIR, "app.js")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/data", methods=["GET"])
def get_data():
    return jsonify(load_results())


@app.route("/data", methods=["POST"])
def post_data():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Expected a JSON object or JSON array payload."}), 400

    if not isinstance(payload, (dict, list)):
        return jsonify({"error": "Payload must be a JSON object or JSON array."}), 400

    received_items = payload if isinstance(payload, list) else [payload]
    stored_results = append_results(received_items)

    return (
        jsonify(
            {
                "message": "Results stored successfully.",
                "received": len(received_items),
                "total": len(stored_results),
                "items": received_items,
            }
        ),
        201,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
