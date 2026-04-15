from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "results.json"


def ensure_results_file() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        DATA_FILE.write_text("[]", encoding="utf-8")


def load_results() -> List[Dict]:
    ensure_results_file()

    try:
        parsed = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def save_results(results: List[Dict]) -> None:
    ensure_results_file()
    DATA_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")


def append_results(items: Union[Dict, List[Dict]]) -> List[Dict]:
    ensure_results_file()
    existing_results = load_results()
    normalized_items = items if isinstance(items, list) else [items]
    existing_results.extend(normalized_items)
    save_results(existing_results)
    return existing_results
