from __future__ import annotations

import json
from pathlib import Path

import joblib

from visualization import safe_name


def save_json(path: Path, data: dict) -> None:
    """Menyimpan dictionary sebagai JSON yang mudah dibaca."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_model_pickle(model_name: str, model_object, model_dir: Path) -> Path:
    """Menyimpan model sebagai file .pkl sesuai kebutuhan tugas."""
    path = model_dir / f"{safe_name(model_name)}.pkl"
    joblib.dump(model_object, path)
    return path


def write_text(path: Path, text: str) -> None:
    """Menyimpan teks biasa, misalnya classification report."""
    path.write_text(text, encoding="utf-8")
