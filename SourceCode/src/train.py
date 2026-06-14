from __future__ import annotations

"""Modul training untuk tugas replikasi.

File ini sengaja dibuat sebagai wrapper yang mudah dijelaskan saat presentasi.
Logika model utama berada di `models.py` dan fungsi training berada di
`evaluation.py`. Dengan struktur ini, mahasiswa dapat menunjukkan bahwa proses
training dipisahkan dari preprocessing dan visualisasi.
"""

from models import build_models, build_selected_feature_models
from evaluation import train_pipeline_model, train_matrix_model


__all__ = [
    "build_models",
    "build_selected_feature_models",
    "train_pipeline_model",
    "train_matrix_model",
]
