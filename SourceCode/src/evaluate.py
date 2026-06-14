from __future__ import annotations

"""Modul evaluasi model.

File ini menyediakan nama modul `evaluate.py` sesuai ketentuan tugas. Fungsi
aslinya direuse dari `evaluation.py` agar tidak terjadi duplikasi logika.
"""

from evaluation import (
    ExperimentResult,
    calculate_metrics,
    get_prediction_score,
    results_to_dataframe,
)


__all__ = [
    "ExperimentResult",
    "calculate_metrics",
    "get_prediction_score",
    "results_to_dataframe",
]
