from __future__ import annotations

from pathlib import Path


# Semua path dibuat relatif terhadap root proyek agar script bisa dijalankan
# dari folder proyek tanpa perlu mengubah hard-coded path.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "models"
REPORT_DIR = RESULTS_DIR / "reports"


REQUIRED_RAW_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
    "NUSW-NB15_features.csv",
]


# Fitur kategorikal utama pada UNSW-NB15. Fitur ini akan diubah menjadi
# representasi numerik dengan One-Hot Encoding.
CATEGORICAL_FEATURES = ["proto", "service", "state"]


# Fitur yang dihapus mengikuti paper utama karena korelasi tinggi.
HIGH_CORRELATION_FEATURES = [
    "sloss",
    "ct_srv_dst",
    "ct_src_dport_ltm",
    "dpkts",
    "ltime",
    "dloss",
    "ct_dst_src_ltm",
]


# Nilai pembanding dari tabel eksperimen paper utama.
PAPER_REFERENCE = {
    "model": "Random Forest Selected Features",
    "accuracy": 0.9945,
    "precision": 0.9972,
    "recall": 0.9965,
    "f1": 0.9965,
    "far": 0.0194,
}


def create_project_directories() -> None:
    """Membuat folder output agar eksperimen tidak gagal saat menyimpan file."""
    for path in [RAW_DIR, PROCESSED_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
