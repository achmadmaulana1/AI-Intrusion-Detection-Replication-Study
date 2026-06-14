from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RAW_DIR, REQUIRED_RAW_FILES


def validate_dataset_files() -> None:
    """Memastikan semua file dataset yang dibutuhkan sudah tersedia."""
    missing = [name for name in REQUIRED_RAW_FILES if not (RAW_DIR / name).exists()]
    if missing:
        message = ["File dataset belum lengkap. Letakkan file berikut di data/raw:"]
        message.extend(f"- {name}" for name in missing)
        message.append("Jalankan juga: python src/download_unsw_nb15.py")
        raise FileNotFoundError("\n".join(message))


def load_feature_names(feature_file: Path | None = None) -> list[str]:
    """Membaca nama kolom dari NUSW-NB15_features.csv.

    File fitur UNSW kadang memiliki format kolom berbeda tergantung sumber,
    sehingga fungsi ini memilih kolom Name jika tersedia.
    """
    feature_file = feature_file or RAW_DIR / "NUSW-NB15_features.csv"
    features = pd.read_csv(feature_file, encoding="ISO-8859-1")
    name_column = "Name" if "Name" in features.columns else features.columns[1]
    return (
        features[name_column]
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.lower()
        .tolist()
    )


def load_unsw_nb15() -> pd.DataFrame:
    """Menggabungkan empat file CSV UNSW-NB15 menjadi satu DataFrame."""
    validate_dataset_files()
    feature_names = load_feature_names()
    frames = []

    for index in range(1, 5):
        path = RAW_DIR / f"UNSW-NB15_{index}.csv"
        print(f"[Data] Membaca {path.name}")
        frames.append(pd.read_csv(path, header=None, low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    if len(feature_names) != df.shape[1]:
        raise ValueError(
            f"Jumlah nama fitur ({len(feature_names)}) tidak sama dengan "
            f"jumlah kolom dataset ({df.shape[1]})."
        )

    df.columns = feature_names
    print(f"[Data] Dataset gabungan: {df.shape[0]:,} baris, {df.shape[1]:,} kolom")
    return df


def stratified_sample(df: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    """Mengambil sample stratified agar eksperimen ringan di laptop tanpa GPU."""
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df

    sampled_parts = []
    for _, part in df.groupby("label"):
        n_rows = max(1, round(sample_size * len(part) / len(df)))
        sampled_parts.append(part.sample(n=min(n_rows, len(part)), random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"[Data] Menggunakan stratified sample: {len(sampled):,} baris")
    return sampled
