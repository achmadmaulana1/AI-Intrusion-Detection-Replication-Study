from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, HIGH_CORRELATION_FEATURES, TABLE_DIR


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan nilai yang bermasalah sesuai alur paper utama."""
    df = df.copy()
    df.columns = [col.strip().replace(" ", "").lower() for col in df.columns]

    if "ct_ftp_cmd" in df.columns:
        df["ct_ftp_cmd"] = df["ct_ftp_cmd"].replace([" ", ""], 0)
        df["ct_ftp_cmd"] = pd.to_numeric(df["ct_ftp_cmd"], errors="coerce").fillna(0).astype(int)

    if "is_ftp_login" in df.columns:
        df["is_ftp_login"] = pd.to_numeric(df["is_ftp_login"], errors="coerce").fillna(0)
        df.loc[~df["is_ftp_login"].isin([0, 1]), "is_ftp_login"] = 0
        df["is_ftp_login"] = df["is_ftp_login"].astype(int)

    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
        df["attack_cat"] = df["attack_cat"].replace({"Backdoors": "Backdoor", "backdoors": "Backdoor"})
        df["attack_cat"] = df["attack_cat"].replace({"nan": np.nan, "": np.nan})

    if "label" not in df.columns:
        raise ValueError("Kolom target 'label' tidak ditemukan.")

    before = len(df)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    available_categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    if available_categorical:
        df = df.dropna(subset=available_categorical)

    print(f"[Preprocess] Baris sebelum cleaning: {before:,}; sesudah: {len(df):,}")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat fitur network_bytes = sbytes + dbytes."""
    df = df.copy()
    if "sbytes" in df.columns and "dbytes" in df.columns:
        sbytes = pd.to_numeric(df["sbytes"], errors="coerce").fillna(0)
        dbytes = pd.to_numeric(df["dbytes"], errors="coerce").fillna(0)
        df["network_bytes"] = sbytes + dbytes
    return df


def drop_high_correlation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus fitur korelasi tinggi yang disebutkan dalam paper."""
    drop_columns = [col for col in HIGH_CORRELATION_FEATURES if col in df.columns]
    print(f"[Preprocess] Menghapus fitur korelasi tinggi: {drop_columns}")
    return df.drop(columns=drop_columns)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Memisahkan X dan y, sekaligus membuang kolom non-numerik yang tidak dipakai."""
    y = df["label"].astype(int)
    target_columns = [col for col in ["label", "attack_cat"] if col in df.columns]
    X = df.drop(columns=target_columns).copy()

    dropped_non_numeric = []
    for col in X.columns:
        if col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype(str).str.strip()
            continue

        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().sum() == 0:
            dropped_non_numeric.append(col)
        else:
            X[col] = converted

    if dropped_non_numeric:
        X = X.drop(columns=dropped_non_numeric)
        pd.DataFrame({"dropped_non_numeric_feature": dropped_non_numeric}).to_csv(
            TABLE_DIR / "dropped_non_numeric_features.csv", index=False
        )
        print(f"[Preprocess] Kolom non-numerik dibuang: {dropped_non_numeric}")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Membuat preprocessing pipeline.

    Prinsip penting replication study:
    scaler dan encoder hanya boleh di-fit pada data training untuk menghindari
    data leakage.
    """
    categorical = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    numeric = [col for col in X.columns if col not in categorical]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
