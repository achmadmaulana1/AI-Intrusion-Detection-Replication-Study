from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "models"
REPORT_DIR = ROOT / "results" / "reports"

PAPER_REFERENCE = {
    "model": "RF with selected features",
    "accuracy": 0.9945,
    "precision": 0.9972,
    "recall": 0.9965,
    "f1": 0.9965,
    "far": 0.0194,
}

DROP_CORRELATED = [
    "sloss",
    "ct_srv_dst",
    "ct_src_dport_ltm",
    "dpkts",
    "ltime",
    "dloss",
    "ct_dst_src_ltm",
]

CATEGORICAL = ["proto", "service", "state"]


@dataclass
class ExperimentResult:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    auc: float
    train_seconds: float
    n_features: int
    tn: int
    fp: int
    fn: int
    tp: int


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_feature_names() -> list[str]:
    feature_file = RAW_DIR / "NUSW-NB15_features.csv"
    if not feature_file.exists():
        raise FileNotFoundError(
            f"File fitur tidak ditemukan: {feature_file}. Jalankan src/download_unsw_nb15.py."
        )
    features = pd.read_csv(feature_file, encoding="ISO-8859-1")
    name_col = "Name" if "Name" in features.columns else features.columns[1]
    names = (
        features[name_col]
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.lower()
        .tolist()
    )
    return names


def load_full_dataset() -> pd.DataFrame:
    files = [RAW_DIR / f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Empat CSV UNSW-NB15 belum lengkap:\n"
            + "\n".join(missing)
            + "\nJalankan src/download_unsw_nb15.py atau letakkan file di data/raw/."
        )
    names = load_feature_names()
    frames = []
    for path in files:
        print(f"[Fase 2] Membaca {path.name}")
        frames.append(pd.read_csv(path, header=None, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    if len(names) == df.shape[1]:
        df.columns = names
    else:
        raise ValueError(
            f"Jumlah nama fitur ({len(names)}) tidak sama dengan jumlah kolom data ({df.shape[1]})."
        )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[Fase 4] Membersihkan missing value dan nilai tidak konsisten")
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
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Drop rows with missing values in categorical columns; numeric nulls are imputed in pipeline.
    available_cats = [col for col in CATEGORICAL if col in df.columns]
    if available_cats:
        df = df.dropna(subset=available_cats)

    print(f"[Fase 4] Baris sebelum cleaning: {before:,}; setelah cleaning: {len(df):,}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[Fase 6] Membuat network_bytes dan menghapus fitur korelasi tinggi")
    df = df.copy()
    if "sbytes" in df.columns and "dbytes" in df.columns:
        df["network_bytes"] = pd.to_numeric(df["sbytes"], errors="coerce").fillna(0) + pd.to_numeric(
            df["dbytes"], errors="coerce"
        ).fillna(0)
    drop_cols = [col for col in DROP_CORRELATED if col in df.columns]
    return df.drop(columns=drop_cols)


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    y = df["label"].astype(int)
    drop_target = [col for col in ["label", "attack_cat"] if col in df.columns]
    X = df.drop(columns=drop_target).copy()

    dropped_non_numeric: list[str] = []
    for col in X.columns:
        if col in CATEGORICAL:
            X[col] = X[col].astype(str).str.strip()
            continue
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().sum() == 0:
            dropped_non_numeric.append(col)
        else:
            X[col] = converted

    if dropped_non_numeric:
        X = X.drop(columns=dropped_non_numeric)
    return X, y, dropped_non_numeric


def sample_if_requested(df: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df
    print(f"[Fase 3] Menggunakan stratified sample {sample_size:,} baris untuk mode laptop")
    sampled_parts = []
    for _, part in df.groupby("label"):
        n = max(1, round(sample_size * len(part) / len(df)))
        sampled_parts.append(part.sample(n=min(n, len(part)), random_state=random_state))
    return pd.concat(sampled_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [col for col in CATEGORICAL if col in X.columns]
    numeric = [col for col in X.columns if col not in categorical]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return np.array([f"feature_{i}" for i in range(len(preprocessor.get_feature_names_out()))])


def fit_evaluate(
    name: str,
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[ExperimentResult, Pipeline]:
    print(f"[Fase 9/13] Training dan evaluasi: {name}")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", estimator),
        ]
    )
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = pipeline.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    result = ExperimentResult(
        model=name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        far=fp / (fp + tn) if (fp + tn) else 0.0,
        auc=roc_auc_score(y_test, y_score),
        train_seconds=train_seconds,
        n_features=pipeline.named_steps["preprocessor"].transform(X_train[:1]).shape[1],
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
    return result, pipeline


def plot_confusion_matrix(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"confusion_matrix_{slug(name)}.png", dpi=180)
    plt.close()


def plot_confusion_matrix_from_predictions(name: str, y_test: pd.Series, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"confusion_matrix_{slug(name)}.png", dpi=180)
    plt.close()


def plot_roc_curves(pipelines: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    plt.figure(figsize=(8, 6))
    for name, pipeline in pipelines.items():
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_score = pipeline.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curves.png", dpi=180)
    plt.close()


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def select_top_features(
    xgb_pipeline: Pipeline,
    top_k: int,
) -> tuple[np.ndarray, list[str]]:
    print(f"[Fase 11] Mengambil {top_k} fitur terpenting dari XGBoost")
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    classifier = xgb_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    ranking = pd.DataFrame({"feature": feature_names, "importance": importances, "index": np.arange(len(feature_names))})
    ranking = ranking.sort_values("importance", ascending=False)
    ranking.to_csv(TABLE_DIR / "xgboost_feature_importance.csv", index=False)

    selected = ranking.head(top_k).copy()
    selected.to_csv(TABLE_DIR / "selected_transformed_features.csv", index=False)
    return selected["index"].to_numpy(dtype=int), selected["feature"].tolist()


def fit_evaluate_selected_matrix(
    name: str,
    estimator,
    X_train_matrix,
    X_test_matrix,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[ExperimentResult, object]:
    print(f"[Fase 12/13] Training dan evaluasi selected features: {name}")
    start = time.perf_counter()
    estimator.fit(X_train_matrix, y_train)
    train_seconds = time.perf_counter() - start
    y_pred = estimator.predict(X_test_matrix)
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test_matrix)[:, 1]
    else:
        y_score = estimator.decision_function(X_test_matrix)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    result = ExperimentResult(
        model=name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        far=fp / (fp + tn) if (fp + tn) else 0.0,
        auc=roc_auc_score(y_test, y_score),
        train_seconds=train_seconds,
        n_features=X_train_matrix.shape[1],
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
    plot_confusion_matrix_from_predictions(name, y_test, y_pred)
    return result, estimator


def write_report(results: list[ExperimentResult], sample_size: int | None, n_rows: int) -> None:
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(TABLE_DIR / "model_metrics.csv", index=False)

    best = df.sort_values(["accuracy", "f1"], ascending=False).iloc[0]
    comparison = {
        "paper_reference": PAPER_REFERENCE,
        "best_replication_model": best.to_dict(),
        "delta_vs_paper_rf_selected": {
            "accuracy": float(best["accuracy"] - PAPER_REFERENCE["accuracy"]),
            "precision": float(best["precision"] - PAPER_REFERENCE["precision"]),
            "recall": float(best["recall"] - PAPER_REFERENCE["recall"]),
            "f1": float(best["f1"] - PAPER_REFERENCE["f1"]),
            "far": float(best["far"] - PAPER_REFERENCE["far"]),
        },
    }
    (REPORT_DIR / "comparison_vs_paper.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    replicated = (
        abs(best["accuracy"] - PAPER_REFERENCE["accuracy"]) <= 0.01
        and abs(best["f1"] - PAPER_REFERENCE["f1"]) <= 0.01
        and abs(best["far"] - PAPER_REFERENCE["far"]) <= 0.02
    )

    lines = [
        "# Analisis Selisih Hasil Replikasi",
        "",
        f"Jumlah baris eksperimen: {n_rows:,}",
        f"Mode sample-size: {sample_size if sample_size else 'full dataset'}",
        "",
        "## Hasil Paper Utama",
        "",
        f"- Model acuan: {PAPER_REFERENCE['model']}",
        f"- Accuracy: {PAPER_REFERENCE['accuracy']:.4f}",
        f"- Precision: {PAPER_REFERENCE['precision']:.4f}",
        f"- Recall: {PAPER_REFERENCE['recall']:.4f}",
        f"- F1-score: {PAPER_REFERENCE['f1']:.4f}",
        f"- FAR: {PAPER_REFERENCE['far']:.4f}",
        "",
        "## Hasil Replikasi Terbaik",
        "",
        f"- Model: {best['model']}",
        f"- Accuracy: {best['accuracy']:.4f}",
        f"- Precision: {best['precision']:.4f}",
        f"- Recall: {best['recall']:.4f}",
        f"- F1-score: {best['f1']:.4f}",
        f"- FAR: {best['far']:.4f}",
        f"- AUC: {best['auc']:.4f}",
        "",
        "## Selisih terhadap Paper",
        "",
        f"- Delta accuracy: {best['accuracy'] - PAPER_REFERENCE['accuracy']:+.4f}",
        f"- Delta precision: {best['precision'] - PAPER_REFERENCE['precision']:+.4f}",
        f"- Delta recall: {best['recall'] - PAPER_REFERENCE['recall']:+.4f}",
        f"- Delta F1-score: {best['f1'] - PAPER_REFERENCE['f1']:+.4f}",
        f"- Delta FAR: {best['far'] - PAPER_REFERENCE['far']:+.4f}",
        "",
        "## Kesimpulan Replikasi",
        "",
        (
            "Hasil dapat dinyatakan mendekati paper utama berdasarkan toleransi accuracy/F1 <= 0.01 "
            "dan FAR <= 0.02."
            if replicated
            else "Hasil belum sepenuhnya mendekati paper utama. Penyebab yang paling mungkin adalah perbedaan split data, "
            "jumlah baris setelah cleaning, versi library, proses encoding, atau penggunaan sample-size."
        ),
    ]
    (REPORT_DIR / "analisis_selisih_hasil.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UNSW-NB15 replication workflow.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional stratified sample size for CPU-friendly runs.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k-features", type=int, default=55)
    parser.add_argument("--skip-svm", action="store_true", help="Skip Linear SVM if runtime is too high.")
    args = parser.parse_args()

    ensure_dirs()

    print("[Fase 1] Environment siap")
    df = load_full_dataset()
    print(f"[Fase 3] Dataset awal: {df.shape[0]:,} baris, {df.shape[1]:,} kolom")
    df = clean_data(df)
    df = engineer_features(df)
    df = sample_if_requested(df, args.sample_size, args.random_state)

    X, y, dropped_non_numeric = prepare_features_and_target(df)
    if dropped_non_numeric:
        pd.DataFrame({"dropped_non_numeric_feature": dropped_non_numeric}).to_csv(
            TABLE_DIR / "dropped_non_numeric_features.csv", index=False
        )
        print(f"[Fase 7] Kolom non-numerik yang dikeluarkan: {', '.join(dropped_non_numeric)}")
    print(f"[Fase 10] Fitur sebelum preprocessing: {X.shape[1]:,}; target: label")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=args.random_state,
        stratify=y,
    )
    print(f"[Fase 8] Train: {len(X_train):,}; Test: {len(X_test):,}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, random_state=args.random_state),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=6,
            min_samples_leaf=9,
            random_state=args.random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=22,
            min_samples_split=6,
            criterion="gini",
            n_jobs=-1,
            random_state=args.random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=12,
            learning_rate=0.1,
            colsample_bylevel=0.5,
            subsample=0.1,
            eval_metric="logloss",
            tree_method="hist",
            random_state=args.random_state,
            n_jobs=-1,
        ),
    }
    if not args.skip_svm:
        models["Linear SVM"] = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=args.random_state,
            n_jobs=-1,
        )

    results: list[ExperimentResult] = []
    pipelines: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        result, pipeline = fit_evaluate(name, estimator, X_train, X_test, y_train, y_test)
        results.append(result)
        pipelines[name] = pipeline
        plot_confusion_matrix(name, pipeline, X_test, y_test)
        joblib.dump(pipeline, MODEL_DIR / f"{slug(name)}.joblib")

    selected_indices, _ = select_top_features(pipelines["XGBoost"], args.top_k_features)
    fitted_preprocessor = pipelines["XGBoost"].named_steps["preprocessor"]
    X_train_transformed = fitted_preprocessor.transform(X_train)
    X_test_transformed = fitted_preprocessor.transform(X_test)
    X_train_sel = X_train_transformed[:, selected_indices]
    X_test_sel = X_test_transformed[:, selected_indices]

    selected_models = {
        "Decision Tree Selected Features": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=6,
            min_samples_leaf=9,
            random_state=args.random_state,
        ),
        "Random Forest Selected Features": RandomForestClassifier(
            n_estimators=300,
            max_depth=22,
            min_samples_split=6,
            criterion="gini",
            n_jobs=-1,
            random_state=args.random_state,
        ),
    }
    for name, estimator in selected_models.items():
        result, pipeline = fit_evaluate_selected_matrix(name, estimator, X_train_sel, X_test_sel, y_train, y_test)
        results.append(result)
        joblib.dump(
            {"preprocessor": fitted_preprocessor, "selected_indices": selected_indices, "classifier": pipeline},
            MODEL_DIR / f"{slug(name)}.joblib",
        )

    # ROC curves for all-feature models only; selected-feature models need reduced X_test.
    plot_roc_curves({k: pipelines[k] for k in models.keys()}, X_test, y_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.assign(label=y_train.values).to_csv(PROCESSED_DIR / "train_processed_input.csv", index=False)
    X_test.assign(label=y_test.values).to_csv(PROCESSED_DIR / "test_processed_input.csv", index=False)

    write_report(results, args.sample_size, len(df))
    print("[Fase 14-16] Selesai. Lihat results/tables, results/figures, dan results/reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
