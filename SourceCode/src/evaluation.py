from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


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


def calculate_metrics(model_name: str, y_true, y_pred, y_score, train_seconds: float, n_features: int) -> ExperimentResult:
    """Menghitung metrik utama termasuk FAR."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0.0

    return ExperimentResult(
        model=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        far=far,
        auc=roc_auc_score(y_true, y_score),
        train_seconds=train_seconds,
        n_features=n_features,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def get_prediction_score(model, X):
    """Mengambil score untuk ROC/AUC dari model probabilistik atau margin model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def train_pipeline_model(model_name: str, estimator, X_train, X_test, y_train, y_test):
    """Melatih model lengkap: preprocessing + classifier."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", estimator),
        ]
    )

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    y_pred = pipeline.predict(X_test)
    y_score = get_prediction_score(pipeline, X_test)
    n_features = pipeline.named_steps["preprocessor"].transform(X_train[:1]).shape[1]
    result = calculate_metrics(model_name, y_test, y_pred, y_score, train_seconds, n_features)

    report_text = classification_report(y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0)
    report_dict = classification_report(
        y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0, output_dict=True
    )
    return pipeline, result, y_pred, y_score, report_text, report_dict


def train_matrix_model(model_name: str, estimator, X_train_matrix, X_test_matrix, y_train, y_test):
    """Melatih model pada matriks fitur terpilih hasil XGBoost importance."""
    start = time.perf_counter()
    estimator.fit(X_train_matrix, y_train)
    train_seconds = time.perf_counter() - start

    y_pred = estimator.predict(X_test_matrix)
    y_score = get_prediction_score(estimator, X_test_matrix)
    result = calculate_metrics(model_name, y_test, y_pred, y_score, train_seconds, X_train_matrix.shape[1])

    report_text = classification_report(y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0)
    report_dict = classification_report(
        y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0, output_dict=True
    )
    return estimator, result, y_pred, y_score, report_text, report_dict


def results_to_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    """Mengubah list hasil eksperimen menjadi tabel CSV-ready."""
    return pd.DataFrame([asdict(result) for result in results])
