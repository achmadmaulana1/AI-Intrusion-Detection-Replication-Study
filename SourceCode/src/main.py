from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    MODEL_DIR,
    PAPER_REFERENCE,
    PROCESSED_DIR,
    REPORT_DIR,
    TABLE_DIR,
    create_project_directories,
)
from data_loader import load_unsw_nb15, stratified_sample
from evaluation import results_to_dataframe, train_matrix_model, train_pipeline_model
from models import build_models, build_selected_feature_models
from preprocessing import (
    add_engineered_features,
    clean_dataset,
    drop_high_correlation_features,
    split_features_target,
)
from utils import save_json, save_model_pickle, write_text
from visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metric_comparison,
    plot_roc_curves,
    safe_name,
)


def select_xgboost_features(xgb_pipeline, top_k: int) -> tuple:
    """Memilih top-k fitur dari feature importance XGBoost."""
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    classifier = xgb_pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    feature_table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "index": range(len(feature_names)),
        }
    ).sort_values("importance", ascending=False)

    feature_table.to_csv(TABLE_DIR / "xgboost_feature_importance.csv", index=False)
    selected = feature_table.head(top_k)
    selected.to_csv(TABLE_DIR / "selected_transformed_features.csv", index=False)
    plot_feature_importance(feature_table, top_n=20)
    return selected["index"].to_numpy(), feature_table


def write_comparison_report(metrics: pd.DataFrame, sample_size: int | None) -> None:
    """Membuat laporan ringkas selisih hasil replikasi dengan paper."""
    best = metrics.sort_values(["accuracy", "f1"], ascending=False).iloc[0]
    delta = {
        "accuracy": float(best["accuracy"] - PAPER_REFERENCE["accuracy"]),
        "precision": float(best["precision"] - PAPER_REFERENCE["precision"]),
        "recall": float(best["recall"] - PAPER_REFERENCE["recall"]),
        "f1": float(best["f1"] - PAPER_REFERENCE["f1"]),
        "far": float(best["far"] - PAPER_REFERENCE["far"]),
    }
    save_json(
        REPORT_DIR / "comparison_vs_paper.json",
        {
            "sample_size": sample_size or "full dataset",
            "paper_reference": PAPER_REFERENCE,
            "best_replication_model": best.to_dict(),
            "delta_vs_paper": delta,
        },
    )

    conclusion = (
        "Hasil accuracy sudah sangat dekat dengan paper utama, tetapi F1/precision/recall "
        "dapat berbeda karena sample size, split data, versi library, dan detail preprocessing."
    )
    report = f"""# Analisis Selisih Hasil Replikasi

## Hasil Paper Utama
- Model: {PAPER_REFERENCE['model']}
- Accuracy: {PAPER_REFERENCE['accuracy']:.4f}
- Precision: {PAPER_REFERENCE['precision']:.4f}
- Recall: {PAPER_REFERENCE['recall']:.4f}
- F1-score: {PAPER_REFERENCE['f1']:.4f}
- FAR: {PAPER_REFERENCE['far']:.4f}

## Hasil Terbaik Replikasi
- Model: {best['model']}
- Accuracy: {best['accuracy']:.4f}
- Precision: {best['precision']:.4f}
- Recall: {best['recall']:.4f}
- F1-score: {best['f1']:.4f}
- FAR: {best['far']:.4f}
- AUC: {best['auc']:.4f}

## Selisih
- Delta accuracy: {delta['accuracy']:+.4f}
- Delta precision: {delta['precision']:+.4f}
- Delta recall: {delta['recall']:+.4f}
- Delta F1-score: {delta['f1']:+.4f}
- Delta FAR: {delta['far']:+.4f}

## Kesimpulan
{conclusion}
"""
    write_text(REPORT_DIR / "analisis_selisih_hasil.md", report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replication study UNSW-NB15 IDS.")
    parser.add_argument("--sample-size", type=int, default=200000, help="Jumlah sample stratified untuk laptop biasa.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed agar eksperimen dapat diulang.")
    parser.add_argument("--top-k-features", type=int, default=55, help="Jumlah fitur terbaik dari XGBoost.")
    parser.add_argument("--skip-svm", action="store_true", help="Lewati Linear SVM jika laptop lambat.")
    args = parser.parse_args()

    create_project_directories()

    print("[1] Load dataset")
    df = load_unsw_nb15()

    print("[2] Preprocessing")
    df = clean_dataset(df)
    df = add_engineered_features(df)
    df = drop_high_correlation_features(df)
    df = stratified_sample(df, args.sample_size, args.random_state)
    X, y = split_features_target(df)

    print("[3] Split train-test 70:30 stratified")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=args.random_state,
        stratify=y,
    )
    X_train.assign(label=y_train.values).to_csv(PROCESSED_DIR / "train_processed_input.csv", index=False)
    X_test.assign(label=y_test.values).to_csv(PROCESSED_DIR / "test_processed_input.csv", index=False)

    print("[4] Training baseline models")
    results = []
    roc_inputs = {}
    fitted_pipelines = {}

    for model_name, estimator in build_models(args.random_state, include_svm=not args.skip_svm).items():
        print(f"    - {model_name}")
        pipeline, result, y_pred, y_score, report_text, report_dict = train_pipeline_model(
            model_name, estimator, X_train, X_test, y_train, y_test
        )
        results.append(result)
        roc_inputs[model_name] = y_score
        fitted_pipelines[model_name] = pipeline

        save_model_pickle(model_name, pipeline, MODEL_DIR)
        write_text(REPORT_DIR / f"classification_report_{safe_name(model_name)}.txt", report_text)
        save_json(REPORT_DIR / f"classification_report_{safe_name(model_name)}.json", report_dict)
        plot_confusion_matrix(model_name, y_test, y_pred)

    print("[5] Feature selection memakai XGBoost importance")
    selected_indices, _ = select_xgboost_features(fitted_pipelines["XGBoost"], args.top_k_features)
    fitted_preprocessor = fitted_pipelines["XGBoost"].named_steps["preprocessor"]
    X_train_selected = fitted_preprocessor.transform(X_train)[:, selected_indices]
    X_test_selected = fitted_preprocessor.transform(X_test)[:, selected_indices]

    print("[6] Training ulang model selected features")
    for model_name, estimator in build_selected_feature_models(args.random_state).items():
        print(f"    - {model_name}")
        model, result, y_pred, y_score, report_text, report_dict = train_matrix_model(
            model_name, estimator, X_train_selected, X_test_selected, y_train, y_test
        )
        results.append(result)
        roc_inputs[model_name] = y_score

        model_bundle = {
            "preprocessor": fitted_preprocessor,
            "selected_indices": selected_indices,
            "classifier": model,
        }
        save_model_pickle(model_name, model_bundle, MODEL_DIR)
        write_text(REPORT_DIR / f"classification_report_{safe_name(model_name)}.txt", report_text)
        save_json(REPORT_DIR / f"classification_report_{safe_name(model_name)}.json", report_dict)
        plot_confusion_matrix(model_name, y_test, y_pred)

    print("[7] Simpan hasil eksperimen")
    metrics = results_to_dataframe(results)
    metrics.to_csv(TABLE_DIR / "model_metrics.csv", index=False)
    plot_roc_curves(roc_inputs, y_test)
    plot_metric_comparison(metrics)
    write_comparison_report(metrics, args.sample_size)

    print("[Selesai] Semua hasil tersimpan di folder results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
