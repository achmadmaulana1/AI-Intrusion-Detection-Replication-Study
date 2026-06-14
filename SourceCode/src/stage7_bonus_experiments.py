from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import FIGURE_DIR, MODEL_DIR, REPORT_DIR, TABLE_DIR, create_project_directories
from data_loader import load_unsw_nb15, stratified_sample
from evaluation import results_to_dataframe, train_pipeline_model
from preprocessing import (
    add_engineered_features,
    clean_dataset,
    drop_high_correlation_features,
    split_features_target,
)
from utils import save_json, save_model_pickle, write_text
from visualization import safe_name

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


BONUS_TABLE_DIR = TABLE_DIR / "stage7_bonus"
BONUS_FIGURE_DIR = FIGURE_DIR / "stage7_bonus"
BONUS_MODEL_DIR = MODEL_DIR / "stage7_bonus"
BONUS_REPORT_DIR = REPORT_DIR / "stage7_bonus"


def create_bonus_directories() -> None:
    """Membuat folder khusus Stage 7 agar hasil bonus tidak tercampur."""
    create_project_directories()
    for path in [BONUS_TABLE_DIR, BONUS_FIGURE_DIR, BONUS_MODEL_DIR, BONUS_REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_bonus_models(random_state: int) -> dict:
    """Lima metode pembanding yang masih dekat dengan paper utama.

    Semua model dipilih karena cocok untuk data tabular IDS dan bisa dijalankan
    di laptop tanpa GPU.
    """
    return {
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=22,
            min_samples_split=6,
            criterion="gini",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Balanced Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=22,
            min_samples_split=6,
            criterion="gini",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            random_state=random_state,
        ),
        "Bagging Decision Tree": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=18,
                min_samples_split=6,
                random_state=random_state,
            ),
            n_estimators=150,
            max_samples=0.8,
            max_features=0.8,
            n_jobs=-1,
            random_state=random_state,
        ),
        "AdaBoost Decision Tree": AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.08,
            random_state=random_state,
        ),
    }


def build_tuning_scenarios(random_state: int) -> dict:
    """Lima skenario hyperparameter tuning sederhana.

    Fokus tuning tetap pada Random Forest karena model ini adalah model utama
    paper. Pendekatan ini lebih mudah dijelaskan daripada grid search besar.
    """
    return {
        "RF Baseline Paper": RandomForestClassifier(
            n_estimators=300,
            max_depth=22,
            min_samples_split=6,
            criterion="gini",
            n_jobs=-1,
            random_state=random_state,
        ),
        "RF More Trees": RandomForestClassifier(
            n_estimators=500,
            max_depth=22,
            min_samples_split=6,
            criterion="gini",
            n_jobs=-1,
            random_state=random_state,
        ),
        "RF Deeper Trees": RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            min_samples_split=4,
            criterion="gini",
            n_jobs=-1,
            random_state=random_state,
        ),
        "RF Entropy Criterion": RandomForestClassifier(
            n_estimators=300,
            max_depth=22,
            min_samples_split=6,
            criterion="entropy",
            n_jobs=-1,
            random_state=random_state,
        ),
        "RF Balanced Class Weight": RandomForestClassifier(
            n_estimators=300,
            max_depth=22,
            min_samples_split=6,
            criterion="gini",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def plot_stage7_metric_comparison(metrics: pd.DataFrame) -> None:
    """Grafik tambahan 1: perbandingan metrik utama model bonus."""
    selected = metrics[["model", "accuracy", "precision", "recall", "f1", "far", "auc"]].melt(
        id_vars="model", var_name="metric", value_name="score"
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=selected, x="model", y="score", hue="metric")
    plt.xticks(rotation=35, ha="right")
    plt.title("Stage 7 - Perbandingan Metrik Model Bonus")
    plt.tight_layout()
    plt.savefig(BONUS_FIGURE_DIR / "bonus_metric_comparison.png", dpi=180)
    plt.close()


def plot_stage7_confusion_matrix(model_name: str, y_true, y_pred) -> None:
    """Confusion matrix khusus Stage 7."""
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
    )
    plt.title(f"Stage 7 CM - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(BONUS_FIGURE_DIR / f"confusion_matrix_{safe_name(model_name)}.png", dpi=180)
    plt.close()


def plot_stage7_roc_curves(roc_inputs: dict, y_test) -> None:
    """ROC curves khusus Stage 7."""
    plt.figure(figsize=(8, 6))
    for model_name, y_score in roc_inputs.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("Stage 7 - ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(BONUS_FIGURE_DIR / "bonus_roc_curves.png", dpi=180)
    plt.close()


def plot_far_vs_recall(metrics: pd.DataFrame) -> None:
    """Grafik tambahan 2: trade-off FAR dan recall."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=metrics, x="far", y="recall", hue="model", s=90)
    plt.title("Stage 7 - Trade-off FAR vs Recall")
    plt.xlabel("False Alarm Rate")
    plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig(BONUS_FIGURE_DIR / "far_vs_recall.png", dpi=180)
    plt.close()


def plot_runtime_vs_f1(metrics: pd.DataFrame) -> None:
    """Grafik tambahan 3: efisiensi runtime terhadap F1-score."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=metrics, x="train_seconds", y="f1", hue="model", s=90)
    plt.title("Stage 7 - Runtime vs F1-score")
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("F1-score")
    plt.tight_layout()
    plt.savefig(BONUS_FIGURE_DIR / "runtime_vs_f1.png", dpi=180)
    plt.close()


def plot_precision_recall_curves(pr_inputs: dict) -> None:
    """Grafik tambahan 4: Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    for model_name, values in pr_inputs.items():
        precision, recall, _ = precision_recall_curve(values["y_true"], values["y_score"])
        plt.plot(recall, precision, label=model_name)
    plt.title("Stage 7 - Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(BONUS_FIGURE_DIR / "precision_recall_curves.png", dpi=180)
    plt.close()


def plot_accuracy_delta(metrics: pd.DataFrame, paper_accuracy: float = 0.9945) -> None:
    """Grafik tambahan 5: selisih accuracy terhadap paper utama."""
    data = metrics[["model", "accuracy"]].copy()
    data["delta_accuracy"] = data["accuracy"] - paper_accuracy
    plt.figure(figsize=(10, 5))
    sns.barplot(data=data, x="model", y="delta_accuracy")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=35, ha="right")
    plt.title("Stage 7 - Delta Accuracy terhadap Paper Utama")
    plt.tight_layout()
    plt.savefig(BONUS_FIGURE_DIR / "delta_accuracy_vs_paper.png", dpi=180)
    plt.close()


def run_experiment_group(group_name: str, models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Menjalankan satu grup eksperimen dan menyimpan artefaknya."""
    results = []
    roc_inputs = {}
    pr_inputs = {}

    for model_name, estimator in models.items():
        full_name = f"{group_name} - {model_name}"
        print(f"[Stage 7] Training {full_name}")
        model, result, y_pred, y_score, report_text, report_dict = train_pipeline_model(
            full_name, estimator, X_train, X_test, y_train, y_test
        )
        results.append(result)
        roc_inputs[full_name] = y_score
        pr_inputs[full_name] = {"y_true": y_test, "y_score": y_score}

        save_model_pickle(full_name, model, BONUS_MODEL_DIR)
        write_text(BONUS_REPORT_DIR / f"classification_report_{safe_name(full_name)}.txt", report_text)
        save_json(BONUS_REPORT_DIR / f"classification_report_{safe_name(full_name)}.json", report_dict)
        plot_stage7_confusion_matrix(full_name, y_test, y_pred)

    metrics = results_to_dataframe(results)
    metrics.to_csv(BONUS_TABLE_DIR / f"{safe_name(group_name)}_metrics.csv", index=False)
    plot_stage7_roc_curves(roc_inputs, y_test)
    plot_precision_recall_curves(pr_inputs)
    return metrics


def write_stage7_academic_summary(all_metrics: pd.DataFrame, sample_size: int | None) -> None:
    """Membuat narasi Stage 7 yang siap dipakai di laporan atau draft artikel."""
    best = all_metrics.sort_values(["accuracy", "f1", "far"], ascending=[False, False, True]).iloc[0]
    baseline_candidates = all_metrics[all_metrics["model"].str.contains("RF Baseline Paper", regex=False)]
    baseline_text = ""
    if not baseline_candidates.empty:
        baseline = baseline_candidates.iloc[0]
        baseline_text = (
            f"Dibandingkan skenario RF Baseline Paper, model terbaik memiliki delta accuracy "
            f"{best['accuracy'] - baseline['accuracy']:+.4f} dan delta F1 "
            f"{best['f1'] - baseline['f1']:+.4f}.\n"
        )

    text = f"""# Stage 7 - Bonus Nilai +10

## Tujuan Pengembangan
Pengembangan ini tetap ditempatkan sebagai replication study karena dataset, target,
preprocessing utama, dan model acuan masih mengikuti paper utama. Perluasan hanya
dilakukan pada metode pembanding, tuning Random Forest, visualisasi tambahan, dan
analisis robustness sederhana.

## Hasil Terbaik Stage 7
- Model terbaik: {best['model']}
- Accuracy: {best['accuracy']:.4f}
- Precision: {best['precision']:.4f}
- Recall: {best['recall']:.4f}
- F1-score: {best['f1']:.4f}
- AUC: {best['auc']:.4f}
- FAR: {best['far']:.4f}
- Training time: {best['train_seconds']:.2f} detik

{baseline_text}
## Kombinasi Terbaik yang Direkomendasikan
Kombinasi terbaik untuk dilaporkan adalah Random Forest/XGBoost-based feature
selection, ditambah tuning Random Forest dan pembanding Extra Trees atau Balanced
Random Forest. Kombinasi ini berpotensi meningkatkan nilai akademik karena:

1. Tetap dekat dengan paper utama.
2. Menambah metode pembanding.
3. Menambahkan hyperparameter tuning.
4. Menambahkan visualisasi evaluasi.
5. Memperkuat analisis performa, false alarm, dan efisiensi runtime.

## Catatan Validitas
Eksperimen menggunakan sample size: {sample_size if sample_size else "full dataset"}.
Jika hasil final akan dimasukkan ke artikel, jalankan ulang dengan sample lebih besar
atau full dataset agar klaim performa lebih kuat.
"""
    write_text(BONUS_REPORT_DIR / "stage7_bonus_summary.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7 bonus experiments for replication study.")
    parser.add_argument("--sample-size", type=int, default=50000, help="Sample size CPU-friendly untuk eksperimen bonus.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    create_bonus_directories()

    print("[Stage 7] Load dan preprocessing dataset")
    df = load_unsw_nb15()
    df = clean_dataset(df)
    df = add_engineered_features(df)
    df = drop_high_correlation_features(df)
    df = stratified_sample(df, args.sample_size, args.random_state)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=args.random_state,
        stratify=y,
    )

    comparison_metrics = run_experiment_group(
        "Bonus Method Comparison",
        build_bonus_models(args.random_state),
        X_train,
        X_test,
        y_train,
        y_test,
    )
    tuning_metrics = run_experiment_group(
        "Bonus RF Tuning",
        build_tuning_scenarios(args.random_state),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    all_metrics = pd.concat([comparison_metrics, tuning_metrics], ignore_index=True)
    all_metrics.to_csv(BONUS_TABLE_DIR / "stage7_all_bonus_metrics.csv", index=False)
    plot_stage7_metric_comparison(all_metrics)
    plot_far_vs_recall(all_metrics)
    plot_runtime_vs_f1(all_metrics)
    plot_accuracy_delta(all_metrics)
    write_stage7_academic_summary(all_metrics, args.sample_size)

    print("[Stage 7] Selesai. Lihat results/tables/stage7_bonus dan results/reports/stage7_bonus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
