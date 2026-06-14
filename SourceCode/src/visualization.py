from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

from config import FIGURE_DIR


def safe_name(text: str) -> str:
    """Mengubah nama model menjadi nama file yang aman."""
    return text.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def plot_confusion_matrix(model_name: str, y_true, y_pred) -> None:
    """Menyimpan confusion matrix sebagai PNG."""
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
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"confusion_matrix_{safe_name(model_name)}.png", dpi=180)
    plt.close()


def plot_roc_curves(roc_inputs: dict, y_test) -> None:
    """Menyimpan grafik ROC untuk semua model."""
    plt.figure(figsize=(8, 6))
    for model_name, y_score in roc_inputs.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curves.png", dpi=180)
    plt.close()


def plot_metric_comparison(metrics: pd.DataFrame) -> None:
    """Menyimpan bar chart accuracy, F1, FAR, dan AUC untuk presentasi."""
    selected = metrics[["model", "accuracy", "f1", "far", "auc"]].melt(
        id_vars="model", var_name="metric", value_name="score"
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(data=selected, x="model", y="score", hue="metric")
    plt.xticks(rotation=35, ha="right")
    plt.title("Perbandingan Metrik Model")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "metric_comparison.png", dpi=180)
    plt.close()


def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 20) -> None:
    """Menyimpan grafik fitur terpenting dari XGBoost."""
    top_features = feature_importance.head(top_n).copy()
    plt.figure(figsize=(9, 7))
    sns.barplot(data=top_features, y="feature", x="importance")
    plt.title(f"Top {top_n} XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"feature_importance_top_{top_n}.png", dpi=180)
    plt.close()
