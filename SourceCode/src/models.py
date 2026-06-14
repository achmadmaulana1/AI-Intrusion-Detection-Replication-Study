from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def build_models(random_state: int, include_svm: bool = True) -> dict:
    """Mendefinisikan model baseline sesuai paper utama.

    Linear SVM dibuat dengan SGDClassifier agar tetap realistis untuk laptop
    tanpa GPU dan dataset besar.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="saga",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=6,
            min_samples_leaf=9,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=22,
            min_samples_split=6,
            criterion="gini",
            n_jobs=-1,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=12,
            learning_rate=0.1,
            colsample_bylevel=0.5,
            subsample=0.1,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if include_svm:
        models["Linear SVM"] = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
            n_jobs=-1,
        )

    return models


def build_selected_feature_models(random_state: int) -> dict:
    """Model yang dilatih ulang setelah 55 fitur terbaik dipilih."""
    return {
        "Decision Tree Selected Features": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=6,
            min_samples_leaf=9,
            random_state=random_state,
        ),
        "Random Forest Selected Features": RandomForestClassifier(
            n_estimators=300,
            max_depth=22,
            min_samples_split=6,
            criterion="gini",
            n_jobs=-1,
            random_state=random_state,
        ),
    }
