import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATA_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_DIR,
    RANDOM_STATE,
    TARGET,
    TARGET_ORDER,
    TEST_SIZE,
    RAW_FEATURES,
)
from data import load_data, split_features_target
from features import build_pipeline

warnings.filterwarnings("ignore")


def build_candidates():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_iter": [200, 400, 700],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [10, 20, 40],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__min_child_weight": [1, 3, 5],
            },
        ),
        "SVM (RBF)": (
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {"model__C": [0.5, 1.0, 5.0, 10.0], "model__gamma": ["scale", 0.01, 0.05]},
        ),
    }


def evaluate_pipeline(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    labels = list(range(len(TARGET_ORDER)))
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=TARGET_ORDER,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": TARGET_ORDER,
    }


def top_raw_features(pipeline, X_test, y_test, top_n=12):
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
        n_jobs=1,
    )
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    return [
        {
            "feature": str(X_test.columns[i]),
            "importance": float(result.importances_mean[i]),
        }
        for i in order
    ]


def build_metadata(df):
    categorical_options = {
        column: sorted(int(v) for v in df[column].unique().tolist())
        for column in CATEGORICAL_FEATURES
    }
    categorical_defaults = {
        column: int(df[column].mode().iloc[0]) for column in CATEGORICAL_FEATURES
    }
    numeric_stats = {}
    for column in RAW_FEATURES:
        if column in CATEGORICAL_FEATURES:
            continue
        series = df[column]
        numeric_stats[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "default": float(series.median()),
            "is_integer": bool(np.all(series == series.round())),
        }
    return {
        "categorical_options": categorical_options,
        "categorical_defaults": categorical_defaults,
        "numeric_stats": numeric_stats,
        "raw_features": RAW_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_order": TARGET_ORDER,
        "target_map": {label: index for index, label in enumerate(TARGET_ORDER)},
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    target_map = {label: index for index, label in enumerate(TARGET_ORDER)}
    y_encoded = y.map(target_map).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates()
    comparison = {}
    fitted = {}

    print(f"Dataset: {len(df)} baris, {X.shape[1]} fitur")
    print(f"Distribusi target: {y.value_counts().to_dict()}\n")
    print("=== Perbandingan model (5-fold CV, macro-F1) ===")

    for name, (estimator, _) in candidates.items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
        )
        comparison[name] = {
            "mean_macro_f1": float(scores.mean()),
            "std_macro_f1": float(scores.std()),
        }
        fitted[name] = pipeline
        print(f"{name:24s} {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(comparison, key=lambda n: comparison[n]["mean_macro_f1"])
    print(f"\nModel terbaik (CV): {best_name}")

    print("\n=== Hyperparameter tuning ===")
    best_pipeline = fitted[best_name]
    param_grid = candidates[best_name][1]
    search = RandomizedSearchCV(
        best_pipeline,
        param_distributions=param_grid,
        n_iter=16,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"Best CV macro-F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    print("\n=== Evaluasi pada test set (20%) ===")
    test_metrics = evaluate_pipeline(search.best_estimator_, X_test, y_test)
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Macro-F1 : {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {test_metrics['weighted_f1']:.4f}")

    print("\n=== Feature importance (permutation) ===")
    top_features = top_raw_features(search.best_estimator_, X_test, y_test)
    for item in top_features[:8]:
        print(f"{item['feature']:45s} {item['importance']:.4f}")

    final_model = search.best_estimator_.fit(X, y_encoded)
    joblib.dump(final_model, BEST_MODEL_PATH)

    report = test_metrics.pop("classification_report")
    test_metrics["per_class_recall"] = {
        label: float(report[label]["recall"])
        for label in TARGET_ORDER
        if label in report
    }
    test_metrics["per_class_f1"] = {
        label: float(report[label]["f1-score"])
        for label in TARGET_ORDER
        if label in report
    }

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "features_raw": int(X.shape[1]),
            "features_total": int(final_model[:-1].transform(X.head(5)).shape[1]),
            "target_counts": {k: int(v) for k, v in y.value_counts().items()},
        },
        "cv_comparison": comparison,
        "best_model": best_name,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_macro_f1": float(search.best_score_),
        "test_metrics": test_metrics,
        "top_features": top_features,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(build_metadata(df), indent=2), encoding="utf-8"
    )

    print(f"\nModel disimpan  : {BEST_MODEL_PATH}")
    print(f"Metrik disimpan : {METRICS_PATH}")
    print(f"Metadata        : {METADATA_PATH}")


if __name__ == "__main__":
    main()
