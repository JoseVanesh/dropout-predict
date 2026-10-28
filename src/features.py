import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).fillna(0.0)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Menambahkan fitur turunan dari data semester 1 dan 2."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        approved_1 = X["Curricular units 1st sem (approved)"]
        enrolled_1 = X["Curricular units 1st sem (enrolled)"]
        eval_1 = X["Curricular units 1st sem (evaluations)"]
        approved_2 = X["Curricular units 2nd sem (approved)"]
        enrolled_2 = X["Curricular units 2nd sem (enrolled)"]
        eval_2 = X["Curricular units 2nd sem (evaluations)"]

        X["approval_ratio_1st"] = _safe_ratio(approved_1, enrolled_1)
        X["approval_ratio_2nd"] = _safe_ratio(approved_2, enrolled_2)
        X["overall_approval_ratio"] = _safe_ratio(
            approved_1 + approved_2, enrolled_1 + enrolled_2
        )
        X["approved_trend"] = approved_2 - approved_1
        X["grade_trend"] = (
            X["Curricular units 2nd sem (grade)"]
            - X["Curricular units 1st sem (grade)"]
        )
        X["evaluations_per_unit_1st"] = _safe_ratio(eval_1, enrolled_1)
        X["evaluations_per_unit_2nd"] = _safe_ratio(eval_2, enrolled_2)
        X["total_approved"] = approved_1 + approved_2
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(NUMERIC_FEATURES, dtype=object)
        return np.asarray(list(input_features), dtype=object)


def build_preprocessor():
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator):
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )
