import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from config import FEATURE_LABELS, TARGET_ORDER


def _raw_feature_of(transformed_name):
    if transformed_name.startswith("categorical__"):
        rest = transformed_name[len("categorical__"):]
        return rest.rsplit("_", 1)[0]
    if transformed_name.startswith("numeric__"):
        return transformed_name[len("numeric__"):]
    return transformed_name


def _format_value(value):
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"
    return str(value)


def _sort_contributions(contributions, row, top_n):
    items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    results = []
    for feature, contribution in items[:top_n]:
        value = row.iloc[0][feature] if feature in row.columns else None
        results.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": _format_value(value),
                "contribution": float(contribution),
                "abs_contribution": abs(float(contribution)),
            }
        )
    total = sum(item["abs_contribution"] for item in results) or 1.0
    for item in results:
        item["share"] = item["abs_contribution"] / total
    return results


def _explain_xgboost(pipeline, row, predicted_index, top_n):
    features = pipeline.named_steps["features"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(features.transform(row))
    names = list(preprocessor.get_feature_names_out())
    matrix = xgb.DMatrix(transformed, feature_names=names)
    contribs = model.get_booster().predict(matrix, pred_contribs=True)

    if contribs.ndim == 2:
        contributions = contribs[0, :-1]
    else:
        contributions = contribs[0, predicted_index, :-1]

    aggregated = {}
    for name, value in zip(names, contributions):
        raw = _raw_feature_of(name)
        aggregated[raw] = aggregated.get(raw, 0.0) + float(value)
    return _sort_contributions(aggregated, row, top_n)


def _explain_occlusion(pipeline, row, reference_row, predicted_index, top_n):
    base = pipeline.predict_proba(row)[0][predicted_index]
    contributions = {}
    for column in row.columns:
        perturbed = reference_row.copy()
        perturbed[column] = row.iloc[0][column]
        proba = pipeline.predict_proba(perturbed)[0][predicted_index]
        contributions[column] = float(base - proba)
    return _sort_contributions(contributions, row, top_n)


def explain_prediction(
    pipeline, row, predicted_index, reference_row=None, top_n=6
):
    model = pipeline.named_steps.get("model")
    if isinstance(model, XGBClassifier):
        try:
            return _explain_xgboost(pipeline, row, predicted_index, top_n)
        except Exception:
            pass
    if reference_row is not None:
        return _explain_occlusion(
            pipeline, row, reference_row, predicted_index, top_n
        )
    return []


def predict_with_explanation(pipeline, row, reference_row=None, top_n=6):
    proba = pipeline.predict_proba(row)[0]
    predicted_index = int(np.argmax(proba))
    explanation = explain_prediction(
        pipeline, row, predicted_index, reference_row=reference_row, top_n=top_n
    )
    probabilities = [
        {
            "label": TARGET_ORDER[index],
            "probability": float(value),
            "percent": round(float(value) * 100, 1),
        }
        for index, value in enumerate(proba)
    ]
    return {
        "predicted_label": TARGET_ORDER[predicted_index],
        "predicted_index": predicted_index,
        "probabilities": probabilities,
        "explanation": explanation,
    }
