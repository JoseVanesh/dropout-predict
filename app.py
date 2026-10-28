import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from config import (  # noqa: E402
    BEST_MODEL_PATH,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    METADATA_PATH,
    METRICS_PATH,
    RAW_FEATURES,
    TARGET_ORDER,
)
from codes import CODE_LABELS  # noqa: E402
from explain import predict_with_explanation  # noqa: E402

app = Flask(__name__)

FORM_SECTIONS = [
    {
        "title": "Profil Mahasiswa",
        "icon": "01",
        "fields": [
            "Marital status",
            "Gender",
            "Age at enrollment",
            "International",
            "Displaced",
            "Educational special needs",
        ],
    },
    {
        "title": "Riwayat Pendaftaran",
        "icon": "02",
        "fields": [
            "Application mode",
            "Application order",
            "Course",
            "Daytime/evening attendance",
            "Previous qualification",
            "Nacionality",
            "Previous qualification (grade)",
            "Admission grade",
        ],
    },
    {
        "title": "Latar Belakang Keluarga",
        "icon": "03",
        "fields": [
            "Mother's qualification",
            "Father's qualification",
            "Mother's occupation",
            "Father's occupation",
        ],
    },
    {
        "title": "Status Finansial",
        "icon": "04",
        "fields": [
            "Debtor",
            "Tuition fees up to date",
            "Scholarship holder",
        ],
    },
    {
        "title": "Semester 1",
        "icon": "05",
        "fields": [
            "Curricular units 1st sem (credited)",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
            "Curricular units 1st sem (without evaluations)",
        ],
    },
    {
        "title": "Semester 2",
        "icon": "06",
        "fields": [
            "Curricular units 2nd sem (credited)",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (evaluations)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Curricular units 2nd sem (without evaluations)",
        ],
    },
    {
        "title": "Kondisi Ekonomi",
        "icon": "07",
        "fields": ["Unemployment rate", "Inflation rate", "GDP"],
    },
]

SPECIAL_SELECTS = {
    "Application order": {
        value: (
            "1 - pilihan pertama"
            if value == 0
            else f"{value + 1} - pilihan ke-{value + 1}"
        )
        for value in range(10)
    },
}

VALUE_LABELS = {
    "Gender": {1: "Laki-laki", 0: "Perempuan"},
    "Daytime/evening attendance": {1: "Pagi", 0: "Sore"},
    "Displaced": {1: "Ya", 0: "Tidak"},
    "Educational special needs": {1: "Ya", 0: "Tidak"},
    "Debtor": {1: "Ya", 0: "Tidak"},
    "Tuition fees up to date": {1: "Ya", 0: "Tidak"},
    "Scholarship holder": {1: "Ya", 0: "Tidak"},
    "International": {1: "Ya", 0: "Tidak"},
}

TARGET_STYLES = {
    "Dropout": {"class": "dropout", "title": "Berisiko Dropout"},
    "Enrolled": {"class": "enrolled", "title": "Masih Terdaftar"},
    "Graduate": {"class": "graduate", "title": "Berpotensi Lulus"},
}


def load_artifacts():
    model = joblib.load(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() else None
    metrics = None
    metadata = None
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metrics, metadata


MODEL, METRICS, METADATA = load_artifacts()


def reference_row():
    data = {}
    for name in RAW_FEATURES:
        if name in CATEGORICAL_FEATURES:
            data[name] = METADATA["categorical_defaults"][name]
        else:
            stats = METADATA["numeric_stats"][name]
            data[name] = stats["default"]
    return pd.DataFrame([data])[RAW_FEATURES]


def value_label(name, value):
    labels = CODE_LABELS.get(name) or VALUE_LABELS.get(name) or {}
    label = labels.get(value)
    return label if label is not None else f"Kode {value}"


def display_value(name, value):
    if name in SPECIAL_SELECTS:
        code = int(value)
        return SPECIAL_SELECTS[name].get(code, f"Pilihan {code + 1}")
    if name in CATEGORICAL_FEATURES:
        code = int(value)
        return f"{value_label(name, code)} (kode {code})"
    if name in BINARY_FEATURES:
        return value_label(name, int(value))
    return f"{float(value):g}"


def _coerce_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def make_field(name, current=None):
    label = FEATURE_LABELS.get(name, name)
    if name in SPECIAL_SELECTS:
        stats = METADATA["numeric_stats"][name]
        default = int(round(stats["default"]))
        options = [
            {"value": value, "label": text}
            for value, text in SPECIAL_SELECTS[name].items()
        ]
        return {
            "name": name,
            "label": label,
            "kind": "select",
            "options": options,
            "value": _coerce_int(current, default),
        }
    if name in CATEGORICAL_FEATURES:
        default = METADATA["categorical_defaults"][name]
        options = [
            {"value": value, "label": value_label(name, value)}
            for value in METADATA["categorical_options"][name]
        ]
        return {
            "name": name,
            "label": label,
            "kind": "select",
            "options": options,
            "value": _coerce_int(current, default),
        }
    if name in BINARY_FEATURES:
        stats = METADATA["numeric_stats"][name]
        default = int(round(stats["default"]))
        options = [
            {"value": 1, "label": value_label(name, 1)},
            {"value": 0, "label": value_label(name, 0)},
        ]
        return {
            "name": name,
            "label": label,
            "kind": "select",
            "options": options,
            "value": _coerce_int(current, default),
        }
    stats = METADATA["numeric_stats"][name]
    is_integer = stats["is_integer"]
    default = int(round(stats["default"])) if is_integer else round(stats["default"], 2)
    return {
        "name": name,
        "label": label,
        "kind": "number",
        "min": int(stats["min"]) if is_integer else round(stats["min"], 2),
        "max": int(stats["max"]) if is_integer else round(stats["max"], 2),
        "step": 1 if is_integer else 0.01,
        "default": default,
        "value": _coerce_float(current, default),
    }


def build_sections(values=None):
    values = values or {}
    sections = []
    for section in FORM_SECTIONS:
        fields = [make_field(name, values.get(name)) for name in section["fields"]]
        sections.append({"title": section["title"], "icon": section["icon"], "fields": fields})
    return sections


def parse_form(form):
    data = {}
    errors = []
    for name in RAW_FEATURES:
        label = FEATURE_LABELS.get(name, name)
        raw = form.get(name, "").strip()
        if raw == "":
            errors.append(f"{label} belum diisi.")
            continue
        if name in CATEGORICAL_FEATURES or name in BINARY_FEATURES:
            try:
                data[name] = int(raw)
            except ValueError:
                errors.append(f"{label} harus berupa angka kode.")
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{label} harus berupa angka.")
            continue
        stats = METADATA["numeric_stats"][name]
        if value < stats["min"] or value > stats["max"]:
            errors.append(
                f"{label} harus di antara {stats['min']:g} dan {stats['max']:g}."
            )
            continue
        data[name] = value
    return data, errors


@app.route("/")
def index():
    return render_template(
        "index.html",
        sections=build_sections(),
        model_ready=MODEL is not None,
        best_model=(METRICS or {}).get("best_model", "-"),
    )


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return render_template(
            "error.html",
            message="Model belum tersedia. Jalankan `python src/train.py` terlebih dahulu.",
        ), 503

    data, errors = parse_form(request.form)
    if errors:
        return render_template(
            "index.html",
            sections=build_sections(request.form.to_dict()),
            model_ready=True,
            best_model=(METRICS or {}).get("best_model", "-"),
            errors=errors,
        ), 400

    row = pd.DataFrame([data])[RAW_FEATURES]
    result = predict_with_explanation(
        MODEL, row, reference_row=reference_row(), top_n=6
    )
    style = TARGET_STYLES.get(result["predicted_label"], {"class": "", "title": ""})
    display_values = [
        {
            "field_label": FEATURE_LABELS.get(name, name),
            "field_value": display_value(name, data[name]),
        }
        for name in RAW_FEATURES
    ]
    return render_template(
        "result.html",
        result=result,
        style=style,
        display_values=display_values,
    )


@app.route("/model")
def model_page():
    if METRICS is None:
        return render_template(
            "error.html",
            message="Metrik belum tersedia. Jalankan `python src/train.py` terlebih dahulu.",
        ), 503
    return render_template(
        "model.html", metrics=METRICS, target_order=TARGET_ORDER
    )


@app.route("/health")
def health():
    return {"status": "ok", "model_ready": MODEL is not None}


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
