from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data.csv"
MODEL_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET = "Target"
TARGET_ORDER = ["Dropout", "Enrolled", "Graduate"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

ENGINEERED_FEATURES = [
    "approval_ratio_1st",
    "approval_ratio_2nd",
    "overall_approval_ratio",
    "approved_trend",
    "grade_trend",
    "evaluations_per_unit_1st",
    "evaluations_per_unit_2nd",
    "total_approved",
]

NUMERIC_FEATURES = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
] + ENGINEERED_FEATURES

BINARY_FEATURES = [
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

FEATURE_LABELS = {
    "Marital status": "Status pernikahan",
    "Application mode": "Metode pendaftaran",
    "Application order": "Urutan pilihan",
    "Course": "Program studi",
    "Daytime/evening attendance": "Kelas pagi/sore",
    "Previous qualification": "Pendidikan sebelumnya",
    "Previous qualification (grade)": "Nilai pendidikan sebelumnya",
    "Nacionality": "Kewarganegaraan",
    "Mother's qualification": "Pendidikan ibu",
    "Father's qualification": "Pendidikan ayah",
    "Mother's occupation": "Pekerjaan ibu",
    "Father's occupation": "Pekerjaan ayah",
    "Admission grade": "Nilai masuk",
    "Displaced": "Mahasiswa pindahan",
    "Educational special needs": "Kebutuhan khusus",
    "Debtor": "Memiliki tunggakan",
    "Tuition fees up to date": "Biaya kuliah lancar",
    "Gender": "Jenis kelamin",
    "Scholarship holder": "Penerima beasiswa",
    "Age at enrollment": "Usia saat mendaftar",
    "International": "Mahasiswa internasional",
    "Curricular units 1st sem (credited)": "SKS dikreditkan (sem 1)",
    "Curricular units 1st sem (enrolled)": "SKS diambil (sem 1)",
    "Curricular units 1st sem (evaluations)": "Jumlah evaluasi (sem 1)",
    "Curricular units 1st sem (approved)": "SKS lulus (sem 1)",
    "Curricular units 1st sem (grade)": "Rata-rata nilai (sem 1)",
    "Curricular units 1st sem (without evaluations)": "SKS tanpa evaluasi (sem 1)",
    "Curricular units 2nd sem (credited)": "SKS dikreditkan (sem 2)",
    "Curricular units 2nd sem (enrolled)": "SKS diambil (sem 2)",
    "Curricular units 2nd sem (evaluations)": "Jumlah evaluasi (sem 2)",
    "Curricular units 2nd sem (approved)": "SKS lulus (sem 2)",
    "Curricular units 2nd sem (grade)": "Rata-rata nilai (sem 2)",
    "Curricular units 2nd sem (without evaluations)": "SKS tanpa evaluasi (sem 2)",
    "Unemployment rate": "Tingkat pengangguran",
    "Inflation rate": "Tingkat inflasi",
    "GDP": "Pertumbuhan GDP",
    "approval_ratio_1st": "Rasio kelulusan (sem 1)",
    "approval_ratio_2nd": "Rasio kelulusan (sem 2)",
    "overall_approval_ratio": "Rasio kelulusan keseluruhan",
    "approved_trend": "Tren SKS lulus",
    "grade_trend": "Tren nilai",
    "evaluations_per_unit_1st": "Evaluasi per SKS (sem 1)",
    "evaluations_per_unit_2nd": "Evaluasi per SKS (sem 2)",
    "total_approved": "Total SKS lulus",
}

RAW_FEATURES = [c for c in CATEGORICAL_FEATURES] + [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]
