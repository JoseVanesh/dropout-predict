# Dropout Predict

Sistem machine learning untuk memprediksi status akademik mahasiswa ke dalam tiga kelas: **Dropout**, **Enrolled**, dan **Graduate**. Model menampilkan probabilitas tiap kelas sekaligus faktor yang paling memengaruhi prediksi, sehingga hasilnya bisa ditindaklanjuti — bukan sekadar angka.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.x-189FDD)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20697-0B5FFF)
![Status](https://img.shields.io/badge/status-ready-067647)

---

## Ringkasan

| Item | Detail |
|---|---|
| Tugas | Klasifikasi multiclass (3 kelas) |
| Dataset | UCI *Predict Students' Dropout and Academic Success* (4.424 baris, 36 fitur) |
| Model terpilih | **XGBoost** (hyperparameter tuning) |
| Test Accuracy | **75,7%** |
| Test Macro-F1 | **0,682** |
| Recall Dropout | **73,2%** |
| Web app | Flask + Jinja2, form 36 fitur, penjelasan TreeSHAP |

## Fitur

- **Prediksi 3 kelas** — Dropout, Enrolled, Graduate beserta probabilitasnya.
- **Penjelasan prediksi** — kontribusi tiap fitur (TreeSHAP) untuk memprediksi satu mahasiswa.
- **Form lengkap 36 fitur** — dengan label yang mudah dibaca, hasil pemetaan kode resmi UCI.
- **Perbandingan model** — Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, SVM.
- **Evaluasi transparan** — cross-validation, confusion matrix, metrik per kelas, permutation importance.
- **Pipeline anti-leakage** — preprocessing dibungkus `Pipeline`, di-fit hanya pada data training.
- **REST-ready** — endpoint `/health` dan struktur Flask yang mudah dikembangkan.

## Hasil Model

Perbandingan 5-fold stratified cross-validation pada data training:

| Model | Macro-F1 (CV) | Std. Dev |
|---|---:|---:|
| Logistic Regression | 0,7100 | ±0,0148 |
| Random Forest | 0,6935 | ±0,0213 |
| HistGradientBoosting | 0,7140 | ±0,0143 |
| **XGBoost** | **0,7264** | **±0,0097** |
| SVM (RBF) | 0,7212 | ±0,0135 |

Evaluasi akhir pada 20% test set:

| Metrik | Nilai |
|---|---:|
| Accuracy | 0,757 |
| Macro-F1 | 0,682 |
| Weighted-F1 | 0,748 |

| Kelas | Recall | F1 |
|---|---:|---:|
| Dropout | 0,732 | 0,752 |
| Enrolled | 0,390 | 0,435 |
| Graduate | 0,905 | 0,858 |

> **Catatan:** kelas `Enrolled` paling sulit diprediksi karena porsinya hanya ±18% dan berada di antara dua kelas lainnya. Metrik utama yang dipakai adalah **macro-F1** dan **recall Dropout**, bukan accuracy semata.

## Cara Kerja

```
data.csv
   │
   ├─ Feature engineering
   │    approval ratio, tren SKS lulus, tren nilai, evaluasi per SKS
   │
   ├─ Preprocessing (ColumnTransformer)
   │    One-Hot Encoding kategorikal  +  StandardScaler numerik
   │
   ├─ Model
   │    XGBoost multiclass (multi:softprob)
   │
   ├─ Evaluasi
   │    Stratified 5-Fold CV → RandomizedSearchCV → test 20%
   │
   └─ Flask
        form input → probabilitas 3 kelas → TreeSHAP top factors
```

Label target di-encode dengan urutan tetap:

```
Dropout = 0, Enrolled = 1, Graduate = 2
```

## Struktur Proyek

```
DroppOut Predict/
├── data.csv                  # dataset UCI (4.424 baris)
├── requirements.txt
├── app.py                    # Flask web app
├── src/
│   ├── config.py             # konstanta, daftar fitur, path
│   ├── codes.py              # kode UCI → label Indonesia
│   ├── data.py               # load & split data
│   ├── features.py           # feature engineering + preprocessor
│   ├── train.py              # training, tuning, evaluasi
│   └── explain.py            # TreeSHAP + fallback occlusion
├── models/
│   ├── best_model.joblib     # model terlatih
│   ├── metrics.json          # hasil evaluasi
│   └── metadata.json         # opsi form, rentang nilai, mapping target
├── templates/                # base, index, result, model, error
├── static/style.css
├── notebooks/01_eda.ipynb    # exploratory data analysis
└── README.md
```

## Instalasi

Prasyarat: **Python 3.10+**

```powershell
git clone https://github.com/JoseVanesh/dropout-predict.git
cd dropout-predict

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```powershell
python app.py
```

Buka **http://127.0.0.1:5000**

| Halaman | Fungsi |
|---|---|
| `/` | Form prediksi 36 fitur |
| `/predict` | Hasil prediksi, probabilitas, dan faktor berpengaruh |
| `/model` | Performa model, confusion matrix, feature importance |
| `/health` | Status aplikasi (JSON) |

Contoh cek status:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/health
```

## Training Ulang

```powershell
python src/train.py
```

Proses ini akan:

1. Membandingkan 5 model dengan Stratified 5-Fold CV (macro-F1).
2. Melakukan `RandomizedSearchCV` pada model terbaik.
3. Mengevaluasi pada test set 20%.
4. Menghitung permutation importance.
5. Menyimpan `best_model.joblib`, `metrics.json`, dan `metadata.json` ke folder `models/`.

## Dataset

**Predict Students' Dropout and Academic Success** — UCI Machine Learning Repository (ID 697).

- 4.424 mahasiswa, 36 fitur, 3 kelas.
- Tidak ada missing value maupun duplikat.
- Fitur mencakup data demografi, jalur pendaftaran, latar belakang orang tua, status finansial, capaian kurikulum semester 1–2, dan indikator ekonomi.
- Lisensi dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Sitasi:

```bibtex
@dataset{realinho_2021_697,
  author    = {Realinho, Valentim and Vieira Martins, Mónica and Machado, Jorge and Baptista, Luís},
  title     = {Predict Students' Dropout and Academic Success},
  year      = {2021},
  publisher = {UCI Machine Learning Repository},
  doi       = {10.24432/C5MC89}
}
```

## Batasan

- Fitur semester 2 membuat model cocok untuk **evaluasi akhir tahun**, bukan peringatan dini di tengah semester. Untuk early warning murni, gunakan hanya fitur semester 1.
- Performa kelas `Enrolled` masih rendah; dapat ditingkatkan dengan penyesuaian class weight, SMOTE, atau model khusus biner.
- Dataset berasal dari satu institusi di Portugal, sehingga generalisasi ke institusi lain perlu validasi ulang.
- Penjelasan TreeSHAP menunjukkan kontribusi fitur terhadap prediksi model, bukan hubungan kausal.

## Roadmap

- [ ] Varian model early warning (hanya data semester 1).
- [ ] Eksperimen SMOTE / focal loss untuk kelas `Enrolled`.
- [ ] Export model ke ONNX dan REST API terpisah.
- [ ] Docker compose untuk deployment.
- [ ] Unit test untuk preprocessing dan endpoint prediksi.

## Lisensi

Kode proyek ini bebas digunakan untuk pembelajaran. Dataset tetap mengikuti lisensi **CC BY 4.0** dari UCI Machine Learning Repository.

---

Dibuat dengan Python, scikit-learn, XGBoost, dan Flask.
