"""Eksperimen tuning Random Forest untuk prediksi churn.

Tujuan:
1. Membandingkan RF dengan semua fitur dan tanpa contract_length.
2. Menguji beberapa tingkat kompleksitas model.
3. Memilih konfigurasi berdasarkan PR-AUC validation.
4. Melaporkan performa testing asli sebagai evaluasi kandidat.

Cara menjalankan:
    python src/08_tuning_model.py

Prasyarat:
    CSV training dan testing tersedia di data/raw/.
    Paket pandas dan scikit-learn sudah terpasang.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# =========================================================
# 1. KONFIGURASI
# =========================================================

RAW = Path("data/raw")
DOCS = Path("docs")
SEED = 42

NUM = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "total_spend",
    "last_interaction",
]

KAT_SEMUA = ["gender", "subscription_type", "contract_length"]
KAT_TANPA = ["gender", "subscription_type"]

DOCS.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. MEMUAT DATA
# =========================================================

def muat(nama):
    files = sorted(RAW.glob(f"*{nama}*.csv"))

    if not files:
        raise SystemExit(
            f"File *{nama}*.csv tidak ditemukan di {RAW.resolve()}"
        )

    if len(files) > 1:
        print(f"Peringatan: beberapa file cocok untuk '{nama}':")
        for f in files:
            print(f"  - {f.name}")
        print(f"Menggunakan: {files[0].name}")

    df = pd.read_csv(files[0]).dropna(how="all")

    df.columns = [
        c.strip().lower().replace(" ", "_").replace(
            "customerid", "customer_id"
        )
        for c in df.columns
    ]

    required = set(NUM + KAT_SEMUA + ["churn"])
    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"Kolom berikut tidak ditemukan di {files[0].name}: "
            f"{sorted(missing)}"
        )

    df["churn"] = pd.to_numeric(df["churn"], errors="raise").astype(int)

    if not set(df["churn"].unique()).issubset({0, 1}):
        raise SystemExit("Kolom churn harus berisi label 0 dan 1.")

    return df.reset_index(drop=True)


train_full = muat("training")
test = muat("testing")

# Split harus konsisten dengan eksperimen 07.
tr, va = train_test_split(
    train_full,
    test_size=0.2,
    stratify=train_full["churn"],
    random_state=SEED,
    shuffle=True,
)

print("=" * 72)
print("UKURAN DATA")
print("=" * 72)
print(f"Train      : {len(tr):,}")
print(f"Validation : {len(va):,}")
print(f"Test asli  : {len(test):,}")
print(f"Churn train: {tr['churn'].mean():.1%}")
print(f"Churn valid: {va['churn'].mean():.1%}")
print(f"Churn test : {test['churn'].mean():.1%}")


# =========================================================
# 3. KONFIGURASI MODEL
# =========================================================

# max_depth dan min_samples_leaf membatasi kompleksitas pohon.
# bootstrap dan fitur lainnya memakai default sklearn.
KONFIGURASI = [
    {
        "nama": "RF semua fitur - depth 10",
        "kat": KAT_SEMUA,
        "max_depth": 10,
        "min_samples_leaf": 1,
    },
    {
        "nama": "RF tanpa CL - depth 10",
        "kat": KAT_TANPA,
        "max_depth": 10,
        "min_samples_leaf": 1,
    },
    {
        "nama": "RF tanpa CL - depth 5",
        "kat": KAT_TANPA,
        "max_depth": 5,
        "min_samples_leaf": 1,
    },
    {
        "nama": "RF tanpa CL - depth 8",
        "kat": KAT_TANPA,
        "max_depth": 8,
        "min_samples_leaf": 1,
    },
    {
        "nama": "RF tanpa CL - leaf 5",
        "kat": KAT_TANPA,
        "max_depth": 10,
        "min_samples_leaf": 5,
    },
    {
        "nama": "RF tanpa CL - leaf 10",
        "kat": KAT_TANPA,
        "max_depth": 10,
        "min_samples_leaf": 10,
    },
]


def buat_model(kat, max_depth, min_samples_leaf):
    prep = ColumnTransformer(
        [
            ("num", "passthrough", NUM),
            (
                "kat",
                OneHotEncoder(handle_unknown="ignore"),
                kat,
            ),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=SEED,
    )

    return Pipeline([
        ("prep", prep),
        ("model", rf),
    ])


def ukur(y, skor):
    pred = (skor >= 0.5).astype(int)

    return {
        "PR-AUC": average_precision_score(y, skor),
        "ROC-AUC": roc_auc_score(y, skor),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "Akurasi": accuracy_score(y, pred),
    }


# =========================================================
# 4. LATIH DAN BANDINGKAN
# =========================================================

hasil = []
model_terlatih = {}

for cfg in KONFIGURASI:
    nama = cfg["nama"]
    kat = cfg["kat"]

    print(f"\nMelatih: {nama}")

    model = buat_model(
        kat=kat,
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
    )

    # Hanya training yang dipakai untuk fitting.
    model.fit(tr[NUM + kat], tr["churn"])

    # Validation digunakan untuk pemilihan konfigurasi.
    skor_valid = model.predict_proba(va[NUM + kat])[:, 1]
    metrik_valid = ukur(va["churn"], skor_valid)

    hasil.append({
        "model": nama,
        "fitur": "tanpa contract_length" if kat == KAT_TANPA
                 else "semua fitur",
        "max_depth": cfg["max_depth"],
        "min_samples_leaf": cfg["min_samples_leaf"],
        "split": "valid",
        "n": len(va),
        **metrik_valid,
    })

    model_terlatih[nama] = model

# Urutan model ditentukan dari validation PR-AUC saja.
valid_rows = [r for r in hasil if r["split"] == "valid"]
valid_rows.sort(key=lambda r: r["PR-AUC"], reverse=True)

pilihan = valid_rows[0]["model"]

print("\n" + "=" * 72)
print("PERINGKAT MODEL BERDASARKAN VALIDATION PR-AUC")
print("=" * 72)

valid_table = pd.DataFrame(valid_rows)
print(valid_table.round(4).to_string(index=False))

print(f"\nKandidat terpilih berdasarkan validation PR-AUC: {pilihan}")

# Evaluasi test setelah kandidat ditentukan berdasarkan validation.
# Test tidak digunakan untuk mengubah urutan atau memilih kandidat.
print("\n" + "=" * 72)
print("EVALUASI TESTING ASLI")
print("=" * 72)

for cfg in KONFIGURASI:
    nama = cfg["nama"]
    kat = cfg["kat"]
    model = model_terlatih[nama]

    skor_test = model.predict_proba(test[NUM + kat])[:, 1]
    metrik_test = ukur(test["churn"], skor_test)

    hasil.append({
        "model": nama,
        "fitur": "tanpa contract_length" if kat == KAT_TANPA
                 else "semua fitur",
        "max_depth": cfg["max_depth"],
        "min_samples_leaf": cfg["min_samples_leaf"],
        "split": "test",
        "n": len(test),
        **metrik_test,
    })

    print(
        f"{nama:32s} | "
        f"PR-AUC {metrik_test['PR-AUC']:.4f} | "
        f"ROC-AUC {metrik_test['ROC-AUC']:.4f} | "
        f"Precision {metrik_test['Precision']:.4f} | "
        f"Recall {metrik_test['Recall']:.4f} | "
        f"F1 {metrik_test['F1']:.4f}"
    )


# =========================================================
# 5. SIMPAN HASIL
# =========================================================

df_hasil = pd.DataFrame(hasil)

kolom_angka = [
    "PR-AUC", "ROC-AUC", "Precision",
    "Recall", "F1", "Akurasi",
]

df_hasil[kolom_angka] = df_hasil[kolom_angka].round(4)

csv_path = DOCS / "hasil_tuning_model.csv"
txt_path = DOCS / "hasil_tuning_model.txt"

df_hasil.to_csv(csv_path, index=False)

with txt_path.open("w", encoding="utf-8") as f:
    f.write("HASIL TUNING RANDOM FOREST\n")
    f.write("=" * 72 + "\n")
    f.write(f"Train: {len(tr):,}\n")
    f.write(f"Validation: {len(va):,}\n")
    f.write(f"Test asli: {len(test):,}\n")
    f.write(f"Kandidat terpilih dari validation: {pilihan}\n\n")
    f.write(df_hasil.to_string(index=False))
    f.write("\n")

print("\n" + "=" * 72)
print("HASIL TERSIMPAN")
print(f"- {csv_path}")
print(f"- {txt_path}")
print("=" * 72)
print(
    "\nCatatan: hasil test adalah evaluasi kandidat pada testing asli. "
    "Jangan memilih ulang model berdasarkan skor test."
)