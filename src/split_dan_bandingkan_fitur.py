"""
Skema split + perbandingan dua kelompok fitur.

Skema (sesuai Bab 6.1.3 laporan):
  training asli  -> train_fit (80%) + valid (20%, stratified, diacak)
  testing asli   -> dipakai UTUH, hanya untuk evaluasi akhir

Model A : 10 fitur asli (model utama)
Model B : 10 fitur asli + fitur aturan (model diagnostik)
          Fitur aturan ditemukan dari data train_fit SAJA.

Ambang keputusan dipilih di data valid (recall >= 70%), lalu diterapkan
apa adanya di data testing. Testing tidak ikut memilih apa pun.

Cara pakai (dari folder utama repo, venv aktif):
    python src/split_dan_bandingkan_fitur.py

Prasyarat: file CSV ada di data/raw/ ; paket: pandas, scikit-learn, pyarrow.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------- Pengaturan -----------------------------
RAW = Path("data/raw")
OUT = Path("data/model_ready")
DOCS = Path("docs")
FILE_TRAIN = RAW / "customer_churn_dataset-training-master.csv"
FILE_TEST = RAW / "customer_churn_dataset-testing-master.csv"
SEED = 42
TARGET_RECALL = 0.70

NUM = ["age", "tenure", "usage_frequency", "support_calls",
       "payment_delay", "total_spend", "last_interaction"]
KAT = ["gender", "subscription_type", "contract_length"]
FLAG_5 = ["is_monthly", "age_hi_pure", "support_calls_hi_pure",
          "payment_delay_hi_pure", "total_spend_lo_pure"]
FLAG_ALL = FLAG_5 + ["rule_any"]
KOLOM_BULAT = ["customer_id", "age", "tenure", "usage_frequency",
               "support_calls", "payment_delay", "last_interaction", "churn"]


# ----------------------------- Muat data ------------------------------
def muat(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"CustomerID": "customer_id"})
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)  # buang baris kosong total
    df[KOLOM_BULAT] = df[KOLOM_BULAT].astype(int)     # total_spend tetap desimal
    return df


# ------------------------- Fitur aturan (Model B) ----------------------
def cari_ambang(train: pd.DataFrame) -> dict:
    """Ekor 'murni churn' dicari hanya dari train_fit: nilai di luar rentang
    pelanggan tidak-churn pasti churn pada data latih."""
    nc = train[train["churn"] == 0]
    return {
        "age_max_nc": nc["age"].max(),
        "support_calls_max_nc": nc["support_calls"].max(),
        "payment_delay_max_nc": nc["payment_delay"].max(),
        "total_spend_min_nc": nc["total_spend"].min(),
    }


def tambah_fitur_aturan(df: pd.DataFrame, a: dict) -> pd.DataFrame:
    out = df.copy()
    out["is_monthly"] = (out["contract_length"] == "Monthly").astype(int)
    out["age_hi_pure"] = (out["age"] > a["age_max_nc"]).astype(int)
    out["support_calls_hi_pure"] = (out["support_calls"] > a["support_calls_max_nc"]).astype(int)
    out["payment_delay_hi_pure"] = (out["payment_delay"] > a["payment_delay_max_nc"]).astype(int)
    out["total_spend_lo_pure"] = (out["total_spend"] < a["total_spend_min_nc"]).astype(int)
    out["rule_any"] = out[FLAG_5].max(axis=1)
    return out


# ------------------------------ Model ---------------------------------
def buat_pipeline(nama: str, dengan_aturan: bool) -> Pipeline:
    langkah = [
        ("num", StandardScaler(), NUM),
        ("kat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), KAT),
    ]
    if dengan_aturan:
        langkah.append(("flag", "passthrough", FLAG_ALL))
    if nama == "LR":
        model = LogisticRegression(max_iter=1000, random_state=SEED)
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, n_jobs=-1, random_state=SEED
        )
    return Pipeline([("prep", ColumnTransformer(langkah)), ("model", model)])


def ambang_recall(y, proba, target=TARGET_RECALL) -> float:
    """Ambang tertinggi yang masih mencapai recall >= target (di data valid)."""
    _, r, t = precision_recall_curve(y, proba)
    ok = r[:-1] >= target
    return float(t[ok].max()) if ok.any() else 0.5


def ukur(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "ambang": thr,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def rincian(df, y, pred, nama_model, fitur, kolom):
    baris = []
    for nilai, g in df.groupby(kolom):
        idx = g.index
        baris.append({
            "model": nama_model, "fitur": fitur, "kelompok": kolom, "nilai": nilai,
            "n": len(g), "churn_rate": y.loc[idx].mean(),
            "precision": precision_score(y.loc[idx], pred[idx], zero_division=0),
            "recall": recall_score(y.loc[idx], pred[idx], zero_division=0),
        })
    return baris


# ------------------------------ Utama ---------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    train_full = muat(FILE_TRAIN)
    test = muat(FILE_TEST)

    # 1) Split: acak + stratified. Wajib diacak karena urutan baris training
    #    berkorelasi dengan label (Bab 4.1 poin 2).
    train_fit, valid = train_test_split(
        train_full, test_size=0.2, stratify=train_full["churn"],
        random_state=SEED, shuffle=True,
    )
    train_fit = train_fit.reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    print("== Ukuran data ==")
    for nama, d in [("train_fit", train_fit), ("valid", valid), ("test (asli, utuh)", test)]:
        print(f"{nama:18s}: {len(d):>8,} baris | churn {d['churn'].mean():.2%}")

    # 2) Fitur aturan: ambang HANYA dari train_fit
    ambang = cari_ambang(train_fit)
    print("\n== Ambang 'murni churn' (dari train_fit) ==")
    for k, v in ambang.items():
        print(f"  {k}: {v}")
    tf, va, te = (tambah_fitur_aturan(d, ambang) for d in (train_fit, valid, test))

    # Simpan hasil split (belum di-encode) supaya semua anggota memakai split yang sama
    tf.to_parquet(OUT / "split_train_fit.parquet", index=False)
    va.to_parquet(OUT / "split_valid.parquet", index=False)
    te.to_parquet(OUT / "split_test.parquet", index=False)

    print("\n== Churn rate per bendera aturan (kalau 100% di semua split = aturan berlaku; "
          "kalau turun di test = aturan tidak ikut berlaku) ==")
    ringkas = []
    for f in FLAG_ALL:
        ringkas.append({
            "bendera": f,
            "train_fit": tf.loc[tf[f] == 1, "churn"].mean(),
            "valid": va.loc[va[f] == 1, "churn"].mean(),
            "test": te.loc[te[f] == 1, "churn"].mean(),
            "n_test": int((te[f] == 1).sum()),
        })
    print(pd.DataFrame(ringkas).round(4).to_string(index=False))

    # 3) Latih & ukur: 2 model x 2 kelompok fitur
    fitur_a = NUM + KAT
    fitur_b = NUM + KAT + FLAG_ALL
    hasil, rinci = [], []
    for nama_model in ["LR", "RF"]:
        for kel, fitur, dgn_aturan in [("A (10 fitur asli)", fitur_a, False),
                                       ("B (+ fitur aturan)", fitur_b, True)]:
            pipe = buat_pipeline(nama_model, dgn_aturan)
            pipe.fit(tf[fitur], tf["churn"])                    # fit HANYA di train_fit
            p_val = pipe.predict_proba(va[fitur])[:, 1]
            p_te = pipe.predict_proba(te[fitur])[:, 1]
            thr = ambang_recall(va["churn"], p_val)             # ambang dipilih di valid
            for nama_data, y, p in [("valid", va["churn"], p_val), ("test", te["churn"], p_te)]:
                hasil.append({"model": nama_model, "fitur": kel, "data": nama_data,
                              **ukur(y, p, thr)})
            pred_te = (p_te >= thr).astype(int)
            for kolom in ["contract_length", "gender"]:
                rinci += rincian(te, te["churn"], pred_te, nama_model, kel, kolom)
            print(f"selesai: {nama_model} | {kel}")

    # Pembanding sederhana: menebak semua churn
    for nama_data, d in [("valid", va), ("test", te)]:
        hasil.append({"model": "Tebak semua churn", "fitur": "-", "data": nama_data,
                      "roc_auc": 0.5, "pr_auc": d["churn"].mean(), "ambang": 0.0,
                      "precision": d["churn"].mean(), "recall": 1.0,
                      "f1": 2 * d["churn"].mean() / (1 + d["churn"].mean())})

    tabel = pd.DataFrame(hasil)
    tabel.to_csv(DOCS / "perbandingan_fitur.csv", index=False)
    pd.DataFrame(rinci).to_csv(DOCS / "rincian_test.csv", index=False)

    print("\n== Hasil lengkap (ambang dipilih di valid agar recall >= 70%) ==")
    print(tabel.round(3).to_string(index=False))

    # 4) Ringkasan: seberapa jauh skor turun dari valid ke test
    lebar = tabel[tabel["model"] != "Tebak semua churn"].pivot_table(
        index=["model", "fitur"], columns="data", values=["pr_auc", "precision", "recall"])
    ringkas = pd.DataFrame({
        "PR-AUC valid": lebar[("pr_auc", "valid")],
        "PR-AUC test": lebar[("pr_auc", "test")],
        "selisih": lebar[("pr_auc", "valid")] - lebar[("pr_auc", "test")],
        "precision test": lebar[("precision", "test")],
        "recall test": lebar[("recall", "test")],
    })
    print("\n== Ringkasan: seberapa jauh skor turun dari valid ke test ==")
    print(ringkas.round(3).to_string())
    print(f"\nPembanding di test (tebak semua churn): precision {te['churn'].mean():.3f}, recall 1.000")
    print("\nTersimpan: docs/perbandingan_fitur.csv, docs/rincian_test.csv, "
          "data/model_ready/split_*.parquet")


if __name__ == "__main__":
    main()