"""Eksperimen: seberapa besar model bergantung pada kolom Contract Length?

Temuan awal: di training semua pelanggan kontrak Monthly churn (100%), di testing hanya ~51%.
Tiga pembanding, masing-masing dilaporkan di TRAIN, VALID, dan TEST (skenario 1, lintas-file):
  1. Model memakai semua fitur (10 fitur)
  2. Model TANPA contract_length (9 fitur)
  3. Aturan sederhana: pelanggan Monthly dianggap churn, selain itu tidak churn
  + pembanding: tebak semua churn

Cara pakai (dari folder utama repo, venv aktif):
    python src/07_bandingkan_contract.py

Prasyarat: CSV di data/raw/; paket pandas, scikit-learn.
Keluaran : ringkasan di terminal + docs/perbandingan_contract.csv + docs/perbandingan_contract.txt

Catatan: metrik pada baris 'train' bersifat in-sample. Ambang model tetap 0,5 agar adil
terhadap aturan (yang hanya punya keputusan 0/1). PR-AUC dan ROC-AUC memakai skor probabilitas
(untuk aturan, skornya 0/1).
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW = Path("data/raw")
SEED = 42
NUM = ["age", "tenure", "usage_frequency", "support_calls",
       "payment_delay", "total_spend", "last_interaction"]
KAT_SEMUA = ["gender", "subscription_type", "contract_length"]
KAT_TANPA = ["gender", "subscription_type"]
lines = []


def out(t=""):
    print(t)
    lines.append(t)


def muat(nama):
    f = sorted(RAW.glob(f"*{nama}*.csv"))
    if not f:
        raise SystemExit(f"File *{nama}*.csv tidak ada di data/raw")
    df = pd.read_csv(f[0]).dropna(how="all")
    df.columns = [c.strip().lower().replace(" ", "_").replace("customerid", "customer_id") for c in df.columns]
    df["churn"] = df["churn"].astype(int)
    return df.reset_index(drop=True)


def pipeline(model, kat):
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("kat", OneHotEncoder(handle_unknown="ignore"), kat)])
    clf = (LogisticRegression(max_iter=1000, random_state=SEED) if model == "LR"
           else RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=SEED))
    return Pipeline([("prep", pre), ("model", clf)])


def ukur(y, skor, pred):
    return {
        "n": len(y), "churn_%": round(y.mean() * 100, 1),
        "PR-AUC": round(average_precision_score(y, skor), 3),
        "ROC-AUC": round(roc_auc_score(y, skor), 3),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "recall": round(recall_score(y, pred, zero_division=0), 3),
        "F1": round(f1_score(y, pred, zero_division=0), 3),
        "akurasi": round(accuracy_score(y, pred), 3),
    }


train_full, test = muat("training"), muat("testing")
tr, va = train_test_split(train_full, test_size=0.2, stratify=train_full["churn"],
                          random_state=SEED, shuffle=True)
bagian = {"train": tr.reset_index(drop=True), "valid": va.reset_index(drop=True), "test": test}
out(f"Ukuran: train {len(tr):,} | valid {len(va):,} | test {len(test):,} (testing asli, utuh)")
out(f"Kontrak Monthly -> churn: train {tr.loc[tr.contract_length == 'Monthly', 'churn'].mean():.1%} | "
    f"valid {va.loc[va.contract_length == 'Monthly', 'churn'].mean():.1%} | "
    f"test {test.loc[test.contract_length == 'Monthly', 'churn'].mean():.1%}")
out()

hasil, model_rf = [], {}
for nama, kat, kode in [("Model semua fitur", KAT_SEMUA, "all"), ("Model tanpa contract_length", KAT_TANPA, "nocl")]:
    for m in ("LR", "RF"):
        pipe = pipeline(m, kat).fit(tr[NUM + kat], tr["churn"])
        if m == "RF":
            model_rf[kode] = (pipe, kat)
        for bag, d in bagian.items():
            p = pipe.predict_proba(d[NUM + kat])[:, 1]
            hasil.append({"pembanding": f"{nama} ({m})", "data": bag, **ukur(d["churn"], p, (p >= 0.5).astype(int))})

for bag, d in bagian.items():
    mono = (d["contract_length"] == "Monthly").astype(int)
    hasil.append({"pembanding": "Aturan: Monthly = churn", "data": bag, **ukur(d["churn"], mono, mono)})
    ones = pd.Series(1, index=d.index)
    hasil.append({"pembanding": "Tebak semua churn", "data": bag,
                  **ukur(d["churn"], ones * 0.0 + d["churn"].mean(), ones)})

tab = pd.DataFrame(hasil)
order = {"train": 0, "valid": 1, "test": 2}
tab["_o"] = tab["data"].map(order)
tab = tab.sort_values(["pembanding", "_o"]).drop(columns="_o")
tab.to_csv("docs/perbandingan_contract.csv", index=False)

for bag in ("train", "valid", "test"):
    out(f"== {bag.upper()} {'(in-sample)' if bag == 'train' else ''} ==")
    out(tab[tab["data"] == bag].drop(columns="data").to_string(index=False))
    out()

out("== Ringkasan PR-AUC dan F1: seberapa turun dari valid ke test ==")
r = tab.pivot_table(index="pembanding", columns="data", values=["PR-AUC", "F1"])
ring = pd.DataFrame({
    "PR-AUC valid": r[("PR-AUC", "valid")], "PR-AUC test": r[("PR-AUC", "test")],
    "selisih": (r[("PR-AUC", "valid")] - r[("PR-AUC", "test")]).round(3),
    "F1 valid": r[("F1", "valid")], "F1 test": r[("F1", "test")]})
out(ring.round(3).to_string())
out()

out("== Di test: RF menurut jenis kontrak (ambang 0,5) ==")
rows = []
for kode, label in (("all", "RF semua fitur"), ("nocl", "RF tanpa contract_length")):
    pipe, kat = model_rf[kode]
    p = pipe.predict_proba(test[NUM + kat])[:, 1]
    for k, g in test.assign(p=p).groupby("contract_length"):
        pred = (g["p"] >= 0.5).astype(int)
        rows.append({"model": label, "kontrak": k, "n": len(g), "churn_%": round(g["churn"].mean() * 100, 1),
                     "ditandai_%": round(pred.mean() * 100, 1),
                     "precision": round(precision_score(g["churn"], pred, zero_division=0), 3),
                     "recall": round(recall_score(g["churn"], pred, zero_division=0), 3)})
out(pd.DataFrame(rows).to_string(index=False))

Path("docs/perbandingan_contract.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nTersimpan: docs/perbandingan_contract.csv, docs/perbandingan_contract.txt")