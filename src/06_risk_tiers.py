"""Tahap 7-8 (PySpark): tiga kelas risiko churn BERBASIS ATURAN + model, simpan skor untuk dashboard.

Kelas: "Pasti churn", "Akan churn", "Pasti tidak churn".

Jalankan dari root proyek (venv aktif), setelah 04_feature_engineering.py berhasil:
    python src/04_risk_tiers.py

Masukan : data/model_ready/churn_features.parquet (kolom rule_any dari rekayasa fitur)
Keluaran: docs/tier_results.txt, data/scored/scored_customers.parquet,
          data/scored/tier_summary.csv

Cara kerja:
1. "Pasti churn"  : rule_any = 1 (kontrak Monthly atau salah satu ambang ekor murni dari
                    data train). Aturannya transparan dan diverifikasi di valid dan test.
2. Pelanggan di luar aturan (rule_any = 0) dinilai model. Model dilatih HANYA pada baris
   train yang rule_any = 0, karena di situlah ketidakpastian yang sebenarnya.
3. Di kelompok itu: skor <= ambang bawah -> "Pasti tidak churn" (NPV >= NPV_TARGET di
   validasi); selebihnya -> "Akan churn" (zona risiko yang perlu dipantau).
4. Data test dipakai SEKALI untuk menilai hasil akhir (dengan selang kepercayaan 95%).
5. Uji bias per gender dan kelompok usia dicetak untuk lembar kepatuhan dan etika.

Catatan: "pasti" berarti keyakinan tinggi yang terukur, bukan jaminan. Setiap kelas
dilaporkan dengan presisi/NPV dan selang kepercayaannya.
"""
import os
from pathlib import Path

import pandas as pd
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

from churn_common import (
    CAT,
    ID_COL,
    LABEL,
    MODEL_NAMES,
    NUM,
    SCORED,
    TIER_SUMMARY,
    add_score,
    build_pipeline,
    get_spark,
    load_split,
    use_features,
)
from tier_utils import (
    TIER_ORDER,
    deciles,
    high_stats,
    low_stats,
    make_tier_fn,
    pick_high,
    pick_low,
    tier_rows,
)

NPV_TARGET = float(os.environ.get("NPV_TARGET", "0.97"))  # kelas "Pasti tidak churn"
MIN_TIER = 500            # minimal pelanggan di kelas "pasti" agar ambang stabil
RULE_SCORE = 2.0          # skor buatan di atas semua skor model: penanda baris berstatus aturan

lines = []


def out(text: str = "") -> None:
    print(text)
    lines.append(text)


spark = get_spark("churn-risk-tiers")
if not use_features():
    raise SystemExit("data/model_ready/churn_features.parquet belum ada. "
                     "Jalankan python src/04_feature_engineering.py dulu.")
df = load_split(spark)
train_all = df.filter(F.col("split") == "train")
train = train_all.filter(F.col("rule_any") == 0).cache()      # model belajar di luar aturan
valid = df.filter(F.col("split") == "valid").cache()
test = df.filter(F.col("split") == "test").cache()
valid_nr = valid.filter(F.col("rule_any") == 0)               # untuk pilih model & ambang

n_train, n_valid, n_test = train.count(), valid.count(), test.count()
out(f"Train (di luar aturan) {n_train} | Valid {n_valid} | Test {n_test} [test dikunci sampai evaluasi akhir]")
out()

ev_pr = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
)
ev_roc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

# ---------- 1. Pilih model ----------
out("== Pemilihan model (validasi, hanya pelanggan di luar aturan) ==")
fitted, scores = {}, {}
for name in MODEL_NAMES:
    model = build_pipeline(name).fit(train)
    pv = model.transform(valid_nr)
    fitted[name] = model
    scores[name] = (ev_pr.evaluate(pv), ev_roc.evaluate(pv))
    out(f"{name:20s} PR-AUC {scores[name][0]:.4f} | ROC-AUC {scores[name][1]:.4f}")
best = max(MODEL_NAMES, key=lambda n: scores[n][0])
model = fitted[best]
out(f"Model terpilih: {best}")
out()

# ---------- 2. Ambang dari data validasi (hanya di luar aturan) ----------
valid_scored = add_score(model.transform(valid)).select("p", "label", "rule_any")
vrows = valid_scored.collect()
vdata_nr = [(r["p"], r["label"]) for r in vrows if r["rule_any"] == 0]
vdata = [(RULE_SCORE if r["rule_any"] == 1 else r["p"], r["label"]) for r in vrows]
prev_nr = sum(y for _, y in vdata_nr) / len(vdata_nr)
out(f"Validasi: {len(vdata)} pelanggan, {len(vdata) - len(vdata_nr)} terkena aturan, "
    f"{len(vdata_nr)} di luar aturan")
out(f"Churn di luar aturan: {prev_nr:.3f} (NPV tanpa model = {1 - prev_nr:.3f}; "
    f"model harus menaikkannya)")
out()
out("== Opsi target 'Pasti tidak churn' (validasi, di luar aturan) ==")
out("target NPV | ambang | pelanggan | NPV | churner yang lolos")
for nt in (0.95, 0.97, 0.99):
    t = pick_low(vdata_nr, nt, MIN_TIER)
    if t is None:
        out(f"{nt:>9.0%} | tidak tercapai dengan minimal {MIN_TIER} pelanggan")
    else:
        n, share, npv, fn = low_stats(vdata_nr, t)
        out(f"{nt:>9.0%} | {t:6.3f} | {n:9d} | {npv:.3f} | {fn}")
out()

t_hi = RULE_SCORE - 0.5   # "Pasti churn" = baris berstatus aturan
t_lo = pick_low(vdata_nr, NPV_TARGET, MIN_TIER)
if t_lo is None:
    out(f"PERINGATAN: target NPV {NPV_TARGET:.0%} tidak tercapai. Kelas 'Pasti tidak churn' dikosongkan.")
    t_lo = -0.01
out(f"Ambang dipakai: NPV target {NPV_TARGET:.0%}, ambang bawah skor {t_lo:.3f}")
out()
tier_of = make_tier_fn(t_hi, t_lo)


def print_tiers(rows, title, with_ci=False):
    out(title)
    for r in rows:
        text = (
            f"{r['tier']:18s} n={r['n']:6d} ({r['share']:5.1%}) | churn aktual {r['churners']:6d} = "
            f"{r['churn_rate']:6.1%} | porsi seluruh churner {r['share_of_churners']:6.1%}"
        )
        if with_ci and r["n"]:
            text += f" | 95% CI [{r['ci_lo']:.1%} - {r['ci_hi']:.1%}]"
        out(text)
    out()


print_tiers(tier_rows(vdata, tier_of), "== Kelas risiko pada data validasi (dipakai memilih ambang) ==")

out("== Skor vs churn aktual per desil (validasi, di luar aturan) ==")
for d, mp, my in deciles(vdata_nr):
    out(f"desil {d:2d}: rata-rata skor {mp:.3f} | churn aktual {my:.3f}")
out()

# ---------- 3. Evaluasi akhir di data test (sekali) ----------
test_pred = add_score(model.transform(test)).cache()
tdata = [
    (RULE_SCORE if r["rule_any"] == 1 else r["p"], r["label"])
    for r in test_pred.select("p", "label", "rule_any").collect()
]
trows = tier_rows(tdata, tier_of)
print_tiers(trows, f"== EVALUASI AKHIR di data test ({best}) ==", with_ci=True)

test_nr = test_pred.filter(F.col("rule_any") == 0)
brier = test_nr.select(F.avg((F.col("p") - F.col("label")) ** 2)).first()[0]
out(
    f"Model di luar aturan (test): PR-AUC {ev_pr.evaluate(test_nr):.4f} "
    f"(pembanding acak = prevalensi {test_nr.filter('label = 1').count() / test_nr.count():.3f}) | "
    f"ROC-AUC {ev_roc.evaluate(test_nr):.4f} | Brier {brier:.4f}"
)
test_pos = sum(1 for _, y in tdata if y == 1.0)
flagged_pos = sum(1 for p, y in tdata if y == 1.0 and tier_of(p) != "Pasti tidak churn")
out(
    f"Churner yang masuk 'Pasti churn' + 'Akan churn': {flagged_pos} dari {test_pos} "
    f"({flagged_pos / test_pos:.1%}); target slide >= 70%: "
    f"{'tercapai' if flagged_pos / test_pos >= 0.70 else 'BELUM tercapai'}"
)
rates_by_tier = [r["churn_rate"] for r in trows]
mono = rates_by_tier[0] >= rates_by_tier[1] >= rates_by_tier[2]
out(f"Tingkat churn aktual naik konsisten antarkelas: {'ya' if mono else 'TIDAK'}")
out()

# ---------- 4. Uji bias (untuk lembar kepatuhan & etika) ----------
tp = test_pred.select("gender", "age", "label", "p", "rule_any").toPandas()
tp["p"] = tp["p"].where(tp["rule_any"] == 0, RULE_SCORE)
tp["flag"] = tp["p"].map(tier_of) != "Pasti tidak churn"
tp["age_group"] = pd.cut(tp["age"], [0, 29, 44, 200], labels=["<30", "30-44", "45+"])


def fairness(col):
    out(f"== Uji bias per {col} (data test; ditandai = bukan 'Pasti tidak churn') ==")
    out(f"{col:<10} {'n':>7} {'churn aktual':>13} {'% ditandai':>11} {'recall':>8} {'precision':>10}")
    for key, g in tp.groupby(col, observed=True):
        pos = g["label"] == 1
        recall = g.loc[pos, "flag"].mean() if pos.any() else float("nan")
        prec = g.loc[g["flag"], "label"].mean() if g["flag"].any() else float("nan")
        out(
            f"{str(key):<10} {len(g):7d} {g['label'].mean():13.3f} {g['flag'].mean():11.3f} "
            f"{recall:8.3f} {prec:10.3f}"
        )
    out()


fairness("gender")
fairness("age_group")

# ---------- 5. Simpan skor semua pelanggan ----------
all_scored = (
    add_score(model.transform(df))
    .withColumn(
        "tier",
        F.when(F.col("rule_any") == 1, "Pasti churn")
        .when(F.col("p") <= t_lo, "Pasti tidak churn")
        .otherwise("Akan churn"),
    )
    .select(ID_COL, "split", *NUM, *CAT, "rule_any", LABEL, F.col("p").alias("score"), "tier")
)
Path("data/scored").mkdir(parents=True, exist_ok=True)
all_scored.toPandas().to_parquet(SCORED, index=False)
(
    all_scored.groupBy("split", "tier")
    .agg(F.count("*").alias("pelanggan"), F.round(F.avg(LABEL) * 100, 2).alias("churn_aktual_pct"))
    .orderBy("split", "tier")
    .toPandas()
    .to_csv(TIER_SUMMARY, index=False)
)
out("Catatan: skor model hanya bermakna untuk baris rule_any = 0 (kelas 'Pasti churn' ditentukan aturan).")
out("Skor pada baris split 'train' bersifat in-sample (model sudah melihat baris itu).")
out("Untuk dashboard keputusan, gunakan baris split 'valid' dan 'test' sebagai acuan akurasi.")

Path("docs").mkdir(exist_ok=True)
Path("docs/tier_results.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nTersimpan: {SCORED}, {TIER_SUMMARY}, docs/tier_results.txt")
spark.stop()