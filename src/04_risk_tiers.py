"""Tahap 4 (PySpark): tentukan tiga kelas risiko churn dan simpan skor untuk dashboard.

Kelas: "Pasti churn", "Akan churn", "Pasti tidak churn".

Jalankan dari root proyek (venv aktif), setelah 02_spark_pipeline.py berhasil:
    python src/04_risk_tiers.py

Masukan : data/model_ready/churn_split.parquet
Keluaran: docs/tier_results.txt, data/scored/scored_customers.parquet,
          data/scored/tier_summary.csv

Cara kerja:
1. Latih Logistic Regression dan Random Forest di data train, pilih yang PR-AUC-nya
   tertinggi di data validasi.
2. Ambang tiga kelas dipilih dari skor di data validasi:
   - "Pasti churn"       : skor >= ambang atas, dengan precision >= PRECISION_TARGET
   - "Pasti tidak churn" : skor <= ambang bawah, dengan NPV >= NPV_TARGET
   - "Akan churn"        : sisanya (zona tidak pasti)
3. Data test dipakai SEKALI untuk menilai hasil akhir (dengan selang kepercayaan 95%).
4. Uji bias per gender dan kelompok usia dicetak untuk lembar kepatuhan dan etika.

Catatan: "pasti" di sini berarti keyakinan tinggi yang terukur, bukan jaminan. Setiap kelas
dilaporkan dengan precision/NPV dan selang kepercayaannya.
"""
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

PRECISION_TARGET = 0.90   # kelas "Pasti churn"
NPV_TARGET = 0.90         # kelas "Pasti tidak churn"
MIN_TIER = 500            # minimal pelanggan di kelas "pasti" agar ambang stabil

lines = []


def out(text: str = "") -> None:
    print(text)
    lines.append(text)


spark = get_spark("churn-risk-tiers")
df = load_split(spark)
train = df.filter(F.col("split") == "train").cache()
valid = df.filter(F.col("split") == "valid").cache()
test = df.filter(F.col("split") == "test").cache()

n_train, n_valid, n_test = train.count(), valid.count(), test.count()
out(f"Train {n_train} | Valid {n_valid} | Test {n_test} [test dikunci sampai evaluasi akhir]")
out()

ev_pr = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
)
ev_roc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

# ---------- 1. Pilih model ----------
out("== Pemilihan model (data validasi) ==")
fitted, scores = {}, {}
for name in MODEL_NAMES:
    model = build_pipeline(name).fit(train)
    pv = model.transform(valid)
    fitted[name] = model
    scores[name] = (ev_pr.evaluate(pv), ev_roc.evaluate(pv))
    out(f"{name:20s} PR-AUC {scores[name][0]:.4f} | ROC-AUC {scores[name][1]:.4f}")
best = max(MODEL_NAMES, key=lambda n: scores[n][0])
model = fitted[best]
out(f"Model terpilih: {best}")
out()

# ---------- 2. Ambang dari data validasi ----------
valid_scored = add_score(model.transform(valid)).select("p", "label")
vdata = [(r["p"], r["label"]) for r in valid_scored.collect()]
prev = sum(y for _, y in vdata) / len(vdata)
out(f"Prevalensi churn di validasi: {prev:.3f}")
out()

out("== Opsi target kelas 'Pasti churn' (validasi) ==")
out("target precision | ambang | pelanggan | % pelanggan | precision | % churner tertangkap")
for pt in (0.80, 0.90, 0.95):
    t = pick_high(vdata, pt, MIN_TIER)
    if t is None:
        out(f"{pt:>16.0%} | tidak tercapai dengan minimal {MIN_TIER} pelanggan")
    else:
        n, share, prec, rec = high_stats(vdata, t)
        out(f"{pt:>16.0%} | {t:6.3f} | {n:9d} | {share:11.1%} | {prec:9.3f} | {rec:20.1%}")
out()
out("== Opsi target kelas 'Pasti tidak churn' (validasi) ==")
out("target NPV | ambang | pelanggan | % pelanggan | NPV | churner yang lolos")
for nt in (0.80, 0.90, 0.95):
    t = pick_low(vdata, nt, MIN_TIER)
    if t is None:
        out(f"{nt:>9.0%} | tidak tercapai dengan minimal {MIN_TIER} pelanggan")
    else:
        n, share, npv, fn = low_stats(vdata, t)
        out(f"{nt:>9.0%} | {t:6.3f} | {n:9d} | {share:11.1%} | {npv:.3f} | {fn}")
out()

t_hi = pick_high(vdata, PRECISION_TARGET, MIN_TIER)
t_lo = pick_low(vdata, NPV_TARGET, MIN_TIER)
if t_hi is None:
    out(f"PERINGATAN: target precision {PRECISION_TARGET:.0%} tidak tercapai. Kelas 'Pasti churn' dikosongkan.")
    t_hi = 1.01
if t_lo is None:
    out(f"PERINGATAN: target NPV {NPV_TARGET:.0%} tidak tercapai. Kelas 'Pasti tidak churn' dikosongkan.")
    t_lo = -0.01
if t_lo >= t_hi:
    out("PERINGATAN: ambang bawah >= ambang atas. Longgarkan target precision/NPV.")
out(f"Ambang dipakai (precision {PRECISION_TARGET:.0%}, NPV {NPV_TARGET:.0%}): atas {t_hi:.3f}, bawah {t_lo:.3f}")
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

out("== Skor vs churn aktual per desil (validasi) ==")
for d, mp, my in deciles(vdata):
    out(f"desil {d:2d}: rata-rata skor {mp:.3f} | churn aktual {my:.3f}")
out()

# ---------- 3. Evaluasi akhir di data test (sekali) ----------
test_pred = add_score(model.transform(test)).cache()
tdata = [(r["p"], r["label"]) for r in test_pred.select("p", "label").collect()]
trows = tier_rows(tdata, tier_of)
print_tiers(trows, f"== EVALUASI AKHIR di data test ({best}) ==", with_ci=True)

brier = test_pred.select(F.avg((F.col("p") - F.col("label")) ** 2)).first()[0]
out(
    f"PR-AUC test {ev_pr.evaluate(test_pred):.4f} | ROC-AUC test {ev_roc.evaluate(test_pred):.4f} "
    f"| Brier {brier:.4f}"
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
tp = test_pred.select("gender", "age", "label", "p").toPandas()
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
        F.when(F.col("p") >= t_hi, "Pasti churn")
        .when(F.col("p") <= t_lo, "Pasti tidak churn")
        .otherwise("Akan churn"),
    )
    .select(ID_COL, "split", *NUM, *CAT, LABEL, F.col("p").alias("score"), "tier")
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
out("Catatan: skor pada baris split 'train' bersifat in-sample (model sudah melihat baris itu).")
out("Untuk dashboard keputusan, gunakan baris split 'valid' dan 'test' sebagai acuan akurasi.")

Path("docs").mkdir(exist_ok=True)
Path("docs/tier_results.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nTersimpan: {SCORED}, {TIER_SUMMARY}, docs/tier_results.txt")
spark.stop()