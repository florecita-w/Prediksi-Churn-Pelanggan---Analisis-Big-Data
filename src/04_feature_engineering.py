"""Tahap 5 (PySpark): Rekayasa fitur.

Jalankan dari root proyek (venv aktif), setelah 02 berhasil, SEKALI PER SKENARIO:
    set SKENARIO=1
    python src/04_feature_engineering.py
    set SKENARIO=2
    python src/04_feature_engineering.py        (perintah 'set' berlaku untuk jendela CMD itu)

Skenario 1: latih di training (train 80% / valid 20%), uji UTUH di file testing.
Skenario 2: HANYA file testing, dibagi 80/20; bagian 80% dibagi lagi 80/20 menjadi
            train (64%) dan valid (16%), sisanya 20% menjadi test.

Masukan : data/model_ready/churn_split.parquet
Keluaran: data/model_ready/churn_features_s{N}.parquet
          docs/feature_engineering_s{N}.txt, docs/feature_rules_s{N}.json

Prinsip anti-kebocoran: ambang 'ekor murni' dicari HANYA dari data train skenario itu,
lalu diterapkan ke train, valid, dan test dengan ambang yang sama.
Fitur turunan ini untuk Model B (diagnostik). Model utama (Model A) memakai 10 fitur asli.
"""
import json
import shutil
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from churn_common import FEATURES_PATH, FLAG_BASES, MODEL_READY, SCEN, SEED, get_spark
from fe_utils import find_pure_tails

PURITY = 0.999   # minimal 99,9% churn di dalam ekor
MIN_N = 500      # ekor harus berisi minimal 500 pelanggan train

lines = []


def out(text=""):
    print(text)
    lines.append(text)


spark = get_spark("churn-feature-engineering")
df = spark.read.parquet(MODEL_READY).withColumn("label", F.col("churn").cast("double"))
if "source" not in df.columns:
    raise SystemExit("Kolom 'source' tidak ada. Jalankan ulang python src/02_spark_pipeline.py (versi terbaru).")
out(f"== Skenario {SCEN} ==")
if SCEN == "2":
    # hanya file testing; 64% train / 16% valid / 20% test, deterministik lewat hash ID
    bucket = F.pmod(F.xxhash64(F.col("customer_id"), F.lit(SEED)), F.lit(100))
    df = df.filter(F.col("source") == "testing").withColumn(
        "split",
        F.when(bucket < 64, "train").when(bucket < 80, "valid").otherwise("test"),
    )
train = df.filter(F.col("split") == "train").cache()
n_train = train.count()
out(f"Train: {n_train} baris")
for sp in ("valid", "test"):
    out(f"{sp.capitalize()}: {df.filter(F.col('split') == sp).count()} baris")

# 1. Cari ekor murni di train
rules = {}
out("\n== Ekor murni (hanya dari data train) ==")
for c in FLAG_BASES:
    g = (
        train.groupBy(F.floor(F.col(c)).alias("v"))
        .agg(F.count("*").alias("n"), F.sum("label").alias("k"))
        .collect()
    )
    rows = [(r["v"], r["n"], int(r["k"])) for r in g]
    lo, hi = find_pure_tails(rows, target=1, purity=PURITY, min_n=MIN_N)
    lo0, hi0 = find_pure_tails(rows, target=0, purity=PURITY, min_n=MIN_N)
    rules[c] = {"lo_max": lo, "hi_min": hi}
    out(f"{c:17s} churn-murni: nilai <= {lo} | nilai >= {hi}"
        f"   (tidak-churn-murni: <= {lo0} | >= {hi0})")

# 2. Bangun fitur turunan (ambang sama untuk semua split)
feat = df.withColumn("is_monthly", (F.col("contract_length") == "Monthly").cast("double"))
flag_cols = []
for c in FLAG_BASES:
    lo, hi = rules[c]["lo_max"], rules[c]["hi_min"]
    lo_cond = F.lit(False) if lo is None else (F.col(c) < F.lit(lo + 1))
    hi_cond = F.lit(False) if hi is None else (F.col(c) >= F.lit(hi))
    feat = feat.withColumn(f"{c}_lo_pure", lo_cond.cast("double"))
    feat = feat.withColumn(f"{c}_hi_pure", hi_cond.cast("double"))
    flag_cols += [f"{c}_lo_pure", f"{c}_hi_pure"]

any_flag = F.col("is_monthly") > 0
for fc in flag_cols:
    any_flag = any_flag | (F.col(fc) > 0)
feat = feat.withColumn("rule_any", any_flag.cast("double"))
feat = feat.withColumn("spend_per_tenure", F.col("total_spend") / F.col("tenure"))
feat = feat.withColumn("support_x_delay", F.col("support_calls") * F.col("payment_delay"))
feat = feat.drop("label").cache()

# 3. Periksa: apakah aturan berlaku sama di valid/test? (bukti tidak bocor)
out("\n== Churn rate di kelompok rule_any = 1 vs 0, per split ==")
chk = (
    feat.groupBy("split", "rule_any")
    .agg(F.count("*").alias("n"), F.round(F.avg("churn"), 4).alias("churn_rate"))
    .orderBy("split", "rule_any")
    .toPandas()
)
out(chk.to_string(index=False))

out("\n== Churn rate per flag (train/valid/test) ==")
for fc in ["is_monthly"] + flag_cols:
    r = (
        feat.filter(F.col(fc) > 0).groupBy("split")
        .agg(F.count("*").alias("n"), F.round(F.avg("churn"), 4).alias("churn_rate"))
        .toPandas()
    )
    if len(r):
        out(f"{fc}: " + "; ".join(f"{x.split} n={x.n} churn={x.churn_rate}" for x in r.itertuples()))

# 4. Cek gender setelah rule_any dikendalikan (bahan P15)
out("\n== Churn rate per gender pada pelanggan di luar aturan (train, rule_any = 0) ==")
gdf = (
    feat.filter((F.col("split") == "train") & (F.col("rule_any") == 0))
    .groupBy("gender")
    .agg(F.count("*").alias("n"), F.round(F.avg("churn"), 4).alias("churn_rate"))
    .toPandas()
)
out(gdf.to_string(index=False))
out("Jika selisih gender masih besar di sini, gender punya pengaruh sendiri; "
    "jika mengecil, selisih itu sebagian besar efek fitur lain.")

# 5. Simpan (pola sama dengan skrip 02: Spark dulu, cadangan pandas di Windows)
out_path = Path(FEATURES_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
if out_path.is_dir():
    shutil.rmtree(out_path, ignore_errors=True)
elif out_path.exists():
    out_path.unlink()
try:
    feat.write.mode("overwrite").parquet(FEATURES_PATH)
    out(f"\nTersimpan lewat Spark: {FEATURES_PATH}")
except Exception as exc:  # umum di Windows tanpa winutils/hadoop.dll
    out(f"\nSpark gagal menulis Parquet ({type(exc).__name__}). Memakai cadangan pandas.")
    if out_path.is_dir():  # Spark sempat membuat folder kosong, hapus dulu
        shutil.rmtree(out_path, ignore_errors=True)
    feat.toPandas().to_parquet(FEATURES_PATH, index=False)
    out(f"Tersimpan lewat pandas: {FEATURES_PATH}")

Path("docs").mkdir(exist_ok=True)
Path(f"docs/feature_rules_s{SCEN}.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
Path(f"docs/feature_engineering_s{SCEN}.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"Ringkasan: docs/feature_engineering_s{SCEN}.txt | Aturan: docs/feature_rules_s{SCEN}.json")