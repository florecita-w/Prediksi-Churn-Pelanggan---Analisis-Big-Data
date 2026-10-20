"""Tahap 2 (PySpark): validasi data, buang baris kosong, split train/valid/test, simpan.

Jalankan dari root proyek (venv aktif), SETELAH 01_ingest_profile.py berhasil:
    python src/02_spark_pipeline.py

Masukan : data/curated/churn.parquet
Keluaran: data/model_ready/churn_split.parquet  (kolom 'split': train / valid / test)

Catatan desain:
- Split memakai hash customer_id (bukan rand): hasilnya sama di setiap perangkat dan setiap
  dijalankan, dan tidak butuh window global sehingga tetap efisien di data besar.
- Karena datanya besar, pembagian tiga split hampir stratified. Tabel akhir menunjukkan
  proporsi churn di tiap split sebagai buktinya.
- Encoding dan standardisasi TIDAK dilakukan di sini, melainkan di dalam Pipeline model
  (skrip 03), supaya hanya belajar dari data train.
"""
import shutil
from pathlib import Path

from pyspark.sql import Window
from pyspark.sql import functions as F

from churn_common import CAT, CURATED, ID_COL, LABEL, MODEL_READY, NUM, SEED, get_spark

TRAIN_PCT, VALID_PCT = 70, 15  # sisanya (15) menjadi test

spark = get_spark("churn-prep")
print("Versi Spark:", spark.version)

df = spark.read.parquet(CURATED)
n_raw = df.count()
print(f"\nBaris mentah: {n_raw}")

# ---------- 1. Baris kosong ----------
df = df.dropna(how="any")
n_clean = df.count()
print(f"Baris dengan nilai kosong yang dibuang: {n_raw - n_clean}")

# ---------- 2. Tipe data ----------
for c in NUM:
    df = df.withColumn(c, F.col(c).cast("double"))
df = df.withColumn(ID_COL, F.col(ID_COL).cast("long")).withColumn(LABEL, F.col(LABEL).cast("int"))

# ---------- 3. Pemeriksaan kualitas ----------
dup_full = n_clean - df.dropDuplicates().count()
dup_id = n_clean - df.select(ID_COL).distinct().count()
print(f"Duplikat penuh      : {dup_full}")
print(f"ID pelanggan ganda  : {dup_id}")
if dup_id > 0:
    print("  -> ID ganda dibuang (dipertahankan satu baris per ID)")
    df = df.dropDuplicates([ID_COL])

neg_expr = " OR ".join(f"{c} < 0" for c in NUM)
print(f"Baris dengan nilai numerik negatif: {df.filter(neg_expr).count()}")
bad_label = df.filter(~F.col(LABEL).isin(0, 1)).count()
print(f"Baris dengan label di luar 0/1     : {bad_label}")
if bad_label:
    raise SystemExit("Label churn berisi nilai selain 0/1. Periksa data sebelum lanjut.")

n_final = df.count()
print(f"\nBaris siap dipakai: {n_final}")

# ---------- 4. Split deterministik berbasis hash ID ----------
bucket = F.pmod(F.xxhash64(F.col(ID_COL), F.lit(SEED)), F.lit(100))
df = df.withColumn(
    "split",
    F.when(bucket < TRAIN_PCT, "train")
    .when(bucket < TRAIN_PCT + VALID_PCT, "valid")
    .otherwise("test"),
)

# ---------- 5. Simpan ----------
out_path = Path(MODEL_READY)
out_path.parent.mkdir(parents=True, exist_ok=True)
if out_path.is_dir():
    shutil.rmtree(out_path, ignore_errors=True)
elif out_path.exists():
    out_path.unlink()

try:
    df.write.mode("overwrite").parquet(MODEL_READY)
    print(f"\nTersimpan lewat Spark: {MODEL_READY}")
except Exception as exc:  # umum di Windows tanpa winutils/hadoop.dll
    print(f"\nSpark gagal menulis Parquet ({type(exc).__name__}). Memakai cadangan pandas.")
    if out_path.is_dir():
        shutil.rmtree(out_path, ignore_errors=True)
    df.toPandas().to_parquet(MODEL_READY, index=False)
    print(f"Tersimpan lewat pandas: {MODEL_READY}")

# ---------- 6. Verifikasi ----------
chk = spark.read.parquet(MODEL_READY)
print(f"\nBaris tersimpan: {chk.count()}")
print("\nProporsi churn per split:")
(
    chk.groupBy("split")
    .agg(
        F.count("*").alias("baris"),
        F.round(F.avg(LABEL) * 100, 2).alias("churn_pct"),
        F.round(F.avg((F.col("gender") == "Male").cast("int")) * 100, 2).alias("pria_pct"),
    )
    .orderBy("split")
    .show()
)

w = Window.partitionBy("split")
print("Komposisi kategori per split (persen):")
for c in CAT:
    (
        chk.groupBy("split", c)
        .count()
        .withColumn("persen", F.round(F.col("count") / F.sum("count").over(w) * 100, 2))
        .orderBy("split", c)
        .show(truncate=False)
    )
spark.stop()