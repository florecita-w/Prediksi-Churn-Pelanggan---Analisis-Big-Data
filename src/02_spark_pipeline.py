"""Tahap 3 (PySpark): validasi data, buang baris kosong, split train/valid/test, simpan.

Jalankan dari root proyek (venv aktif), SETELAH 01_ingest_profile.py berhasil:
    python src/02_spark_pipeline.py

Masukan : data/curated/churn.parquet       (training asli, 440.833 baris)
          data/curated/churn_test.parquet  (testing asli, 64.374 baris, berlabel)
Keluaran: data/model_ready/churn_split.parquet  (kolom 'split': train / valid / test,
          kolom 'source': training / testing = asal file; dipakai skenario 2)

Skema split (sesuai laporan Bab 6.1.3):
    training asli -> train (80%) + valid (20%)   diacak lewat hash customer_id
    testing asli  -> test, dipakai UTUH, hanya untuk evaluasi akhir

Catatan desain:
- Split train/valid memakai hash customer_id (bukan rand): hasilnya sama di setiap perangkat
  dan setiap dijalankan, serta tidak butuh window global sehingga tetap efisien di data besar.
  Urutan baris training berkorelasi dengan label, tetapi hash mengacak, jadi aman.
- Proporsi churn tiap split dicetak di akhir sebagai bukti bahwa train dan valid seimbang.
- Testing sengaja TIDAK digabung dengan training: sebarannya berbeda (churn 47,4% vs 56,7%)
  dan perbedaan itulah yang menunjukkan kinerja model pada data baru.
- Pembersihan dilakukan terpisah per file. ID pelanggan tidak dianggap ganda lintas file,
  karena nomor ID di training dan testing tidak saling berkaitan.
- Encoding dan standardisasi TIDAK dilakukan di sini, melainkan di dalam Pipeline model,
  supaya hanya belajar dari data train.
"""
import shutil
from pathlib import Path

from pyspark.sql import Window
from pyspark.sql import functions as F

from churn_common import (
    CAT,
    CURATED,
    CURATED_TEST,
    ID_COL,
    LABEL,
    MODEL_READY,
    NUM,
    SEED,
    get_spark,
)

TRAIN_PCT = 80  # sisanya (20) menjadi valid; testing asli menjadi test

spark = get_spark("churn-prep")
print("Versi Spark:", spark.version)


def bersihkan(df, nama):
    """Validasi dan pembersihan satu file. Mengembalikan DataFrame bersih."""
    print(f"\n===== {nama} =====")
    n_raw = df.count()
    print(f"Baris mentah: {n_raw}")

    df = df.dropna(how="any")
    n_clean = df.count()
    print(f"Baris dengan nilai kosong yang dibuang: {n_raw - n_clean}")

    for c in NUM:
        df = df.withColumn(c, F.col(c).cast("double"))
    df = df.withColumn(ID_COL, F.col(ID_COL).cast("long")).withColumn(LABEL, F.col(LABEL).cast("int"))

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
        raise SystemExit(f"Label churn di {nama} berisi nilai selain 0/1. Periksa data sebelum lanjut.")

    print(f"Baris siap dipakai: {df.count()}")
    return df


# ---------- 1-3. Muat dan bersihkan, per file ----------
if not Path(CURATED_TEST).exists():
    raise SystemExit(f"{CURATED_TEST} belum ada. Jalankan python src/01_ingest_profile.py dulu "
                     "(file testing harus ada di data/raw).")
train_full = bersihkan(spark.read.parquet(CURATED), "TRAINING asli")
test = bersihkan(spark.read.parquet(CURATED_TEST), "TESTING asli")

# ---------- 4. Split ----------
# training: hash ID -> 80% train, 20% valid. testing: seluruhnya test.
bucket = F.pmod(F.xxhash64(F.col(ID_COL), F.lit(SEED)), F.lit(100))
train_full = (
    train_full.withColumn("split", F.when(bucket < TRAIN_PCT, "train").otherwise("valid"))
    .withColumn("source", F.lit("training"))
)
test = test.withColumn("split", F.lit("test")).withColumn("source", F.lit("testing"))
df = train_full.unionByName(test)

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
print("\nProporsi churn per split (train dan valid harus mirip; test memang berbeda):")
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