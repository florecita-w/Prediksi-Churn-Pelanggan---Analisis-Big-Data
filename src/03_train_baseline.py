"""Tahap 3 (PySpark): model dasar Logistic Regression dan Random Forest.

Jalankan dari root proyek (venv aktif), setelah 02_spark_pipeline.py berhasil:
    python src/03_train_baseline.py

Masukan : data/model_ready/churn_split.parquet
Keluaran: ringkasan di terminal + docs/baseline_results.txt

Catatan desain:
- Encoding dan standardisasi ada DI DALAM Pipeline, jadi hanya belajar dari data train.
- Kelas churn hampir seimbang (sekitar 57% churn), jadi TIDAK memakai bobot kelas.
- Evaluasi hanya di data validasi. Data test SENGAJA tidak disentuh (dipakai sekali di skrip 04).
- PERHATIAN: karena churn adalah kelas mayoritas, menandai SEMUA pelanggan sebagai churn
  sudah menghasilkan recall 100%. Target recall >= 70% saja tidak cukup bermakna di data ini,
  jadi selalu baca bersama precision, specificity, dan PR-AUC dibanding prevalensi churn.
"""
import time
from pathlib import Path

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

from churn_common import (
    MODEL_NAMES,
    add_score,
    build_pipeline,
    confusion,
    feature_names,
    get_spark,
    load_split,
    rates,
)

RECALL_TARGET = 0.70

lines = []


def out(text: str = "") -> None:
    print(text)
    lines.append(text)


spark = get_spark("churn-baseline")
df = load_split(spark)
train = df.filter(F.col("split") == "train").cache()
valid = df.filter(F.col("split") == "valid").cache()  # data test sengaja tidak dipakai

n_train = train.count()
n_valid = valid.count()
pos_train = train.filter("label = 1").count()
pos_valid = valid.filter("label = 1").count()
prev = pos_valid / n_valid

out(f"Train: {n_train} baris, churn {pos_train} ({pos_train / n_train:.1%})")
out(f"Valid: {n_valid} baris, churn {pos_valid} ({prev:.1%})")
out()
out("== Pembanding sederhana (data validasi) ==")
out(f"Menebak semua 'churn'      : recall 100%, precision {prev:.3f}, specificity 0%")
out(f"Menebak semua 'tidak churn': accuracy {1 - prev:.3f}")
out(f"PR-AUC model acak          : {prev:.3f} (model harus jauh di atas ini)")
out()

ev_roc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
ev_pr = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
)

for name in MODEL_NAMES:
    t0 = time.time()
    model = build_pipeline(name).fit(train)
    pred = add_score(model.transform(valid)).cache()

    roc = ev_roc.evaluate(pred)
    pr = ev_pr.evaluate(pred)
    # ambang agar 70% churner terdeteksi = kuantil ke-30 dari skor pelanggan yang benar-benar churn
    t_rec = pred.filter("label = 1").approxQuantile("p", [1 - RECALL_TARGET], 0.001)[0]

    out(f"=== {name} (data validasi, {time.time() - t0:.0f} detik) ===")
    out(f"ROC-AUC : {roc:.3f}")
    out(f"PR-AUC  : {pr:.3f}  (acak = {prev:.3f})")
    for label_text, t in (("0.500", 0.5), (f"{t_rec:.3f} (recall >= {RECALL_TARGET:.0%})", t_rec)):
        r = rates(*confusion(pred, t))
        out(
            f"Ambang {label_text:<24}-> precision {r['precision']:.3f} | recall {r['recall']:.3f} | "
            f"specificity {r['specificity']:.3f} | F1 {r['f1']:.3f} | accuracy {r['accuracy']:.3f}"
        )

    # kepentingan fitur
    est = model.stages[-1]
    if hasattr(est, "featureImportances"):
        vals = [float(v) for v in est.featureImportances.toArray()]
        kind = "kepentingan fitur"
    else:
        vals = [abs(float(v)) for v in est.coefficients.toArray()]
        kind = "|koefisien| pada fitur terstandardisasi"
    names = feature_names(pred.schema["features_raw"])
    if len(names) != len(vals):
        names = [f"fitur_{i}" for i in range(len(vals))]
    top = sorted(zip(names, vals), key=lambda x: -x[1])[:8]
    out(f"Teratas ({kind}):")
    for n_, v_ in top:
        out(f"    {n_:<32} {v_:.3f}")
    out()
    pred.unpersist()

out("Catatan: skor sangat tinggi (ROC-AUC mendekati 1) pada data seperti ini patut dicurigai")
out("berasal dari aturan pembuatan data yang sederhana. Laporkan sebagai keterbatasan,")
out("bukan sebagai bukti model hebat.")

Path("docs").mkdir(exist_ok=True)
Path("docs/baseline_results.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nRingkasan tersimpan: docs/baseline_results.txt")
spark.stop()