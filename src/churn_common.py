"""Konfigurasi dan fungsi bersama untuk skrip 02-04 (Customer Churn Dataset).

File ini tidak dijalankan langsung. Skrip 02, 03, dan 04 mengimpornya.
Letakkan di folder src/ bersama skrip lainnya.
"""
import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CURATED = "data/curated/churn.parquet"
MODEL_READY = "data/model_ready/churn_split.parquet"
SCORED = "data/scored/scored_customers.parquet"
TIER_SUMMARY = "data/scored/tier_summary.csv"
SEED = 42

ID_COL = "customer_id"      # hanya identitas, TIDAK dipakai sebagai fitur
LABEL = "churn"
NUM = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "total_spend",
    "last_interaction",
]
CAT = ["gender", "subscription_type", "contract_length"]
MODEL_NAMES = ("Logistic Regression", "Random Forest")


def get_spark(app_name: str) -> SparkSession:
    """Spark lokal. Memori driver dapat diubah: set SPARK_DRIVER_MEMORY=2g (default 4g)."""
    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_split(spark):
    return spark.read.parquet(MODEL_READY).withColumn("label", F.col(LABEL).cast("double"))


def build_pipeline(name: str) -> Pipeline:
    """Imputasi tidak diperlukan (data lengkap setelah baris kosong dibuang).
    Encoding dan standardisasi ada di dalam Pipeline, jadi hanya belajar dari data train."""
    indexer = StringIndexer(
        inputCols=CAT, outputCols=[f"{c}_idx" for c in CAT], handleInvalid="keep"
    )
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in CAT],
        outputCols=[f"{c}_oh" for c in CAT],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=NUM + [f"{c}_oh" for c in CAT], outputCol="features_raw"
    )
    pre = [indexer, encoder, assembler]
    if name == "Logistic Regression":
        scaler = StandardScaler(
            inputCol="features_raw", outputCol="features", withMean=False, withStd=True
        )
        clf = LogisticRegression(
            featuresCol="features", labelCol="label", regParam=0.001, maxIter=100
        )
        return Pipeline(stages=pre + [scaler, clf])
    clf = RandomForestClassifier(
        featuresCol="features_raw",
        labelCol="label",
        numTrees=100,
        maxDepth=10,
        seed=SEED,
    )
    return Pipeline(stages=pre + [clf])


def add_score(pred):
    """Tambahkan kolom p = probabilitas kelas churn (1)."""
    return pred.withColumn("p", vector_to_array(F.col("probability"))[1])


def confusion(pred, t: float):
    rows = (
        pred.select(
            (F.col("p") >= t).cast("int").alias("yhat"), F.col("label").cast("int").alias("y")
        )
        .groupBy("yhat", "y")
        .count()
        .collect()
    )
    c = {(r["yhat"], r["y"]): r["count"] for r in rows}
    return c.get((1, 1), 0), c.get((1, 0), 0), c.get((0, 1), 0), c.get((0, 0), 0)


def rates(tp: int, fp: int, fn: int, tn: int) -> dict:
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "accuracy": (tp + tn) / n if n else 0.0,
        "flagged": (tp + fp) / n if n else 0.0,
    }


def feature_names(field) -> list:
    """Nama fitur dari metadata kolom features_raw (untuk tabel kepentingan fitur)."""
    attrs = field.metadata.get("ml_attr", {}).get("attrs", {})
    pairs = []
    for group in attrs.values():
        for a in group:
            pairs.append((a["idx"], a["name"]))
    return [n for _, n in sorted(pairs)]