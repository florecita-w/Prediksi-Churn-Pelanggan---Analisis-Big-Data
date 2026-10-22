"""Konfigurasi dan fungsi bersama untuk skrip 02-04 (Customer Churn Dataset).

File ini tidak dijalankan langsung. Skrip 02, 03, dan 04 mengimpornya.

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

CURATED = "data/curated/churn.parquet"            # training asli
CURATED_TEST = "data/curated/churn_test.parquet"  # testing asli (berlabel, dipakai utuh)
MODEL_READY = "data/model_ready/churn_split.parquet"
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

# --- Skenario evaluasi (pilih lewat variabel lingkungan SKENARIO) ---
#   1 = lintas-file : latih di training (train 80% / valid 20%), uji UTUH di file testing
#   2 = dalam-file  : file testing dibagi 80/20 (latih+valid = 80%, uji = 20%)
SCEN = os.environ.get("SKENARIO", "1").strip()
if SCEN not in ("1", "2"):
    raise SystemExit("SKENARIO harus 1 atau 2. Contoh (CMD): set SKENARIO=2")
SUFFIX = f"_s{SCEN}"
SCORED = f"data/scored/scored_customers_s{SCEN}.parquet"
TIER_SUMMARY = f"data/scored/tier_summary_s{SCEN}.csv"

# --- Rekayasa fitur (skrip 04_feature_engineering.py) ---
FEATURES_PATH = f"data/model_ready/churn_features{SUFFIX}.parquet"
FLAG_BASES = ["age", "support_calls", "payment_delay", "total_spend"]
FE_NUM = (
    ["is_monthly", "rule_any", "spend_per_tenure", "support_x_delay"]
    + [f"{c}_{side}_pure" for c in FLAG_BASES for side in ("lo", "hi")]
)
# USE_GENDER=0 untuk melatih tanpa gender (keputusan etika, lihat P15). Kolom gender tetap
# ada di data supaya pemeriksaan bias per gender di skrip 04 tetap jalan.
CAT_MODEL = CAT if os.environ.get("USE_GENDER", "1").strip() == "1" else [c for c in CAT if c != "gender"]


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


def use_features() -> bool:
    """True bila file fitur skenario aktif sudah dibuat oleh 04_feature_engineering.py."""
    return os.path.exists(FEATURES_PATH)


def use_fe() -> bool:
    """Model A (default) memakai fitur asli saja. USE_FE=1 -> Model B (+ fitur turunan)."""
    return os.environ.get("USE_FE", "0").strip() == "1"


def run_tag() -> str:
    """Penanda hasil: _s1_A = skenario 1, Model A; _s2_B = skenario 2, Model B (+ fitur turunan)."""
    return f"_s{SCEN}_{'B' if use_fe() else 'A'}"


def model_num() -> list:
    return NUM + FE_NUM if use_fe() else NUM


def load_split(spark):
    if not use_features():
        raise SystemExit(f"{FEATURES_PATH} belum ada. Jalankan: "
                         f"set SKENARIO={SCEN} lalu python src/04_feature_engineering.py")
    return spark.read.parquet(FEATURES_PATH).withColumn("label", F.col(LABEL).cast("double"))


def build_pipeline(name: str) -> Pipeline:
    """Imputasi tidak diperlukan (data lengkap setelah baris kosong dibuang).
    Encoding dan standardisasi ada di dalam Pipeline, jadi hanya belajar dari data train."""
    cats = CAT_MODEL
    indexer = StringIndexer(
        inputCols=cats, outputCols=[f"{c}_idx" for c in cats], handleInvalid="keep"
    )
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cats],
        outputCols=[f"{c}_oh" for c in cats],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=model_num() + [f"{c}_oh" for c in cats], outputCol="features_raw"
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