"""Tahap 1: baca CSV mentah, buat profil singkat, simpan Parquet terkurasi.

Jalankan dari root proyek (venv aktif):
    python src/01_ingest_profile.py                    # training DAN testing sekaligus
    python src/01_ingest_profile.py nama_file.csv      # satu file saja (disimpan sbg training)

Jika data/raw berisi file bernama '*training*' dan '*testing*', keduanya diproses:
    training -> data/curated/churn.parquet       + docs/data_profile.txt
    testing  -> data/curated/churn_test.parquet  + docs/data_profile_testing.txt
"""
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
CURATED = Path("data/curated")
DOCS = Path("docs")


def pick_jobs():
    """Daftar (csv, parquet, profil). Training dan testing diproses bila keduanya ada."""
    files = sorted(RAW.glob("*.csv"))
    if not files:
        sys.exit("Tidak ada file CSV di data/raw. Letakkan dataset di sana dulu.")
    if len(sys.argv) > 1:
        path = RAW / sys.argv[1]
        if not path.exists():
            sys.exit(f"File tidak ditemukan: {path}")
        return [(path, "churn.parquet", "data_profile.txt")]
    train = [f for f in files if "training" in f.name.lower()]
    test = [f for f in files if "testing" in f.name.lower()]
    if len(train) == 1 and len(test) == 1:
        return [
            (train[0], "churn.parquet", "data_profile.txt"),
            (test[0], "churn_test.parquet", "data_profile_testing.txt"),
        ]
    if len(files) == 1:
        return [(files[0], "churn.parquet", "data_profile.txt")]
    print("Ada lebih dari satu CSV di data/raw:")
    for f in files:
        print("  -", f.name)
    sys.exit("Jalankan ulang dengan nama file: python src/01_ingest_profile.py <nama_file.csv>")


def to_snake(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"_+", "_", name).lower()


def ingest(path: Path, parquet_name: str, profile_name: str) -> None:
    df = pd.read_csv(path)
    df.columns = [to_snake(c) for c in df.columns]

    lines = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out(f"File          : {path.name}")
    out(f"Ukuran file   : {path.stat().st_size / 1024:.1f} KB")
    out(f"Baris x kolom : {df.shape[0]} x {df.shape[1]}")
    out()

    out("== Tipe data ==")
    out(df.dtypes.to_string())
    out()

    out("== Missing value ==")
    miss = pd.DataFrame({"jumlah": df.isna().sum(), "persen": (df.isna().mean() * 100).round(2)})
    out(miss.to_string())
    out()

    out(f"== Baris duplikat penuh: {int(df.duplicated().sum())} ==")
    out()

    if "churn" in df.columns:
        out("== Distribusi churn ==")
        dist = pd.DataFrame({
            "jumlah": df["churn"].value_counts(),
            "persen": (df["churn"].value_counts(normalize=True) * 100).round(2),
        })
        out(dist.to_string())
        out()
    else:
        out("PERINGATAN: kolom 'churn' tidak ditemukan. Kolom yang ada: " + ", ".join(df.columns))
        out()

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    out("== Statistik numerik ==")
    out(df[num_cols].describe().T.round(2).to_string())
    out()

    out("== Nilai negatif per kolom numerik ==")
    neg = {c: int((df[c] < 0).sum()) for c in num_cols}
    out(pd.Series(neg).to_string())
    out()

    out("== Kolom kategorikal ==")
    if cat_cols:
        for c in cat_cols:
            out(f"[{c}]")
            out(df[c].value_counts(dropna=False).to_string())
            out()
    else:
        out("(tidak ada)")
        out()

    out("== Contoh 5 baris pertama ==")
    out(df.head().to_string())

    CURATED.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    parquet_path = CURATED / parquet_name
    df.to_parquet(parquet_path, index=False)
    back = pd.read_parquet(parquet_path)
    assert back.shape == df.shape, "Jumlah baris/kolom Parquet tidak sama dengan CSV"

    (DOCS / profile_name).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nParquet tersimpan: {parquet_path} ({back.shape[0]} baris)")
    print(f"Ringkasan tersimpan: docs/{profile_name}")


def main() -> None:
    for path, parquet_name, profile_name in pick_jobs():
        print(f"\n########## {path.name} ##########")
        ingest(path, parquet_name, profile_name)


if __name__ == "__main__":
    main()