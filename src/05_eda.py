"""Tahap 4 - EDA (Pertemuan 10: Laporan Analisis Eksploratif & Hipotesis).
Input : data/curated/churn.parquet
Output: reports/eda/*.png, docs/eda_summary.txt, docs/eda_tables/*.csv
Jalankan: python src/05_eda.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "data/curated/churn.parquet"
FIG = "reports/eda"
TAB = "docs/eda_tables"
NUM = ["age", "tenure", "usage_frequency", "support_calls",
       "payment_delay", "total_spend", "last_interaction"]
CAT = ["gender", "subscription_type", "contract_length"]
for d in (FIG, TAB, "docs"):
    os.makedirs(d, exist_ok=True)

df = pd.read_parquet(SRC)
n0 = len(df)
df = df.dropna(subset=["churn"]).copy()
df["churn"] = df["churn"].astype(int)
base = df["churn"].mean()
out = []
def say(s=""):
    print(s); out.append(s)

say(f"Baris curated: {n0:,} | setelah buang baris kosong: {len(df):,}")
say(f"Churn rate keseluruhan: {base:.2%} (kelas mayoritas, bukan imbalanced)")

# 1. Distribusi target
fig, ax = plt.subplots(figsize=(4, 3.5))
vc = df["churn"].value_counts().sort_index()
ax.bar(["Tidak churn", "Churn"], vc.values, color=["#4C78A8", "#E45756"])
for i, v in enumerate(vc.values):
    ax.text(i, v, f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, vc.max() * 1.2); ax.set_title("Distribusi target churn")
fig.tight_layout(); fig.savefig(f"{FIG}/01_target.png", dpi=150); plt.close(fig)

# 2. Ringkasan numerik per kelas
desc = df.groupby("churn")[NUM].agg(["mean", "median"]).round(2)
desc.to_csv(f"{TAB}/numeric_by_churn.csv")
say("\n== Rata-rata/median fitur numerik per kelas churn ==")
say(desc.T.to_string())

# 3. Histogram per kelas
fig, axes = plt.subplots(2, 4, figsize=(15, 6))
for ax, c in zip(axes.ravel(), NUM):
    bins = np.linspace(df[c].min(), df[c].max(), 25)
    for k, col, lab in ((0, "#4C78A8", "Tidak churn"), (1, "#E45756", "Churn")):
        ax.hist(df.loc[df.churn == k, c], bins=bins, alpha=.55, color=col, label=lab)
    ax.set_title(c, fontsize=9)
axes.ravel()[0].legend(fontsize=7)
axes.ravel()[-1].axis("off")
fig.suptitle("Distribusi fitur numerik menurut churn"); fig.tight_layout()
fig.savefig(f"{FIG}/02_numeric_hist.png", dpi=150); plt.close(fig)

# 4. Churn rate per kategori
rows = []
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, c in zip(axes, CAT):
    g = df.groupby(c)["churn"].agg(["mean", "size"]).reset_index()
    ax.bar(g[c].astype(str), g["mean"], color="#E45756")
    ax.axhline(base, color="k", ls="--", lw=1)
    for i, (m, s) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, m, f"{m:.1%}\nn={s:,}", ha="center", va="bottom", fontsize=7)
        rows.append((c, g[c].iloc[i], s, m))
    ax.set_ylim(0, 1.05); ax.set_title(f"Churn rate per {c}", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/03_categorical.png", dpi=150); plt.close(fig)
cat_tab = pd.DataFrame(rows, columns=["fitur", "nilai", "n", "churn_rate"])
cat_tab.to_csv(f"{TAB}/churn_by_category.csv", index=False)
say("\n== Churn rate per kategori (garis dasar %.1f%%) ==" % (base * 100))
say(cat_tab.assign(churn_rate=lambda t: (t.churn_rate * 100).round(1)).to_string(index=False))

# 5. Churn rate per bin numerik
fig, axes = plt.subplots(2, 4, figsize=(15, 6))
bin_rows = []
for ax, c in zip(axes.ravel(), NUM):
    b = pd.qcut(df[c], 8, duplicates="drop") if df[c].nunique() > 10 else df[c]
    g = df.groupby(b, observed=True)["churn"].mean()
    ax.plot(range(len(g)), g.values, marker="o", color="#E45756")
    ax.axhline(base, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(i) for i in g.index], rotation=60, fontsize=6)
    ax.set_ylim(0, 1.05); ax.set_title(f"Churn rate vs {c}", fontsize=9)
    for i, v in g.items():
        bin_rows.append((c, str(i), round(v, 4)))
axes.ravel()[-1].axis("off")
fig.tight_layout(); fig.savefig(f"{FIG}/04_churn_by_bin.png", dpi=150); plt.close(fig)
pd.DataFrame(bin_rows, columns=["fitur", "bin", "churn_rate"]).to_csv(
    f"{TAB}/churn_by_numeric_bin.csv", index=False)

# 6. Korelasi
corr = df[NUM + ["churn"]].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, shrink=.8); ax.set_title("Korelasi (Pearson)")
fig.tight_layout(); fig.savefig(f"{FIG}/05_correlation.png", dpi=150); plt.close(fig)
cc = corr["churn"].drop("churn").sort_values(key=abs, ascending=False)
say("\n== Korelasi fitur numerik dengan churn (urut |r|) ==")
say(cc.round(3).to_string())
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(NUM) for b in NUM[i + 1:]
         if abs(corr.loc[a, b]) >= 0.5]
say("Pasangan fitur berkorelasi kuat (|r|>=0.5): " +
    (", ".join(f"{a}-{b} ({r:.2f})" for a, b, r in pairs) or "tidak ada"))

# 7. Outlier (IQR)
say("\n== Outlier (aturan 1.5xIQR) ==")
orow = []
for c in NUM:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    k = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    orow.append((c, df[c].min(), df[c].max(), k, round(k / len(df) * 100, 2)))
ot = pd.DataFrame(orow, columns=["fitur", "min", "max", "n_outlier", "persen"])
ot.to_csv(f"{TAB}/outliers.csv", index=False); say(ot.to_string(index=False))

# 8. Pola tidak wajar: kelompok dengan churn ekstrem
say("\n== Pola tidak wajar: segmen dengan churn rate ekstrem ==")
for c, col in (("support_calls", "support_calls"), ("payment_delay", "payment_delay")):
    top = df.groupby(col)["churn"].agg(["mean", "size"])
    lo, hi = top["mean"].min(), top["mean"].max()
    say(f"{c}: churn rate berkisar {lo:.1%} - {hi:.1%} antar nilai")
for c in NUM:
    for k in (0, 1):
        pass
pure = []
for c in NUM:
    g = df.groupby(pd.qcut(df[c], 10, duplicates="drop"), observed=True)["churn"].mean()
    if g.max() > .97 or g.min() < .03:
        pure.append(f"{c}: bin dengan churn {g.min():.1%}-{g.max():.1%}")
say("Bin hampir 'murni' satu kelas (indikasi data sintetis/aturan deterministik): " +
    ("; ".join(pure) or "tidak ditemukan"))

# 9. Hipotesis otomatis
say("\n== Draf hipotesis (verifikasi di model & uji statistik) ==")
for c, r in cc.items():
    if abs(r) >= 0.1:
        arah = "naik" if r > 0 else "turun"
        say(f"H: peluang churn {arah} saat {c} meningkat (r={r:.2f}).")
w = cat_tab.loc[(cat_tab.churn_rate - base).abs().sort_values(ascending=False).index[:3]]
for _, r in w.iterrows():
    say(f"H: {r.fitur}={r.nilai} berbeda dari rata-rata (churn {r.churn_rate:.1%} vs {base:.1%}).")
say("H0 per fitur: tidak ada perbedaan churn antar kelompok; uji dengan chi-kuadrat (kategorikal) / t-test (numerik).")

open("docs/eda_summary.txt", "w", encoding="utf-8").write("\n".join(out))
print("\nSelesai. Gambar: reports/eda/ | Ringkasan: docs/eda_summary.txt | Tabel: docs/eda_tables/")