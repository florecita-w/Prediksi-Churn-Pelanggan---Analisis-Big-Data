"""Fungsi murni (tanpa Spark) untuk menentukan kelas risiko churn.

Dipisahkan dari skrip 04 supaya logikanya bisa diuji sendiri.
Letakkan di folder src/ bersama skrip lainnya.

Format data: list of (p, y) dengan p = skor churn (0-1), y = label aktual (0.0 / 1.0).
"""
from math import sqrt

TIER_ORDER = ["Pasti churn", "Akan churn", "Pasti tidak churn"]


def pick_high(data, target, min_n):
    """Ambang atas terendah sehingga precision kumulatif (skor >= ambang) >= target
    dengan minimal min_n pelanggan. None jika tidak ada."""
    ds = sorted(data, key=lambda r: -r[0])
    best, tp = None, 0
    for i, (p, y) in enumerate(ds, start=1):
        tp += y == 1.0
        if i < len(ds) and ds[i][0] == p:  # lewati sampai akhir kelompok skor yang sama
            continue
        if i >= min_n and tp / i >= target:
            best = p
    return best


def pick_low(data, target, min_n):
    """Ambang bawah tertinggi sehingga NPV kumulatif (skor <= ambang) >= target
    dengan minimal min_n pelanggan. None jika tidak ada."""
    ds = sorted(data, key=lambda r: r[0])
    best, neg = None, 0
    for i, (p, y) in enumerate(ds, start=1):
        neg += y == 0.0
        if i < len(ds) and ds[i][0] == p:
            continue
        if i >= min_n and neg / i >= target:
            best = p
    return best


def high_stats(data, t):
    """Untuk kelas 'Pasti churn': (jumlah, porsi, precision, porsi churner tertangkap)."""
    total_pos = sum(1 for _, y in data if y == 1.0)
    sel = [y for p, y in data if p >= t]
    n = len(sel)
    tp = sum(1 for y in sel if y == 1.0)
    return n, n / len(data), (tp / n if n else 0.0), (tp / total_pos if total_pos else 0.0)


def low_stats(data, t):
    """Untuk kelas 'Pasti tidak churn': (jumlah, porsi, NPV, churner yang lolos)."""
    sel = [y for p, y in data if p <= t]
    n = len(sel)
    tn = sum(1 for y in sel if y == 0.0)
    return n, n / len(data), (tn / n if n else 0.0), n - tn


def wilson(k, n, z=1.96):
    """Selang kepercayaan 95% (Wilson) untuk proporsi k/n."""
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    a = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def make_tier_fn(t_hi, t_lo):
    def tier_of(p):
        if p >= t_hi:
            return "Pasti churn"
        if p <= t_lo:
            return "Pasti tidak churn"
        return "Akan churn"

    return tier_of


def tier_rows(data, tier_of):
    """Ringkasan per kelas: jumlah, porsi, churn aktual, selang kepercayaan, porsi churner."""
    total_pos = sum(1 for _, y in data if y == 1.0)
    rows = []
    for tier in TIER_ORDER:
        ys = [y for p, y in data if tier_of(p) == tier]
        n = len(ys)
        k = int(sum(ys))
        lo, hi = wilson(k, n)
        rows.append(
            {
                "tier": tier,
                "n": n,
                "share": n / len(data) if data else 0.0,
                "churners": k,
                "churn_rate": k / n if n else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
                "share_of_churners": k / total_pos if total_pos else 0.0,
            }
        )
    return rows


def deciles(data, bins=10):
    """Rata-rata skor vs churn aktual per desil skor (cek kalibrasi)."""
    ds = sorted(data, key=lambda r: r[0])
    size = len(ds) // bins
    out = []
    for b in range(bins):
        chunk = ds[b * size : (b + 1) * size] if b < bins - 1 else ds[b * size :]
        out.append(
            (
                b + 1,
                sum(p for p, _ in chunk) / len(chunk),
                sum(y for _, y in chunk) / len(chunk),
            )
        )
    return out