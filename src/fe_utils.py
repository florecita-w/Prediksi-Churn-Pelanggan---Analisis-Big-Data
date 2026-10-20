"""Fungsi bantu Python untuk rekayasa fitur (tanpa Spark, bisa diuji terpisah)."""

def find_pure_tails(rows, target=1, purity=0.999, min_n=500):
    """Cari 'ekor' nilai fitur yang hampir seluruhnya satu kelas.

    rows   : list (nilai, jumlah_baris, jumlah_churn), data TRAIN saja.
    target : 1 = cari ekor yang hampir semuanya churn, 0 = hampir semuanya tidak churn.
    Hasil  : (lo_max, hi_min). Nilai <= lo_max (ekor bawah) atau >= hi_min (ekor atas)
             'murni'. None bila tidak ada ekor dengan >= min_n baris.
    """
    rows = sorted(rows, key=lambda r: r[0])

    def hits(r):
        return r[2] if target == 1 else r[1] - r[2]

    def scan(seq):
        n = h = 0
        best = None
        for r in seq:
            n += r[1]
            h += hits(r)
            if n and h / n >= purity and n >= min_n:
                best = r[0]
        return best

    return scan(rows), scan(reversed(rows))