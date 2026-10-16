# Prediksi Churn Pelanggan Layanan Berlangganan

Proyek akhir mata kuliah **Analisis Big Data** - **Team 6**

Membangun pipeline data end-to-end (akuisisi, penyimpanan Parquet, persiapan data, EDA, pemodelan, evaluasi) untuk memprediksi pelanggan yang berisiko churn, lalu menyajikan skor risiko lewat dashboard CRM.

## Tim

| Nama | NIM | Peran |
|---|---|---|
| Adhelia Issabel | H1101241001 | Manajer Proyek & Arsitek Data |
| Aurellya Yocelyn Prasista | H1101241043 | Data Engineer (akuisisi, pipeline, penyimpanan) |
| Florecita Wenny | H1101241039 | Analis Data / ML Engineer (EDA, fitur, model, dasbor) |

Seluruh anggota bertanggung jawab atas QA, dokumentasi, dan kepatuhan (profil kualitas, lembar etika, laporan akhir).

## Sumber Data

| Peran | Dataset | Catatan |
|---|---|---|
| Primer | Customer Churn Dataset (Kaggle: `samuelsemaya/e-commerce-customer-churn`) | Penyedia asli: Muhammad Shahid Azeem. Asal data (nyata/sintetis) tidak dijelaskan pada deskripsinya |
| Sekunder | Marketing Campaign Performance Dataset (Kaggle, Manisha Bhattacharjee) | 200.000 baris, 16 kolom, CC0. Hanya konteks, tidak dapat digabung per pelanggan |
| Sekunder | Customer Service Interaction Dataset (Kaggle, Bhrami Zadafiya) | 10.000 baris, 13 kolom, CDLA-Sharing 1.0. Hanya konteks |

Data **tidak disimpan di repositori ini**. Setiap anggota mengunduhnya sendiri (lihat bagian Setup).

## Struktur Folder

```
.
├── data/
│   ├── raw/            # data mentah hasil unduhan (tidak di-commit)
│   ├── curated/        # data terkurasi, format Parquet
│   ├── model_ready/    # data siap model (encoded, standardisasi)
│   └── scored/         # hasil skor risiko churn untuk dashboard
├── notebooks/          # notebook per tahap (profil, EDA, model)
├── src/                # kode pipeline yang dapat dipakai ulang
├── dashboard/          # aplikasi dashboard CRM
├── docs/               # asal-usul data, lembar etika, diagram
├── reports/            # laporan per pertemuan
├── requirements.txt
└── README.md
```

## Alur Pipeline

`Unduh (Kaggle)` -> `raw` -> `curated (Parquet)` -> `persiapan data` -> `EDA` -> `rekayasa fitur` -> `model (Logistic Regression, Random Forest)` -> `evaluasi` -> `skor risiko` -> `Dashboard CRM`

Target utama: **recall minimal 70%**, dengan metrik pendukung Precision, F1-Score, ROC-AUC, dan PR-AUC.

## Setup

Yang **tidak** ada di repositori dan harus di-install di setiap perangkat: Python, Java, virtual environment beserta paketnya, serta token Kaggle dan data. Semuanya dijelaskan di bawah.

### 1. Prasyarat (instal sekali per perangkat)

| Kebutuhan | Versi disarankan | Cek |
|---|---|---|
| Python | 3.10 - 3.12 | `python --version` |
| Java (JDK) | 17 (wajib untuk PySpark) | `java -version` |
| Git | terbaru | `git --version` |
| VS Code | terbaru | - |

Tips: di Windows, centang **Add Python to PATH** saat memasang Python. Setelah memasang Java, tutup dan buka kembali VS Code agar terbaca.

### 2. Clone repositori

```bash
git clone <url-repositori>
cd <nama-folder-repositori>
```

### 3. Buat dan aktifkan virtual environment

```bash
python -m venv .venv
```

Aktifkan:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Jika PowerShell menolak menjalankan skrip, jalankan sekali: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Di VS Code: **Ctrl+Shift+P** -> **Python: Select Interpreter** -> pilih yang berada di `.venv`.

### 4. Instal paket

```bash
pip install -r requirements.txt
```

### 5. Siapkan folder data

Folder data kosong tidak ikut ter-clone kecuali berisi `.gitkeep`. Jika folder `data/...` belum ada di perangkat Anda:

```bash
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/raw','data/curated','data/model_ready','data/scored']]"
```

### 6. Pasang token Kaggle

Ambil API token di akun Kaggle Anda (**Settings** -> bagian **API**) dan ikuti petunjuk di halaman tersebut. Untuk format klasik, simpan `kaggle.json` di:

- Windows: `C:\Users\<nama>\.kaggle\kaggle.json`
- macOS / Linux: `~/.kaggle/kaggle.json` (lalu jalankan `chmod 600 ~/.kaggle/kaggle.json`)

**Jangan pernah meng-commit token ini.** File tersebut sudah dimasukkan ke `.gitignore`.

### 7. Unduh data primer

```bash
kaggle datasets download -d samuelsemaya/e-commerce-customer-churn -p data/raw --unzip
```

Jika CLI bermasalah, unduh manual dari halaman dataset di browser, ekstrak, lalu letakkan file di `data/raw/`.

Dataset sekunder diunduh dengan cara yang sama, masing-masing sebagai satu file CSV (cek slug pada tautan Kaggle-nya).

### 8. Verifikasi

```bash
python -c "import os; print(os.listdir('data/raw'))"
python -c "import pandas as pd, glob; f=glob.glob('data/raw/*.csv')[0]; df=pd.read_csv(f); print(f); print(df.shape); print(df.head()); print(df.dtypes)"
```

Catat nama file dan jumlah baris di `docs/provenance.md`.

## Alur Kerja Git

1. Selalu ambil perubahan terbaru sebelum mulai: `git pull`.
2. Buat branch per tugas: `git checkout -b nama/tugas` (contoh: `florecita/eda`).
3. Commit kecil dengan pesan jelas: `git commit -m "Tambah EDA distribusi churn"`.
4. Push lalu buka Pull Request ke `main`. Minta anggota lain meninjau sebelum merge.
5. **Jangan commit data, token, atau folder `.venv`.** Semuanya sudah diabaikan lewat `.gitignore`; jika Anda melihat file besar muncul di `git status`, hentikan dan periksa.

## Troubleshooting

| Masalah | Penyebab umum | Solusi |
|---|---|---|
| PySpark gagal start / `JAVA_HOME is not set` | Java belum terpasang atau tidak terbaca | Pasang JDK 17, atur `JAVA_HOME`, buka ulang VS Code |
| Error saat menulis Parquet dengan Spark di Windows (pesan terkait `winutils` / `HADOOP_HOME`) | Spark lokal di Windows butuh komponen Hadoop | Gunakan WSL, atau pasang winutils, atau simpan sementara dengan `pandas.to_parquet` |
| `kaggle: command not found` | venv belum aktif | Aktifkan `.venv` lalu ulangi |
| `401 Unauthorized` dari Kaggle | Token salah/belum terpasang | Buat ulang token dan pasang ulang sesuai langkah 6 |
| PowerShell menolak mengaktifkan venv | Execution policy | Lihat langkah 3 |

## Catatan Etika & Keterbatasan

- Asal-usul data primer tidak dijelaskan pada deskripsinya (nyata atau sintetis). Hal ini dicatat di `docs/provenance.md`.
- Dataset sekunder bersifat konteks dan tidak digabung ke data pelanggan primer.
- Pengujian bias (per gender dan kelompok usia) dilakukan dan didokumentasikan di lembar kepatuhan dan etika.