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
| Primer | Customer Churn Dataset (Kaggle: `muhammadshahidazeem/customer-churn-dataset`) | Penyedia: Muhammad Shahid Azeem. Lisensi GPL 2. Dua file: training 440.833 baris dan testing 64.374 baris, masing-masing 12 kolom. Penyedia menyatakan data kemungkinan simulasi/anonim |
| Sekunder | Marketing Campaign Performance Dataset (Kaggle: `manishabhatt22/marketing-campaign-performance-dataset`) | Penyedia: Manisha Bhattacharjee. 200.000 baris, 16 kolom, CC0. Dinyatakan fiktif oleh penyedia. Hanya konteks, tidak dapat digabung per pelanggan |
| Sekunder | Customer Service Interaction Dataset (Kaggle: `zadafiyabhrami/customer-service-ineraction-dataset`) | Penyedia: Bhrami Zadafiya. 10.000 baris, 13 kolom, CDLA-Sharing 1.0. Dinyatakan sintetis oleh penyedia. Hanya konteks. Ejaan `ineraction` pada alamat memang demikian, jangan dikoreksi |

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

## Catatan Penting Data Primer

Baca sebelum menulis kode apa pun, supaya tidak mengulang kesalahan yang sudah ditemukan di Bab IV dan V laporan:

- **`CustomerID` tidak dipakai sebagai fitur.** ID tidak konsisten antara file training dan testing, dan pada training berkorelasi kuat dengan `Churn` (r = -0,84) karena terkait urutan baris. Model memakai 10 fitur lain.
- **Urutan baris training tidak acak.** Baris-baris awal hampir seluruhnya berstatus churn. Acak data (`shuffle`) sebelum membagi data, dan jangan memakai `head()` sebagai cuplikan yang dianggap mewakili data.
- **Training memiliki 1 baris kosong total** (baris ke-199.297 pada file). Buang baris ini sebelum mengonversi tipe data.
- **Training dan testing berbeda sebaran** (proporsi churn 56,71% dan 47,37%). Training dibagi menjadi train-fit (80%) dan train-validation (20%, *stratified*). Data testing tidak disentuh sampai evaluasi akhir, dan hasil utama dilaporkan dari testing.

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

### 7. Unduh data

Data primer (satu unduhan berisi dua file CSV: training dan testing):

```bash
kaggle datasets download -d muhammadshahidazeem/customer-churn-dataset -p data/raw --unzip
```

Data sekunder (masing-masing satu file CSV):

```bash
kaggle datasets download -d manishabhatt22/marketing-campaign-performance-dataset -p data/raw --unzip
kaggle datasets download -d zadafiyabhrami/customer-service-ineraction-dataset -p data/raw --unzip
```

Jika CLI bermasalah, unduh manual dari halaman dataset di browser, ekstrak, lalu letakkan file di `data/raw/`.

### 8. Verifikasi

```bash
python -c "import os; print(sorted(os.listdir('data/raw')))"
python -c "import pandas as pd, glob; [print(f, pd.read_csv(f).shape) for f in sorted(glob.glob('data/raw/*.csv'))]"
```

Hasilnya harus cocok dengan tabel berikut:

| File | Ukuran yang diharapkan (baris, kolom) |
|---|---|
| `customer_churn_dataset-training-master.csv` | (440833, 12), termasuk 1 baris kosong total |
| `customer_churn_dataset-testing-master.csv` | (64374, 12) |
| `marketing_campaign_dataset.csv` | (200000, 16) |
| `customer_service_interactions.csv` | (10000, 13) |

Jika ukuran berbeda, kemungkinan dataset yang terunduh salah atau versinya berubah. Hentikan dan periksa sebelum melanjutkan.

Catat nama file, jumlah baris, dan **tanggal unduh** di `docs/provenance.md`.

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

- Penyedia data primer menyatakan data kemungkinan bersifat simulasi/anonim, sehingga keaslian terhadap data pelanggan nyata tidak dapat diverifikasi secara independen. Hal ini dicatat di `docs/provenance.md`.
- Kedua dataset sekunder dinyatakan fiktif/sintetis oleh penyedianya. Keduanya bersifat konteks dan tidak digabung ke data pelanggan primer.
- Pengujian bias (per gender dan kelompok usia) direncanakan dan akan didokumentasikan di lembar kepatuhan dan etika (Pertemuan 15).
