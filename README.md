# Analisis Sentimen Ulasan Aplikasi Disney+ (Google Play)

Proyek ini melakukan **analisis sentimen** terhadap ulasan pengguna aplikasi **Disney+** di Google Play Store.
Alur kerja: **scraping data -> preprocessing teks -> labeling -> ekstraksi fitur -> modeling -> evaluasi**.

> **Konteks tugas:** proyek contoh dari dosen memakai aplikasi **OVO** (`ovo.id`). Tugas meminta studi kasus diganti
> menjadi **Disney+** (`com.disney.disneyplus`). Dokumentasi ini menjelaskan seluruh perubahan dari versi OVO ke Disney+
> serta langkah menjalankan proyek dari nol, sehingga bisa diikuti ulang oleh orang lain (mis. teman satu kelas).

---

## 1. Ringkasan Perubahan: OVO -> Disney+

Perubahan dilakukan pada **dua notebook**:

1. `Scrapping/playstore/scrapping_with_playstore.ipynb` (pengambilan data)
2. `analisis-sentiment-reviews-ovo.ipynb` (analisis & pemodelan)

### 1.1 Notebook Scraping (`Scrapping/playstore/scrapping_with_playstore.ipynb`)

| No | Lokasi kode | Sebelum (OVO) | Sesudah (Disney+) |
|----|-------------|---------------|-------------------|
| 1 | Cell `result = app(...)` | `'ovo.id'` | `'com.disney.disneyplus'` |
| 2 | Cell `app_id = ...` | `app_id = "ovo.id"` | `app_id = "com.disney.disneyplus"` |
| 3 | Variabel `csv_filename` | `"reviews_ovo.csv"` | `"reviews_disneyplus.csv"` |
| 4 | Cell `pd.read_csv(...)` | `pd.read_csv("reviews_ovo.csv")` | `pd.read_csv("Scrapping/playstore/reviews_disneyplus.csv")` |

**Cara menemukan package id aplikasi lain:** buka halaman Play Store aplikasinya, lihat URL
`https://play.google.com/store/apps/details?id=com.disney.disneyplus`. Nilai setelah `id=` itulah yang dipakai.
Untuk Disney+ nilainya `com.disney.disneyplus`.

### 1.2 Notebook Analisis (`analisis-sentiment-reviews-ovo.ipynb`)

| No | Lokasi kode | Sebelum (OVO) | Sesudah (Disney+) |
|----|-------------|---------------|-------------------|
| 1 | Cell import paling atas | belum ada | ditambah `import torch` |
| 2 | Cell baca data | `df = pd.read_csv("./reviews_ovo.csv")` | `df = pd.read_csv("Scrapping/playstore/reviews_disneyplus.csv")` |
| 3 | Cell preprocessing | belum ada | ditambah `nltk.download("punkt_tab")` dan `nltk.download("stopwords")` |
| 4 | Cell simpan TF-IDF | `models/tfidf_vectorizer.pkl` | `tfidf_vectorizer.pkl` (disimpan di root proyek) |
| 5 | Setelah `train_test_split` | belum ada | **cell baru SMOTE** (menyeimbangkan kelas) |
| 6 | Model Naive Bayes | `model_nb.fit(X_train_tf, y_train_tf)` | `model_nb.fit(X_train_res, y_train_res)` |

**Isi cell SMOTE yang ditambahkan:**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tf, y_train_tf)
```

**Alasan perubahan:**
- Dataset Disney+ jauh lebih sedikit dibanding OVO, sehingga distribusi kelas sangat tidak seimbang
  (ulasan negatif mendominasi). SMOTE dipakai agar model tidak bias ke kelas mayoritas.
- `import torch` diperlukan oleh `transformers` (pipeline IndoBERT).
- `nltk.download(...)` memastikan resource tokenizer & stopword tersedia saat pertama kali dijalankan.
- Nama file `reviews_ovo.csv` diganti `reviews_disneyplus.csv` karena data yang diambil sekarang milik Disney+.

> Catatan: nama file notebook masih `analisis-sentiment-reviews-ovo.ipynb`. Nama ini **boleh dibiarkan**
> (isinya sudah Disney+), atau di-rename menjadi `analisis-sentiment-reviews-disneyplus.ipynb` bila ingin lebih rapi.

---

## 2. Struktur Proyek

```text
Sentiment_Analysis/
├── analisis-sentiment-reviews-ovo.ipynb   # analisis sentimen (data Disney+)
├── tfidf_vectorizer.pkl                   # hasil simpan TF-IDF (dibuat saat run)
├── README.md
├── .gitignore
└── Scrapping/
    ├── playstore/
    │   ├── scrapping_with_playstore.ipynb  # scraping data Disney+
    │   └── reviews_disneyplus.csv          # hasil scraping (dibuat saat run)
    ├── twitter/
    │   └── scraping_with_x.ipynb           # eksperimen scraping X/Twitter (opsional)
    └── TikTok-Api/                          # folder eksperimen (opsional)
```

---

## 3. Prasyarat

Install terlebih dahulu:

- **Python 3.10 – 3.12** (disarankan 3.12; TensorFlow belum tentu mendukung versi lebih baru).
  Cek dengan:
  ```powershell
  python --version
  ```
- **Git** (untuk clone/push).
- **Jupyter** (bisa lewat `pip` atau ekstensi Jupyter di VS Code).
- Koneksi internet (untuk scraping, unduh resource NLTK, dan unduh model Transformers/Torch).

---

## 4. Langkah 1 — Ambil Proyek

### Jika menyalin dari GitHub dosen (clone)

```powershell
git clone https://github.com/Sintasitinuriah/Sentiment_Analysis.git
cd Sentiment_Analysis
```

### Jika sudah punya folder proyek

Cukup buka foldernya di terminal/VS Code, lalu lanjut ke Langkah 2.

---

## 5. Langkah 2 — Lakukan Perubahan OVO -> Disney+

Ikuti tabel pada **Bagian 1**. Ringkasnya:

1. Buka `Scrapping/playstore/scrapping_with_playstore.ipynb`.
   - Ganti `'ovo.id'` menjadi `'com.disney.disneyplus'` (2 tempat).
   - Ganti `"reviews_ovo.csv"` menjadi `"reviews_disneyplus.csv"`.
   - Ganti cell pembacaan CSV menjadi `pd.read_csv("Scrapping/playstore/reviews_disneyplus.csv")`.
2. Buka `analisis-sentiment-reviews-ovo.ipynb`.
   - Ganti `pd.read_csv("./reviews_ovo.csv")` menjadi `pd.read_csv("Scrapping/playstore/reviews_disneyplus.csv")`.
   - Tambahkan `import torch` di cell import.
   - Tambahkan `nltk.download("punkt_tab")` dan `nltk.download("stopwords")` di awal cell preprocessing.
   - Ubah path simpan TF-IDF menjadi `tfidf_vectorizer.pkl`.
   - Tambahkan cell SMOTE dan ubah `model_nb.fit(...)` menjadi `fit(X_train_res, y_train_res)`.

> **Penting soal path:** contoh di atas memakai **path relatif** dari root proyek agar portabel.
> Jika notebook Anda masih memakai path absolut (mis. `D:/KULIAH/...`), ganti dengan path relatif di atas
> supaya bisa dijalankan di komputer lain.

---

## 6. Langkah 3 — Buat Virtual Environment (venv)

Jalankan dari root proyek (`Sentiment_Analysis/`):

```powershell
python -m venv .venv
```

Perintah ini membuat folder `.venv/` berisi Python & pip khusus proyek (terisolasi dari sistem).

---

## 7. Langkah 4 — Aktifkan venv

Pilih sesuai terminal yang dipakai:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

Jika muncul error `running scripts is disabled on this system`, izinkan sekali saja:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
lalu ulangi perintah aktivasi.

**Windows CMD**
```cmd
.venv\Scripts\activate.bat
```

**Git Bash**
```bash
source .venv/Scripts/activate
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

Jika berhasil, prompt terminal diawali `(.venv)`. Cek dengan:
```powershell
python --version
where python
```
Pastikan yang muncul adalah `...\Sentiment_Analysis\.venv\Scripts\python.exe`.

---

## 8. Langkah 5 — Install Library yang Dibutuhkan

Upgrade pip dulu:

```powershell
python -m pip install --upgrade pip
```

Install library utama:

```powershell
python -m pip install numpy pandas scikit-learn matplotlib seaborn wordcloud nltk emoji joblib imbalanced-learn google-play-scraper transformers torch jupyter
```

Install library yang lebih berat (Word2Vec & Deep Learning) secara terpisah:

```powershell
python -m pip install gensim tensorflow
```

**Penjelasan library:**

| Library | Kegunaan |
|---------|----------|
| `google-play-scraper` | Mengambil ulasan dari Google Play Store |
| `pandas`, `numpy` | Pengolahan & manipulasi data |
| `nltk` | Tokenisasi dan stopword |
| `emoji` | Menangani emoji pada teks |
| `scikit-learn` | TF-IDF, train/test split, Naive Bayes, Logistic Regression, MLP, metrik |
| `imbalanced-learn` | SMOTE untuk menyeimbangkan kelas |
| `gensim` | Word2Vec |
| `tensorflow` (termasuk Keras) | Model CNN-LSTM |
| `transformers`, `torch` | Pipeline IndoBERT (labeling dengan BERT) |
| `matplotlib`, `seaborn`, `wordcloud` | Visualisasi grafik & word cloud |
| `joblib` | Menyimpan model `.pkl` |
| `jupyter` | Menjalankan notebook |

Opsional, menyimpan daftar dependency:

```powershell
python -m pip freeze > requirements.txt
```
Orang lain bisa meng-install ulang dengan:
```powershell
python -m pip install -r requirements.txt
```

---

## 9. Langkah 6 — Unduh Resource NLTK

Bisa lewat notebook (cell preprocessing sudah berisi `nltk.download`), atau manual:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

---

## 10. Langkah 7 — Siapkan Folder `models/`

Beberapa model disimpan ke folder `models/`, jadi buat foldernya lebih dulu:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
```
(Linux/macOS: `mkdir -p models`)

---

## 11. Langkah 8 — Jalankan Scraping Data Disney+

1. Jalankan Jupyter dari root proyek:
   ```powershell
   python -m jupyter notebook
   ```
   (atau buka folder proyek di VS Code, buka notebook, lalu pilih kernel `.venv\Scripts\python.exe`).
2. Buka `Scrapping/playstore/scrapping_with_playstore.ipynb`.
3. Jalankan cell dari atas ke bawah.
4. Pastikan terbentuk file `Scrapping/playstore/reviews_disneyplus.csv`.

> Jika CSV tersimpan di lokasi lain (karena working directory Jupyter berbeda), sesuaikan path pada
> notebook analisis di Langkah 9.

---

## 12. Langkah 9 — Jalankan Analisis Sentimen

1. Buka `analisis-sentiment-reviews-ovo.ipynb`.
2. Pastikan kernel yang dipilih adalah `.venv`.
3. Jalankan cell **berurutan dari atas**:
   - Import & baca data
   - EDA & preprocessing
   - Labeling rating -> sentimen
   - Ekstraksi fitur TF-IDF & Word2Vec
   - Split data + SMOTE
   - Model: Naive Bayes, Logistic Regression, MLP, CNN-LSTM
   - Evaluasi: accuracy, classification report, confusion matrix
   - Uji prediksi teks baru
4. Output yang diharapkan:
   - `tfidf_vectorizer.pkl` di root proyek
   - `model_w2v.model` (Word2Vec)
   - Model di folder `models/` (`model_naive_bayes.pkl`, `model_logistic_regression.pkl`, `model_mlp.pkl`, `cnn_lstm_model.h5`, dll.)

---

## 13. Langkah 10 — Push ke GitHub Milik Sendiri


### 13.1 Buat repository baru

1. Buka GitHub -> **New repository**.
2. Beri nama (mis. `Sentiment_Analysis_Disney`).
3. **Jangan** centang "Add a README/.gitignore/license" (biarkan kosong) agar tidak konflik.
4. Salin URL repo, mis. `https://github.com/<username>/Sentiment_Analysis_Disney.git`.

### 13.2 Ganti remote `origin`

Di root proyek:

```powershell
git remote -v
git remote set-url origin https://github.com/<username>/Sentiment_Analysis_Disney.git
git remote -v
```

**Alternatif (menyimpan repo dosen sebagai `upstream`):**

```powershell
git remote rename origin upstream
git remote add origin https://github.com/<username>/Sentiment_Analysis_Disney.git
```

### 13.3 Commit & push

Pastikan `.venv/` tidak ikut ter-commit (sudah diatur di `.gitignore`):

```powershell
git status
git add .
git commit -m "Analisis sentimen Disney+ (perubahan dari studi kasus OVO)"
git push -u origin main
```

### 13.4 Autentikasi

- **GitHub CLI (disarankan):**
  ```powershell
  gh auth login
  ```
- **HTTPS biasa:** gunakan **Personal Access Token (PAT)** sebagai password, bukan password akun.
  Buat di GitHub -> Settings -> Developer settings -> Personal access tokens.
- **SSH:** tambahkan SSH key ke GitHub, lalu pakai URL `git@github.com:<username>/<repo>.git`.

### 13.5 Jika memakai fork

Buka repo dosen di GitHub -> tombol **Fork** -> clone hasil fork -> push perubahan ke fork Anda.
Cara ini otomatis menyimpan hubungan ke repo asal.

---

## 14. Troubleshooting

| Masalah | Penyebab | Solusi |
|---------|----------|--------|
| `ModuleNotFoundError: No module named 'tensorflow'` / `gensim` | Library belum ter-install di venv | `python -m pip install tensorflow gensim` |
| `NameError: name 'label_sentimen_duafile' is not defined` | Cell "Label dengan BERT" memanggil fungsi yang tidak ada (sisa versi OVO) | **Lewati/comment** cell tersebut, atau ganti dengan `analyze_sentiment` yang sudah didefinisikan di cell sebelumnya |
| `FileNotFoundError: reviews_disneyplus.csv` | Path CSV salah / scraping belum dijalankan | Jalankan notebook scraping, lalu samakan path di notebook analisis |
| Error shape pada CNN-LSTM (`expected ... 3 dimensions`) | Input Word2Vec masih 2D `(n, 100)`, sedangkan `Conv1D` butuh `(n, 100, 1)` | Tambahkan sebelum `fit`/`predict`: `X_train_w2v = X_train_w2v.reshape(-1, 100, 1)`, `X_test_w2v = X_test_w2v.reshape(-1, 100, 1)`, dan reshape juga `X_new_w2v` saat prediksi |
| `FileNotFoundError: models/...` | Folder `models/` belum ada | `New-Item -ItemType Directory -Force models` |
| `Set-ExecutionPolicy` error saat aktivasi | Policy PowerShell membatasi script | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Tokenizer NLTK error (`punkt`/`punkt_tab` belum ada) | Resource belum diunduh | Jalankan langkah unduh NLTK (Langkah 6) |
| Scraping lambat / data sedikit | Disney+ punya lebih sedikit ulasan berbahasa Indonesia dibanding OVO | Wajar; tunggu proses `reviews_all` selesai |

---

## 15. Catatan Reproducibility

- Selalu aktifkan `.venv` sebelum menjalankan notebook.
- Gunakan path relatif, hindari path absolut seperti `D:/KULIAH/...` agar bisa dijalankan di komputer lain.
- Jangan commit folder `.venv/` (sudah masuk `.gitignore`).
- Simpan `requirements.txt` bila ingin hasil instalasi identik.
- Notebook berisi output lama dapat membingungkan; setelah menjalankan ulang, simpan ulang notebook
  (**Save**) agar output sesuai dataset Disney+.
