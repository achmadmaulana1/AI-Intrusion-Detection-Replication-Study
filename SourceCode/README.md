# AI Intrusion Detection Replication Study

Mahasiswa: Achmad Maulana  
NIM: 241730016  
Kelas: INF 4A  
Mata kuliah: Kecerdasan Buatan

## Deskripsi

Replication study paper IDS berbasis machine learning pada dataset UNSW-NB15.

## Artikel Utama

Enhanced Intrusion Detection Systems Performance with UNSW-NB15 Data Analysis (Algorithms, 2024). DOI: https://doi.org/10.3390/a17020064

## Dataset

UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Cara Instalasi

```bash
pip install -r SourceCode/requirements.txt
```

## Cara Menjalankan

```bash
python SourceCode/src/main.py --sample-size 200000
python SourceCode/src/stage7_bonus_experiments.py --sample-size 50000
```

## Notebook

Buka `Notebook/notebook_replikasi.ipynb` dan jalankan cell berurutan.

## Hasil Ringkas

Random Forest Selected Features memperoleh accuracy 0,9939, precision 0,9768, recall 0,9747, F1 0,9757, AUC 0,9998, FAR 0,0034.

## Error Umum

- Dataset tidak ditemukan: pastikan CSV ada di Dataset/raw.
- Memory error: turunkan sample-size.
- ModuleNotFoundError: jalankan pip install -r requirements.txt.

## Link Google Drive

(https://drive.google.com/drive/folders/1pE8rd0I8EoOm4Hh3mqFE3cyixRudAMHu?usp=sharing)

## Link GitHub

[https://github.com/achmadmaulana1/AI-Intrusion-Detection-Replication-Study.git]
