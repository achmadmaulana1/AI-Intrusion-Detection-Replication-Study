from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REQUIRED_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
    "NUSW-NB15_features.csv",
]


def existing_files() -> list[str]:
    return [name for name in REQUIRED_FILES if (RAW_DIR / name).exists()]


def try_kaggle_download(dataset: str) -> bool:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        print("Kaggle CLI tidak ditemukan. Lewati download Kaggle otomatis.")
        return False

    print(f"Mencoba download via Kaggle dataset: {dataset}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        kaggle,
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(RAW_DIR),
        "--unzip",
    ]
    result = subprocess.run(cmd, text=True)
    return result.returncode == 0


def try_kagglehub_download(dataset: str) -> bool:
    try:
        import kagglehub
    except Exception:
        print("kagglehub tidak tersedia. Lewati download kagglehub otomatis.")
        return False

    print(f"Mencoba download via kagglehub dataset: {dataset}")
    try:
        downloaded = Path(kagglehub.dataset_download(dataset))
    except Exception as exc:
        print(f"Download kagglehub gagal: {exc}")
        return False

    copied = 0
    for name in REQUIRED_FILES:
        matches = list(downloaded.rglob(name))
        if matches:
            target = RAW_DIR / name
            if not target.exists():
                shutil.copy2(matches[0], target)
            copied += 1
    print(f"kagglehub menemukan {copied}/{len(REQUIRED_FILES)} file wajib.")
    return copied == len(REQUIRED_FILES)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare UNSW-NB15 dataset files for the replication pipeline."
    )
    parser.add_argument(
        "--kaggle-dataset",
        default="mrwellsdavid/unsw-nb15",
        help="Kaggle dataset slug used if Kaggle CLI credentials are available.",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    found = existing_files()
    if len(found) == len(REQUIRED_FILES):
        print("Semua file dataset wajib sudah tersedia:")
        for name in found:
            print(f"- {RAW_DIR / name}")
        return 0

    print("File dataset belum lengkap.")
    print("Ditemukan:")
    for name in found:
        print(f"- {name}")
    print("Belum ditemukan:")
    for name in REQUIRED_FILES:
        if name not in found:
            print(f"- {name}")

    ok = try_kagglehub_download(args.kaggle_dataset)
    if ok and len(existing_files()) == len(REQUIRED_FILES):
        print("Download kagglehub berhasil dan semua file wajib tersedia.")
        return 0

    ok = try_kaggle_download(args.kaggle_dataset)
    if ok and len(existing_files()) == len(REQUIRED_FILES):
        print("Download Kaggle berhasil dan semua file wajib tersedia.")
        return 0

    print()
    print("Download otomatis belum berhasil.")
    print("Silakan unduh dataset UNSW-NB15 dari sumber resmi berikut:")
    print("https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print()
    print("Letakkan file berikut di folder:")
    print(RAW_DIR)
    for name in REQUIRED_FILES:
        print(f"- {name}")
    print()
    print("Catatan: halaman resmi UNSW mengarah ke SharePoint/OneDrive. Jika meminta login,")
    print("gunakan browser untuk mengunduh file CSV lalu jalankan ulang pipeline.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
