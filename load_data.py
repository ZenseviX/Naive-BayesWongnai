import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
import csv

DATA_PATH = Path("w_review_train.csv")
FALLBACK_DATA_PATH = Path("data/w_review_train.csv")
DATA_URL = "https://raw.githubusercontent.com/wongnai/wongnai-corpus/master/review/review_dataset.zip"


def download_dataset():
    data_directory = FALLBACK_DATA_PATH.parent
    data_directory.mkdir(parents=True, exist_ok=True)
    zip_path = data_directory / "review_dataset.zip"

    print("Downloading Wongnai review dataset...")
    urllib.request.urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(FALLBACK_DATA_PATH.name, data_directory)

    zip_path.unlink()
    return FALLBACK_DATA_PATH


def load_data(path=None):
    """โหลดข้อมูลรีวิว คืนค่า DataFrame ที่มีคอลัมน์ review, rating"""
    if path is None:
        path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    else:
        path = Path(path)

    if not path.exists():
        if path == FALLBACK_DATA_PATH:
            path = download_dataset()
        else:
            raise FileNotFoundError(
                f"ไม่พบไฟล์ข้อมูล: {path.resolve()} "
                "(วาง w_review_train.csv ที่โฟลเดอร์โปรเจกต์หรือ data/w_review_train.csv)"
            )

    path_string = str(path)
    if path_string.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path_string, header=None)
    else:
        csv.field_size_limit(10 ** 9)
        rows = []
        with open(path, encoding="utf-8", newline="") as data_file:
            for row in csv.reader(data_file, delimiter=";"):
                if len(row) >= 2:
                    rows.append(row[:2])
        df = pd.DataFrame(rows)

    df = df.iloc[:, :2]
    df.columns = ["review", "rating"]

    # ตัดแถวหัวตาราง/แถวที่ rating ไม่ใช่ตัวเลขออก
    df = df[pd.to_numeric(df["rating"], errors="coerce").notna()].copy()
    df["rating"] = df["rating"].astype(int)
    df["review"] = df["review"].astype(str)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.head())
    print(df["rating"].value_counts().sort_index())
