# Naive-Bayes Wongnai

ทำนายคะแนนดาว (1–5) จากรีวิวร้านอาหารภาษาไทยของ [Wongnai](https://github.com/wongnai/wongnai-corpus) ด้วยอัลกอริทึม Naive Bayes

โปรเจกต์มี 2 ส่วนที่ทำงานแยกกัน:

| ส่วน | ไฟล์ | รายละเอียด |
|------|------|-----------|
| Naive Bayes เขียนเองด้วย numpy | `main.py` | ตัดคำ, นับความถี่, คำนวณความน่าจะเป็น, วัดความแม่นยำ และสร้าง heatmap / wordcloud ของคำเด่นแต่ละดาว |
| scikit-learn (TF-IDF + MultinomialNB) | `load_data.py`, `features.py`, `train.py`, `predict.py` | โหลดข้อมูล, สร้างฟีเจอร์, เทรนโมเดล, รายงานผลรายดาว และบันทึกโมเดลไว้ใช้ทำนาย |

## โครงสร้างไฟล์

```
.
├── main.py           # Naive Bayes แบบเขียนเอง + กราฟวิเคราะห์คำ
├── load_data.py      # โหลดและทำความสะอาดข้อมูลรีวิว (ดาวน์โหลดอัตโนมัติถ้ายังไม่มี)
├── features.py       # ตัดคำไทย, stopwords, TF-IDF
├── train.py          # เทรน MultinomialNB และบันทึก model.joblib
├── predict.py        # โหลดโมเดลแล้วทำนายรีวิวใหม่
└── requirements.txt
```

## การติดตั้ง

ต้องใช้ Python 3.9 ขึ้นไป

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## วิธีใช้งาน

ข้อมูลจะถูกดาวน์โหลดไปที่ `data/w_review_train.csv` อัตโนมัติในการรันครั้งแรก

**1. รัน Naive Bayes แบบเขียนเอง (สรุปผล + กราฟ)**

```bash
python main.py
```

ผลลัพธ์จะอยู่ในโฟลเดอร์ `output/` ได้แก่ `summary.txt`, `test_predictions.csv`, `distinctive_words_heatmap.png` และ `wordcloud.png`

**2. เทรนโมเดลด้วย scikit-learn**

```bash
python train.py
```

พิมพ์ accuracy, classification report และ confusion matrix จากนั้นบันทึกโมเดลเป็น `model.joblib`

**3. ทำนายรีวิวใหม่**

```bash
python predict.py "อาหารอร่อยมาก บริการดี ราคาไม่แพง"
```

ถ้าไม่ใส่ข้อความ โปรแกรมจะให้พิมพ์รีวิวทีละบรรทัด (พิมพ์บรรทัดว่างเพื่อออก)

## หมายเหตุ

- `main.py` ใช้ฟอนต์ไทยแสดงในกราฟ (Arial Unicode บน macOS หรือ Tahoma บน Windows) ถ้าไม่พบฟอนต์ กราฟอาจแสดงภาษาไทยเป็นสี่เหลี่ยม
- ไฟล์ `data/`, `output/` และ `*.joblib` ถูกตั้งค่าไว้ใน `.gitignore` จึงไม่ถูกอัปโหลดขึ้น repo
