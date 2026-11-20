"""ทำนายคะแนนดาวของรีวิวใหม่ด้วยโมเดลที่เทรนไว้ (model.joblib)

ตัวอย่าง:
    python predict.py "อาหารอร่อยมาก บริการดี"
    python predict.py            # พิมพ์รีวิวทีละบรรทัด, บรรทัดว่างเพื่อออก
"""
import sys
from pathlib import Path

import joblib

from features import tokenize_review

MODEL_PATH = Path("model.joblib")


def load_pipeline(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบโมเดล: {path.resolve()} (รัน `python train.py` ก่อนเพื่อสร้างโมเดล)"
        )
    return joblib.load(path)


def predict_review(review, vectorizer, classifier):
    """คืนค่า (ดาวที่ทำนาย, dict ความน่าจะเป็นของแต่ละดาว)"""
    text = " ".join(tokenize_review(review))
    matrix = vectorizer.transform([text])
    probabilities = classifier.predict_proba(matrix)[0]
    best = int(probabilities.argmax())
    scores = {int(c): float(p) for c, p in zip(classifier.classes_, probabilities)}
    return int(classifier.classes_[best]), scores


def show(review, vectorizer, classifier):
    rating, scores = predict_review(review, vectorizer, classifier)
    print(f"ทำนาย: {rating} ดาว")
    for star, probability in scores.items():
        print(f"  {star} ดาว: {probability:.1%}")


def main():
    vectorizer, classifier = load_pipeline()

    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]), vectorizer, classifier)
        return

    while True:
        review = input("รีวิว> ").strip()
        if not review:
            break
        show(review, vectorizer, classifier)


if __name__ == "__main__":
    main()
