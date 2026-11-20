from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


THAI_STOPWORDS = thai_stopwords()
EXTRA_STOPWORDS = {"ร้าน", "กิน", "อร่อย", "นะ", "คะ", "ค่ะ", "ก็", "และ", "ใน"}


def clean_review(review):
    return str(review).strip().lower()


def tokenize_review(review):
    cleaned_review = clean_review(review)
    words = word_tokenize(cleaned_review, engine="newmm")
    return [
        word
        for word in words
        if len(word.strip()) > 1
        and word not in THAI_STOPWORDS
        and word not in EXTRA_STOPWORDS
    ]


def build_corpus(dataframe):
    return [" ".join(tokenize_review(review)) for review in dataframe["review"]]


def get_tfidf():
    return TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
