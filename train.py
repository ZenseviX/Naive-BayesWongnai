import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from load_data import load_data
from features import build_corpus, get_tfidf


def prepare_dataset(test_size=0.2, seed=42):
    df = load_data()
    X = build_corpus(df)
    y = df["rating"]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def vectorize_reviews(X_train, X_validation):
    vectorizer = get_tfidf()
    train_matrix = vectorizer.fit_transform(X_train)
    validation_matrix = vectorizer.transform(X_validation)
    return vectorizer, train_matrix, validation_matrix


def fit_classifier(train_matrix, y_train):
    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(train_matrix, y_train)
    return classifier


def evaluate_model(classifier, validation_matrix, y_validation):
    return classifier.score(validation_matrix, y_validation)


def print_detailed_report(classifier, validation_matrix, y_validation):
    """พิมพ์ precision/recall/F1 รายดาว และ confusion matrix"""
    predictions = classifier.predict(validation_matrix)
    print("\nClassification report:")
    print(classification_report(y_validation, predictions, zero_division=0))
    print("Confusion matrix (แถว = ดาวจริง, คอลัมน์ = ดาวที่ทำนาย):")
    print(confusion_matrix(y_validation, predictions, labels=classifier.classes_))


def save_pipeline(vectorizer, classifier, output_path="model.joblib"):
    joblib.dump((vectorizer, classifier), output_path)


def main():
    X_train, X_validation, y_train, y_validation = prepare_dataset()
    vectorizer, train_matrix, validation_matrix = vectorize_reviews(
        X_train,
        X_validation,
    )
    classifier = fit_classifier(train_matrix, y_train)
    accuracy = evaluate_model(classifier, validation_matrix, y_validation)

    print("Validation accuracy:", round(accuracy, 4))
    print_detailed_report(classifier, validation_matrix, y_validation)
    save_pipeline(vectorizer, classifier)
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
