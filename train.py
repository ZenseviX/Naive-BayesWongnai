import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import re
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from load_data import load_data
from features import build_corpus, get_tfidf


# ใช้ฟอนต์ Windows ที่รองรับภาษาไทยเพื่อให้กราฟแสดงข้อความได้ครบ
WINDOWS_THAI_FONT = r"C:\Windows\Fonts\tahoma.ttf"
if os.path.exists(WINDOWS_THAI_FONT):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(WINDOWS_THAI_FONT)
    thai_font_name = font_manager.FontProperties(
        fname=WINDOWS_THAI_FONT
    ).get_name()
    plt.rcParams["font.family"] = thai_font_name
    sns.set_theme(font=thai_font_name)


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


def save_evaluation_report(classifier, validation_matrix, y_validation, output_dir="output"):
    """บันทึกตารางเปรียบเทียบผลลัพธ์ กราฟ Confusion Matrix และกราฟ Accuracy รายดาว"""
    predictions = classifier.predict(validation_matrix)
    labels = [1, 2, 3, 4, 5]
    report = classification_report(
        y_validation,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ตารางความแม่นยำรายระดับดาว
    metrics_rows = []
    for label in labels:
        row = report[str(label)]
        metrics_rows.append({
            "Rating": f"{label} ดาว",
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1_score": row["f1-score"],
            "Support": int(row["support"]),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    overall_row = pd.DataFrame([{
        "Rating": "Overall",
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1_score": report["weighted avg"]["f1-score"],
        "Support": int(report["weighted avg"]["support"]),
    }])
    metrics_df = pd.concat([metrics_df, overall_row], ignore_index=True)
    metrics_df.to_csv(
        output_path / "model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ตารางเปรียบเทียบจำนวนรีวิวจริงและที่ทำนายถูก
    comparison_rows = []
    matrix = confusion_matrix(y_validation, predictions, labels=labels)
    for index, label in enumerate(labels):
        actual_count = int(matrix[index].sum())
        correct_count = int(matrix[index, index])
        predicted_count = int(matrix[:, index].sum())
        comparison_rows.append({
            "Rating": f"{label} ดาว",
            "Actual_Count": actual_count,
            "Predicted_As_This_Rating": predicted_count,
            "Correct_Predictions": correct_count,
            "Class_Accuracy": correct_count / actual_count if actual_count else 0,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        output_path / "rating_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # กราฟ Confusion Matrix
    fig, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"{label} ดาว" for label in labels],
        yticklabels=[f"{label} ดาว" for label in labels],
        ax=axis,
    )
    axis.set_title("Confusion Matrix: Rating Prediction", fontsize=14, fontweight="bold")
    axis.set_xlabel("Predicted Rating")
    axis.set_ylabel("Actual Rating")
    fig.tight_layout()
    fig.savefig(output_path / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # กราฟเปรียบเทียบ Precision, Recall และ F1 รายดาว
    class_metrics = metrics_df.iloc[:-1].copy()
    x = range(len(class_metrics))
    fig, axis = plt.subplots(figsize=(11, 6))
    width = 0.24
    axis.bar(
        [value - width for value in x],
        class_metrics["Precision"],
        width=width,
        label="Precision",
        color="#4472C4",
    )
    axis.bar(
        list(x),
        class_metrics["Recall"],
        width=width,
        label="Recall",
        color="#70AD47",
    )
    axis.bar(
        [value + width for value in x],
        class_metrics["F1_score"],
        width=width,
        label="F1-score",
        color="#ED7D31",
    )
    axis.set_xticks(list(x), class_metrics["Rating"])
    axis.set_ylim(0, 1)
    axis.set_title("Model Performance by Star Rating", fontsize=14, fontweight="bold")
    axis.set_xlabel("Actual Star Rating")
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    for bar_group in axis.containers:
        axis.bar_label(bar_group, fmt="%.2f", padding=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path / "metrics_by_rating.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    overall_accuracy = accuracy_score(y_validation, predictions)

    # ตารางเปรียบเทียบสองแบบ Naive Bayes: เขียนเอง และ scikit-learn
    custom_accuracy = None
    custom_summary_path = output_path / "summary.txt"
    if custom_summary_path.exists():
        summary_text = custom_summary_path.read_text(encoding="utf-8")
        match = re.search(r"ความแม่นยำ \(Accuracy\)\s*:\s*([0-9.]+)%", summary_text)
        if match:
            custom_accuracy = float(match.group(1)) / 100

    model_comparison_rows = [{
        "Model": "scikit-learn TF-IDF + MultinomialNB",
        "Accuracy": overall_accuracy,
        "Accuracy_Percent": overall_accuracy * 100,
        "Implementation": "train.py",
    }]
    if custom_accuracy is not None:
        model_comparison_rows.append({
            "Model": "NumPy + custom Naive Bayes",
            "Accuracy": custom_accuracy,
            "Accuracy_Percent": custom_accuracy * 100,
            "Implementation": "main.py",
        })

    pd.DataFrame(model_comparison_rows).to_csv(
        output_path / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if len(model_comparison_rows) > 1:
        comparison_fig, comparison_axis = plt.subplots(figsize=(9, 5))
        model_names = [row["Model"] for row in model_comparison_rows]
        accuracies = [row["Accuracy_Percent"] for row in model_comparison_rows]
        bars = comparison_axis.barh(model_names, accuracies, color=["#4472C4", "#70AD47"])
        comparison_axis.set_xlim(0, 100)
        comparison_axis.set_xlabel("Accuracy (%)")
        comparison_axis.set_title("Comparison of Naive Bayes Models", fontsize=14, fontweight="bold")
        comparison_axis.grid(axis="x", alpha=0.25)
        comparison_axis.set_axisbelow(True)
        for bar, value in zip(bars, accuracies):
            comparison_axis.text(
                value + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
        comparison_fig.tight_layout()
        comparison_fig.savefig(
            output_path / "model_comparison.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(comparison_fig)

    return overall_accuracy, metrics_df, comparison_df


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
    overall_accuracy, metrics_df, comparison_df = save_evaluation_report(
        classifier,
        validation_matrix,
        y_validation,
    )

    print("Validation accuracy:", round(accuracy * 100, 2), "%")
    print_detailed_report(classifier, validation_matrix, y_validation)
    save_pipeline(vectorizer, classifier)
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
