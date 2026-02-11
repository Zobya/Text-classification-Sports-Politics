# Final .py file structured 

#Imports:
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Loading data
def load_data():
    documents = []
    labels = []

    # 1 = Sports
    # 0 = Politics

    with open("data/sports.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(line.strip())
                labels.append(1)

    with open("data/politics.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(line.strip())
                labels.append(0)

    return documents, labels

# main function
def main():

    print("Loading dataset...")
    documents, labels = load_data()
    print(f"Total samples: {len(documents)}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        documents,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # TF-IDF (unigrams only)
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    results = {}

    # Model 1 — Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    y_pred_nb = nb.predict(X_test_tfidf)
    results["Naive Bayes"] = accuracy_score(y_test, y_pred_nb)

    # Model 2 — Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_test_tfidf)
    results["Logistic Regression"] = accuracy_score(y_test, y_pred_lr)

    # Model 3 — Linear SVM
    svm = LinearSVC()
    svm.fit(X_train_tfidf, y_train)
    y_pred_svm = svm.predict(X_test_tfidf)
    results["Linear SVM"] = accuracy_score(y_test, y_pred_svm)

    print("\nModel Comparison Results")
    print("-" * 30)
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")


if __name__ == "__main__":
    main()
