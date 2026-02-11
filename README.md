# Sports vs Politics Text Classification

## Project Overview
This project implements a text classification system to classify news articles as either **Sports** or **Politics** using classical Machine Learning techniques.

The models compared in this project:
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

TF-IDF was used for feature representation.

---

## Dataset

The dataset was derived from the BBC News Archive dataset.

- Sports articles: 511
- Politics articles: 417
- Total samples: 928

The dataset was split using stratified train-test splitting:
- 80% Training
- 20% Testing

---

## Methodology

1. The dataset was loaded and labeled (1 = Sports, 0 = Politics).
2. Stratified train-test split (80/20) was applied.
3. Text data was converted to TF-IDF feature vectors.
4. Three machine learning models were trained and evaluated:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM
5. Performance was evaluated using accuracy, precision, recall, and F1-score.

---

## Feature Representation

TF-IDF (Term Frequency – Inverse Document Frequency) was used to convert text into numerical feature vectors.

Experiments were conducted using:
- Unigrams (1,1)
- Unigrams + Bigrams (1,2)

---

## Models Compared

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM

---

## Results

| Model | Accuracy |
|-------|----------|
| Naive Bayes | 99.46% |
| Logistic Regression | 99.46% |
| Linear SVM | 100% |

Linear SVM achieved the highest accuracy.

---

## Project Structure

```
SportsPoliticsClassifier/
│
├── classifier.py # Main classification script
├── experiment.ipynb # Development notebook with experiments
├── requirements.txt # Python dependencies
├── data/
│ ├── sports.txt
│ └── politics.txt
└── README.md
```
---

## Dependencies

This project requires:

- Python 3.8 or higher
- pandas
- scikit-learn

Install dependencies using:

pip install pandas scikit-learn

---
## How to Run

1. Clone the repository:

git clone <your-repo-link>

2. Navigate to the project directory:

cd SportsPoliticsClassifier

3. Install required packages:

pip install pandas scikit-learn

4. Run the classifier:

python classifier.py

---

## Limitations

- The dataset is derived from a single news source (BBC), which may limit generalization.
- The task is lexically separable, which makes classification relatively easy.
- No hyperparameter tuning was performed.
- Deep learning models were not explored.

---

## Author

Your Zobiya 
Course name: NLU / Semester 2

