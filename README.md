# Fake News Classifier

This project uses machine learning to detect fake news articles based on their content. Built with Python and deployed via Streamlit.

---

## Dataset

Trained on the [Kaggle Fake News competition dataset](https://www.kaggle.com/c/fake-news) (`train.csv`), where each article has `title`, `author`, `text` and a binary `label` (1 = unreliable, 0 = reliable). Only `train.csv` is used, since the competition's `test.csv` has no labels; it is split into train/test sets for evaluation.

---

## Features

-  TF-IDF vectorization of article text
-  Logistic Regression, Naive Bayes, Random Forest, and XGBoost classifiers
-  Model evaluation: Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix
-  Word importance plots for interpretability
-  Live Streamlit app with text input + prediction

---

## Screenshots

Example of real news:
![real_news_screenshot](https://github.com/user-attachments/assets/276e7437-90b7-47ad-9aee-c48e6e1b5444)

Example of fake news:
![fake_news_screenshot](https://github.com/user-attachments/assets/0d7ceaa5-c991-4dd5-accf-157c42ba1a01)

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 84.90% | 88.63% | 80.05% | 84.12% |
| Random Forest | 92.03% | 92.33% | 91.66% | 92.00% |
| Logistic Regression | 92.05% | 91.57% | 92.63% | 92.09% |
| **XGBoost** | **93.88%** | **92.49%** | **95.52%** | **93.98%** |


---

## Try It Locally

1. Clone the repo  
2. Install requirements  

pip install -r app/requirements.txt

3. Run the app  

streamlit run app/streamlit_app.py

---

Tech Stack: Python · XGBoost · Scikit-learn · Streamlit · NLTK · Pandas · TF-IDF · Matplotlib · Seaborn · Git · GitHub

---

## Author

Built by Santiago Martínez — computer systems engineer
