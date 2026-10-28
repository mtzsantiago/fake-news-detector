# Fake News Classifier

This project uses machine learning to detect fake news articles based on their content. Built with Python and deployed via Streamlit.

Live demo: https://mtzsantiago-fake-news-detector.streamlit.app/

---

## Dataset

Trained on the [Kaggle Fake News competition dataset](https://www.kaggle.com/c/fake-news) (`train.csv`, 20,800 articles; 20,761 after dropping rows with missing text). Each article has a binary `label` (1 = unreliable, 0 = reliable). Only `train.csv` is used, since the competition's `test.csv` has no labels. The `author` column is dropped, and the model uses article title + text only.

---

## Preprocessing

Lowercasing, removal of non-alphabetic characters and stopwords, lemmatization (NLTK), and filtering to English vocabulary. Text is vectorized with TF-IDF (5,000 features, `min_df=5`) and split 80/20 (stratified, `random_state=42`) before fitting the vectorizer, so the test set stays unseen.

---

## Features

-  TF-IDF vectorization of article text
-  Logistic Regression, Naive Bayes, Random Forest, and XGBoost classifiers
-  Model evaluation: Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix
-  Word importance plots for interpretability
-  Live Streamlit app with text input + prediction
-  NLP preprocessing: stopword removal, lemmatization, English-vocabulary filtering
  
---

## Limitations

- Trained on a 2016-era English news dataset focused mostly on US politics, so it may not generalize to other topics, regions or time periods.
- It learns writing style and vocabulary patterns, not factual accuracy. A well-written false article can be classified as real, and a plain true one as fake.
- Low-confidence predictions (close to 50%) should be treated as uncertain.

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

```bash
git clone https://github.com/mtzsantiago/fake-news-detector.git
cd fake-news-detector
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

NLTK data is downloaded automatically on first run.

---

Tech Stack: Python · XGBoost · Scikit-learn · Streamlit · NLTK · Pandas · TF-IDF · Matplotlib · Seaborn · Git · GitHub

---

## Author

Built by Santiago Martínez — Computer Systems Engineering graduate. [LinkedIn](https://www.linkedin.com/in/mtzsantiago/)
