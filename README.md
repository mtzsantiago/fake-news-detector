# Fake News Classifier

This project uses machine learning to detect fake news articles based on their content. Built with Python, trained on a labeled Kaggle dataset, and deployed via Streamlit.

---

## Features

- ✅ TF-IDF vectorization of article text
- ✅ Logistic Regression, Naive Bayes, Random Forest, and XGBoost classifiers
- ✅ Model evaluation: Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix
- ✅ Word importance plots for interpretability
- ✅ Live Streamlit app with text input + prediction

---

## Screenshots

Example of real news:
![real_news_screenshot](https://github.com/user-attachments/assets/276e7437-90b7-47ad-9aee-c48e6e1b5444)

Example of fake news:
![fake_news_screenshot](https://github.com/user-attachments/assets/0d7ceaa5-c991-4dd5-accf-157c42ba1a01)

---

## Model Performance (XGBoost)

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 94.25%    |
| Precision  | 93.10%    |
| Recall     | 95.57%    |
| F1 Score   | 94.32%    |


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

Built by Santiago Martínez — aspiring data scientist & systems engineer
