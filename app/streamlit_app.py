import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Load saved model and vectorizer
model = pickle.load(open('models/xgb_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

# App UI
st.title("📰 Fake News Classifier")
st.markdown("Enter a news article below. The model will predict whether it's real or fake.")

user_input = st.text_area("Paste the article text here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        if len(user_input.split()) < 50:
            st.warning("This article may be too short for accurate analysis.")

        # Clean input minimally (optional: match training pipeline)
        input_df = pd.Series([user_input])
        transformed_input = vectorizer.transform(input_df)

        prediction = model.predict(transformed_input)[0]
        probability = model.predict_proba(transformed_input)[0][prediction]

        label = "🟢 REAL" if prediction == 0 else "🔴 FAKE"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {probability*100:.2f}%")

        st.markdown("---")
        st.caption("Model: XGBoost | Vectorizer: TF-IDF | Trained on cleaned article content | Built by Santiago Martínez")