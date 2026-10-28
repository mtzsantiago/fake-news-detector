import pickle
import re
from pathlib import Path

import nltk
import streamlit as st
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource
def load_resources():
    """Load the model, vectorizer and NLTK resources once per session."""
    for pkg in ("stopwords", "wordnet", "words"):
        nltk.download(pkg, quiet=True)

    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    stop_words = set(stopwords.words("english"))
    english_vocab = set(words.words())
    lemmatizer = WordNetLemmatizer()
    return model, vectorizer, stop_words, english_vocab, lemmatizer


def clean_text(text, stop_words, english_vocab, lemmatizer):
    """Same preprocessing used during training."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [w for w in text.split() if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [w for w in tokens if w in english_vocab]
    return " ".join(tokens)


model, vectorizer, stop_words, english_vocab, lemmatizer = load_resources()

# App UI
st.title("📰 Fake News Classifier")
st.markdown(
    "Enter a news article below. The model will predict whether it's real or fake."
)

user_input = st.text_area("Paste the article text here:", height=250)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        if len(user_input.split()) < 50:
            st.warning("This article may be too short for accurate analysis.")

        cleaned = clean_text(user_input, stop_words, english_vocab, lemmatizer)

        if cleaned.strip() == "":
            st.warning("No usable English text found after cleaning.")
        else:
            transformed_input = vectorizer.transform([cleaned])

            prediction = int(model.predict(transformed_input)[0])
            probability = model.predict_proba(transformed_input)[0][prediction]

            label = "🟢 REAL" if prediction == 0 else "🔴 FAKE"
            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {probability * 100:.2f}%")

st.markdown("---")
st.caption(
    "Model: XGBoost | Vectorizer: TF-IDF | Trained on the Kaggle Fake News "
    "dataset (English news, 2016 era). It detects writing patterns, not facts. "
    "Built by Santiago Martínez"
)
