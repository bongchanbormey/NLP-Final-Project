import streamlit as st
import joblib
import re
import html
from emoji import demojize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# 1. Preprocessing
def preprocess_text(text):
    text = text.lower()                                  # Convert to lowercase
    text = re.sub(r'\brt\b', '', text)                   # Remove RT
    text = html.unescape(text)                           # Decode HTML entities
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)    # Remove URLs
    text = re.sub(r'@\w+|\#', '', text)                  # Remove Twitter handles and hashtags
    text = re.sub(r'[^a-zA-Z\s!?.]', '', text)           # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()             # Remove extra whitespace
    text = demojize(text)    

    tokens = word_tokenize(text)  

    # Stopwords excluding negation
    stop_words = set(stopwords.words('english'))
    negations = {"no", "not", "nor", "n't", "ain", "aren", "couldn", "didn", "doesn",
                 "hadn", "hasn", "haven", "isn", "mightn", "mustn", "needn", "shan",
                 "shouldn", "wasn", "weren", "won", "wouldn"}
    stop_words = stop_words - negations

    processed_tokens = []
    lemmatizer = WordNetLemmatizer()

    sent_pos = {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
    tagged = pos_tag(tokens)

    for token, tag in tagged:
        if token not in stop_words and len(token) > 2:
            lemmatizer_token = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatizer_token)

            if tag in sent_pos:
                processed_tokens.append(f"{lemmatizer_token}_{tag}")

    return ' '.join(processed_tokens)


# 2. Load model + vectorizer
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("hate_speech_model.pkl")
    return vectorizer, model

vectorizer, model = load_artifacts()

# id -> label mapping
LABEL_MAP = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither",
}


# 3. Streamlit UI
st.title("Harmful Hate Speech Detector")
st.write("Type in a sentence:")

user_text = st.text_area("Enter text here:", height=150)

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please type something first.")
    else:
        # Preprocess input like training data
        processed = preprocess_text(user_text)
        X = vectorizer.transform([processed])

        # Prediction
        pred_class = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]

        label = LABEL_MAP.get(pred_class, "Unknown")

        st.markdown(f"### Prediction: **{label}**")

        # Show class probabilities
        st.write("Confidence:")
        for class_id, class_label in LABEL_MAP.items():
            st.write(f"- {class_label}: {pred_proba[class_id]:.2%}")
