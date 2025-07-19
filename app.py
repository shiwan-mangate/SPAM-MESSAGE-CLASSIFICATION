import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
sn = SnowballStemmer("english")

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    text = [sn.stem(word) for word in text]
    return " ".join(text)

# Load your models
try:
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    st.error("❌ Failed to load model/vectorizer. Make sure 'model.pkl' and 'vectorizer.pkl' are in the same folder as app.py.")
    st.stop()

# Streamlit UI
st.title("📨 SMS SPAM CLASSIFIER")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess and vectorize
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Output
        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
