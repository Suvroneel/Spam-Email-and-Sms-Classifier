import nltk
from textblob import TextBlob
import streamlit as st
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Google Sheets connection
from streamlit_gsheets import GSheetsConnection

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Establish Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)
existing_data = conn.read(worksheet="User Data", usecols=list(range(2)), ttl=5)
existing_data = existing_data.dropna(how="all")

# Initialize PorterStemmer
ps = PorterStemmer()

# Custom CSS for styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #74ebd5, #acb6e5);
    background-size: cover;
    position: relative;
    padding-bottom: 60px; /* Prevent content overlap with footer */
}
.stApp::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.3); /* Semi-transparent overlay */
    z-index: 0;
}
.stApp > div {
    position: relative;
    z-index: 1;
}
.stApp header {
    z-index: 0;
}
h1 {
    font-size: 36px !important;
    text-align: center;
    margin-bottom: 30px;
    color: white !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
}
h2, p, label, .stTextArea, .stButton > button {
    color: white !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
}
.stTextArea {
    margin-bottom: 20px;
}
.stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.1);
    border: 2px solid #00d4ff;
    border-radius: 10px;
    padding: 15px;
    color: white !important;
    font-size: 16px;
}
.stButton > button {
    background-color: #00d4ff;
    border: none;
    border-radius: 25px;
    padding: 10px 30px;
    font-weight: bold;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #00b0cc;
}
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: rgba(26, 37, 38, 0.8);
    text-align: center;
    padding: 10px;
    z-index: 2;
}
.footer p {
    color: white;
    font-size: 16px;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown(
    '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
    unsafe_allow_html=True)

nav_bar = """
<nav class="navbar fixed-top navbar-expand-lg navbar-dark"
     style="background-color: #1a2526; padding: 15px 0;">
  <div class="container-fluid">
    <a class="navbar-brand text-white" href="https://www.linkedin.com/in/suvroneel-nathak-593602197/">Suvroneel Nathak</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav">
        <li class="nav-item">
          <a class="nav-link active" aria-current="page" href="#" style="color: #00d4ff !important;">Home</a>
        </li>
        <li class="nav-item">
          <a class="nav-link text-white" href="#">About Us</a>
        </li>
        <li class="nav-item">
          <a class="nav-link text-white" href="#">Contact</a>
        </li>
        <li class="nav-item">
          <a class="nav-link text-white" href="https://github.com/Suvroneel">My Other Projects</a>
        </li> 
      </ul>
    </div>
  </div>
</nav>
<style>
.navbar-nav .nav-link:hover {
    color: #00d4ff !important;
    transition: color 0.3s ease;
}
</style>
"""
st.markdown(nav_bar, unsafe_allow_html=True)

# Text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('./Model/vectorizer_mkii.pkl', 'rb'))
model = pickle.load(open('./Model/model_mkii.pkl', 'rb'))

# Main app
st.title("Spam Email Classifier (v 2.1.0)")

input_sms = st.text_area("Enter the message", key="input_sms", help="Paste your email or SMS here")
prediction = ""

# Prediction logic
if st.button('Predict'):
    with st.spinner("Classifying..."):
        if not input_sms:
            st.warning("Write a message first.")
            st.stop()

        b = TextBlob(input_sms)
        if b.correct() == input_sms:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.markdown('<span style="color: #ff4d4d; font-size: 24px; font-weight: bold;">Spam 🔴</span>', unsafe_allow_html=True)
                prediction = "Spam"
            else:
                st.markdown('<span style="color: #00cc66; font-size: 24px; font-weight: bold;">Not Spam 🟢</span>', unsafe_allow_html=True)
                prediction = "Not Spam"

            input_data = pd.DataFrame([{"Input Sentence": input_sms, "Output": prediction}])
            updated_df = pd.concat([existing_data, input_data], ignore_index=True)
            conn.update(worksheet="User Data", data=updated_df)
        else:
            st.markdown('<span style="color: #ff4d4d; font-size: 24px; font-weight: bold;">Spam 🔴</span>', unsafe_allow_html=True)
            prediction = "Spam"
            input_data = pd.DataFrame([{"Input Sentence": input_sms, "Output": prediction}])
            updated_df = pd.concat([existing_data, input_data], ignore_index=True)
            conn.update(worksheet="User Data", data=updated_df)

# Footer
st.markdown("""
<div class="footer">
    <p>Created by Suvroneel Nathak</p>
</div>
""", unsafe_allow_html=True)
