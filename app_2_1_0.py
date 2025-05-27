import base64
import nltk
from textblob import TextBlob
import toml
from streamlit_gsheets import GSheetsConnection
import streamlit as st
import pickle
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)
existing_data = conn.read(worksheet="User Data", usecols=list(range(2)), ttl=5)
existing_data = existing_data.dropna(how="all")

nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()

# Background setup
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('13.jpg')

# Custom styles
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }
        .main-container {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
            margin-top: 120px;
        }
        .navbar-custom {
            background-color: rgba(0, 0, 0, 0.8);
            padding: 1rem;
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
        }
        .navbar-custom a {
            color: white;
            margin-right: 15px;
            text-decoration: none;
            font-weight: bold;
        }
    </style>
    <div class="navbar-custom">
        <a href="#">Home</a>
        <a href="#">About</a>
        <a href="#">Contact</a>
        <a href="https://github.com/Suvroneel">Other Projects</a>
    </div>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title(" Spam Email Classifier (v 2.1.0) - Test")

input_sms = st.text_area("Enter the message")

# Load model
tfidf = pickle.load(open('./Model/vectorizer_mkii.pkl', 'rb'))
model = pickle.load(open('./Model/model_mkii.pkl', 'rb'))

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

prediction = ""
b = TextBlob(input_sms)

if st.button('🚀 Predict'):
    if not input_sms:
        st.warning("Write a message first.")
        st.stop()

    if b.correct() == input_sms:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 This looks like **Spam**!")
            prediction = "Spam"
        else:
            st.success("✅ This looks **Safe** (Not Spam)")
            prediction = "Not spam"
    else:
        st.error("🚨 This looks like **Spam**!")
        prediction = "Spam"

    input_data = pd.DataFrame([
        {"Input Sentence": input_sms, "Output": prediction}
    ])
    updated_df = pd.concat([existing_data, input_data], ignore_index=True)
    conn.update(worksheet="User Data", data=updated_df)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center; padding: 10px; font-size: 14px;'>
        🚀 Made by <a href="https://www.linkedin.com/in/suvroneel-nathak-593602197/" target="_blank">Suvroneel Nathak</a> |
        <a href="https://github.com/Suvroneel" target="_blank">GitHub</a>
    </div>
""", unsafe_allow_html=True)
