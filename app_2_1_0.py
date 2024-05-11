import base64

import nltk
from textblob import TextBlob
import toml
# GSheets lib
from streamlit_gsheets import GSheetsConnection

import streamlit as st
import pickle
import string
import pandas as pd
from pandas import *

from nltk import *
from nltk.corpus import *
from string import *




# Establishing Google sheets connection

conn = st.connection("gsheets", type=GSheetsConnection)

# fetching exist data ue cols = no of cols , ttl = time to live
existing_data = conn.read(worksheet="User Data", usecols=list(range(2)), ttl=5)

existing_data = existing_data.dropna(how="all")  # droping empty vals

# st.dataframe(existing_data)


nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()
#####backgrouund###

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('12.png')



################
###############Navbar####################


# st.set_page_config(layout="wide")


st.markdown(
    '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
    unsafe_allow_html=True)

nav_bar = """
<nav class="navbar fixed-top navbar-expand-lg navbar-dark"
         style="background-color: black">
  <div class="container-fluid">
    <a class="navbar-brand text-white" href="https://www.linkedin.com/in/suvroneel-nathak-593602197/" >Suvroneel Nathak</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav">
        <li class="nav-item">
          <a class="nav-link active " aria-current="page" href="#">Home</a>
        </li>
        <li class="nav-item">
          <a class="nav-link text-white" href="#">About Us</a>
        </li>
        <li class="nav-item">
          <a class="nav-link text-white" href="#">Contact</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="https://github.com/Suvroneel">My Other Projects</a>
        </li> 
      </ul>
    </div>
  </div>
</nav>
"""
st.markdown(nav_bar, unsafe_allow_html=True)

# Short description of the app

# set streamlit header z-index
st.markdown('''
<style>
.stApp header {
    z-index: 0;
}
</style>
''', unsafe_allow_html=True)


################################################

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    # Removing Special Characters ; text is a array now
    for i in text:
        if i.isalnum():
            y.append(i)

    # copying values of y to text and emptying y
    text = y[:]
    y.clear()

    # Removing stop words & punctutations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# rb - read

tfidf = pickle.load(open('./Model/vectorizer_mkii.pkl', 'rb'))
model = pickle.load(open('./Model/model_mkii.pkl', 'rb'))

st.title("Email/Sms Classifier (v 2.1.0)")



input_sms = st.text_area("Enter the message")  # takes input message

prediction = ""

b = TextBlob(input_sms)

# If button is pressed
if st.button('Predict'):

    if not input_sms:
        st.warning("Write a message first .")
        st.stop()

    if b.correct() == input_sms:

        # 1 pre-process

        transformed_sms = transform_text(input_sms)

        # 2 vectorizevector input
        vector_input = tfidf.transform([transformed_sms])

        # 3 predict - extracting 0th item since spam are in 1 , 0
        result = model.predict(vector_input)[0]

        # 4 Display
        if result == 1:
            st.header("Spam")
            prediction = "Spam"

        else:
            st.header("Not spam")
            prediction = "Not spam"

        input_data = pd.DataFrame(
            [
                {
                    "Input Sentence": input_sms,
                    "Output": prediction
                }
            ]
        )
        # add inputdata to existing data
        updated_df = pd.concat([existing_data, input_data], ignore_index=True)

        # update goggle sheets
        conn.update(worksheet="User Data", data=updated_df)

    else:
        st.header("Spam")
        prediction = "Spam"
        input_data = pd.DataFrame(
            [
                {
                    "Input Sentence": input_sms,
                    "Output": prediction
                }
            ]
        )
        # add inputdata to existing data
        updated_df = pd.concat([existing_data, input_data], ignore_index=True)
        conn.update(worksheet="User Data", data=updated_df)

theme_bg_color = st.get_option("theme.backgroundColor")

# Footer
st.markdown(f"""
    <style>
        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: {theme_bg_color};
            text-align: center;
            padding: 10px;
        }}
    </style>
    <div class="footer">
        <p style="font-size: 16px;">Created by Suvroneel Nathak</p>
    </div>
""", unsafe_allow_html=True)

