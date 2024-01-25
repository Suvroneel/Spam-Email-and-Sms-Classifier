import nltk
import streamlit as st
import pickle
import string

from nltk import *
from nltk.corpus import *
from  string import *

nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()



def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y=[]
  #Removing Special Characters ; text is a array now
  for i in text:
    if i.isalnum():
      y.append(i)

#copying values of y to text and emptying y
  text=y[:]
  y.clear()


  #Removing stop words & punctutations
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text=y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))


  return " ".join(y)



# rb - read

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/Sms Classifier")
input_sms = st.text_area("Enter the message")

#If buton is pressed
if st.button('Predict'):
  # 1 pre-process

  transformed_sms= transform_text(input_sms)

# 2 vectorizevector input
  vector_input=tfidf.transform([transformed_sms])


# 3 predict - extracting 0th item since spam are in 1 , 0 
  result = model.predict(vector_input)[0]


# 4 Display
  if result ==1:
      st.header("Not spam")
    
  else:
      st.header("Spam")
