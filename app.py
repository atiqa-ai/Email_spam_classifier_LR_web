import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()




def transform_text(text):
    text = text.lower()  # converts text into lower case
    text = nltk.word_tokenize(text)  # splits text into words

    y = []
    for i in text:
        if i.isalnum():  # removing special characters
            y.append(i)

    text = y[:]  # copying list here
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # word stemming

    return " ".join(y)


tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

st.title("SMS/ Email Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):

#preprocessing
 transformed_sms = transform_text(input_sms)

#vectorizer
 vector_input = tfidf.transform([transformed_sms])

#predict
 model_output = model.predict(vector_input)[0]

#Display output
 if model_output == 1:
    st.header('Spam')
 else:
    st.header('Not Spam')