import streamlit as st
import pickle
import numpy as np

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

def predict_spam(email_text):
    transformed_text = tfidf.transform([email_text]).toarray()
    prediction = model.predict(transformed_text)
    return 'Spam' if prediction == 1 else 'Not Spam'

st.title("Email Spam Detection")

st.write("This app predicts whether an email is spam or not. Paste your email text below.")

email_text = st.text_area("Enter the email text here")

if st.button("Predict"):
    if email_text:
        prediction = predict_spam(email_text)
        st.write(f"The email is classified as: **{prediction}**")
    else:
        st.write("Please enter some email text to classify.")
