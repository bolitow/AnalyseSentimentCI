# app.py
import streamlit as st
from predict import SentimentPredictor

predictor = SentimentPredictor()

st.title("Analyse de sentiment avec FastText")
text_input = st.text_area("Entrez un texte")
if st.button("Prédire"):
    result = predictor.predict(text_input)
    st.write(f"Positive : {result['positive']:.2%}")
    st.write(f"Negative : {result['negative']:.2%}")