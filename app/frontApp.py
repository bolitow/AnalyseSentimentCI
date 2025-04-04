# front.py
import streamlit as st
import requests

st.title("Analyse de sentiment via API")

text_input = st.text_area("Entrez un texte")

if st.button("Prédire"):
    # Appel à l'API backend sur le port 5000
    response = requests.post("http://127.0.0.1:5000/predict", json={"text": text_input})

    if response.status_code == 200:
        result = response.json()
        st.write(f"Positive : {result['positive']:.2%}")
        st.write(f"Negative : {result['negative']:.2%}")
    else:
        st.error("Erreur lors de la prédiction. Vérifiez que le backend est démarré.")
