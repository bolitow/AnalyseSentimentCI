# front.py
import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv('API_URL', 'https://projet-analyse-tweet-7c7caa373be8.herokuapp.com/predict')
DECISION_URL = os.getenv('DECISION_URL', 'https://projet-analyse-tweet-7c7caa373be8.herokuapp.com/decision')

# Page configuration
st.set_page_config(
    page_title="Analyse de Sentiment",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .positive-box {
        background-color: rgba(76, 175, 80, 0.2);
        border: 1px solid #4CAF50;
    }
    .negative-box {
        background-color: rgba(244, 67, 54, 0.2);
        border: 1px solid #F44336;
    }
    .info-text {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
    .prediction-label {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-value {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Analyse de Sentiment</h1>", unsafe_allow_html=True)

# Main content
text_input = st.text_area(
    "Entrez un texte à analyser",
    height=150,
    help="Entrez le texte dont vous souhaitez analyser le sentiment"
)

# Add example buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Exemple positif"):
        text_input = "J'adore ce produit, il est fantastique ! Je le recommande vivement à tous mes amis."
        st.session_state.text_input = text_input
        st.experimental_user()
with col2:
    if st.button("Exemple négatif"):
        text_input = "Je suis très déçu par ce service. Le produit ne fonctionne pas comme prévu et le support client est inexistant."
        st.session_state.text_input = text_input
        st.experimental_user()

# Prediction button
if st.button("Analyser le sentiment", type="primary"):
    if not text_input:
        st.warning("Veuillez entrer un texte à analyser.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                # Call API
                response = requests.post(API_URL, json={"text": text_input}, timeout=10, verify=False)

                if response.status_code == 200:
                    result = response.json()
                    positive = result['positive']
                    negative = result['negative']

                    # Display results
                    st.markdown("### Résultats de l'analyse")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            f"""
                            <div class='result-box positive-box'>
                                <p class='prediction-label'>Sentiment Positif</p>
                                <p class='prediction-value'>{positive:.2%}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                    with col2:
                        st.markdown(
                            f"""
                            <div class='result-box negative-box'>
                                <p class='prediction-label'>Sentiment Négatif</p>
                                <p class='prediction-value'>{negative:.2%}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                    # Overall sentiment
                    st.markdown("### Interprétation")
                    if positive > negative:
                        st.success(f"Le texte est globalement **positif** avec une confiance de {positive:.2%}")
                    else:
                        st.error(f"Le texte est globalement **négatif** avec une confiance de {negative:.2%}")

                    # Add explanation
                    st.markdown("<p class='info-text'>Les pourcentages représentent la probabilité que le texte exprime un sentiment positif ou négatif.</p>", unsafe_allow_html=True)

                    # Store prediction ID in session state
                    st.session_state.prediction_id = result.get('prediction_id')

                    # Add Accept/Reject buttons
                    st.markdown("### Êtes-vous d'accord avec cette prédiction?")
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("✅ Accepter", type="primary"):
                            with st.spinner("Envoi de votre décision (accept)..."):
                                try:
                                    response = requests.post(
                                        DECISION_URL, 
                                        json={"prediction_id": st.session_state.prediction_id, "decision": "accept"},
                                        timeout=10,
                                        verify=False
                                    )
                                    if response.status_code == 200:
                                        result = response.json()
                                        st.success(f"Merci pour votre retour ! {result.get('message', '')}")
                                    else:
                                        st.error(f"Erreur lors de l'envoi de votre décision (Code: {response.status_code})")
                                except requests.exceptions.RequestException as e:
                                    st.error(f"Erreur de connexion à l'API: {str(e)}")

                    with col2:
                        if st.button("❌ Rejeter", type="secondary"):
                            with st.spinner("Envoi de votre décision (reject)..."):
                                try:
                                    response = requests.post(
                                        DECISION_URL, 
                                        json={"prediction_id": st.session_state.prediction_id, "decision": "reject"},
                                        timeout=10,
                                        verify=False
                                    )
                                    if response.status_code == 200:
                                        result = response.json()
                                        st.success(f"Merci pour votre retour ! {result.get('message', '')}")
                                    else:
                                        st.error(f"Erreur lors de l'envoi de votre décision (Code: {response.status_code})")
                                except requests.exceptions.RequestException as e:
                                    st.error(f"Erreur de connexion à l'API: {str(e)}")

                else:
                    st.error(f"Erreur lors de la prédiction (Code: {response.status_code}). Veuillez réessayer.")

            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API: {str(e)}")
                st.info("Vérifiez que le backend est démarré et accessible.")
