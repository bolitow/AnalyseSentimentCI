# front.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv('API_URL', 'https://projet-analyse-tweet-7c7caa373be8.herokuapp.com/predict')
DECISION_URL = os.getenv('DECISION_URL', 'https://projet-analyse-tweet-7c7caa373be8.herokuapp.com/decision')

# Page configuration
st.set_page_config(
    page_title="Analyse de Sentiment",
    layout="centered",
    initial_sidebar_state="collapsed"
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
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🤖 Analyse de Sentiment</h1>", unsafe_allow_html=True)

# Initialize session state
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'feedback_sent' not in st.session_state:
    st.session_state.feedback_sent = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None


# Function to reset state
def reset_analysis():
    st.session_state.prediction_done = False
    st.session_state.feedback_sent = False
    st.session_state.current_prediction = None


# Main content
text_input = st.text_area(
    "Entrez un texte à analyser",
    height=150,
    help="Entrez le texte dont vous souhaitez analyser le sentiment",
    key="text_area"
)

# Prediction button
if st.button("🔍 Analyser le sentiment", type="primary", use_container_width=True):
    if not text_input:
        st.warning("⚠️ Veuillez entrer un texte à analyser.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                # Call API
                response = requests.post(API_URL, json={"text": text_input}, timeout=10, verify=False)

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.current_prediction = result
                    st.session_state.prediction_done = True
                    st.session_state.feedback_sent = False
                    st.rerun()
                else:
                    st.error(f"❌ Erreur lors de la prédiction (Code: {response.status_code})")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Erreur de connexion à l'API: {str(e)}")
                st.info("ℹ️ Vérifiez que le backend est démarré et accessible.")

# Display results if prediction is done
if st.session_state.prediction_done and st.session_state.current_prediction:
    result = st.session_state.current_prediction
    positive = result['positive']
    negative = result['negative']

    # Display results
    st.markdown("### 📊 Résultats de l'analyse")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class='result-box positive-box'>
                <p class='prediction-label'>😊 Sentiment Positif</p>
                <p class='prediction-value'>{positive:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class='result-box negative-box'>
                <p class='prediction-label'>😔 Sentiment Négatif</p>
                <p class='prediction-value'>{negative:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Overall sentiment
    st.markdown("### 💡 Interprétation")
    if positive > negative:
        st.success(f"✅ Le texte est globalement **positif** avec une confiance de {positive:.2%}")
    else:
        st.error(f"❌ Le texte est globalement **négatif** avec une confiance de {negative:.2%}")

    # Add explanation
    st.markdown(
        "<p class='info-text'>Les pourcentages représentent la probabilité que le texte exprime un sentiment positif ou négatif.</p>",
        unsafe_allow_html=True)

    # Feedback section
    if not st.session_state.feedback_sent:
        st.markdown("---")
        st.markdown("### 🤔 Êtes-vous d'accord avec cette prédiction?")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Oui, c'est correct", type="primary", use_container_width=True):
                with st.spinner("Envoi de votre retour..."):
                    try:
                        response = requests.post(
                            DECISION_URL,
                            json={
                                "prediction_id": result.get('prediction_id'),
                                "decision": "accept"
                            },
                            timeout=10,
                            verify=False
                        )
                        if response.status_code == 200:
                            st.session_state.feedback_sent = True
                            st.rerun()
                        else:
                            st.error(f"❌ Erreur lors de l'envoi (Code: {response.status_code})")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")

        with col2:
            if st.button("❌ Non, c'est incorrect", type="secondary", use_container_width=True):
                with st.spinner("Envoi de votre retour..."):
                    try:
                        response = requests.post(
                            DECISION_URL,
                            json={
                                "prediction_id": result.get('prediction_id'),
                                "decision": "reject"
                            },
                            timeout=10,
                            verify=False
                        )
                        if response.status_code == 200:
                            st.session_state.feedback_sent = True
                            st.rerun()
                        else:
                            st.error(f"❌ Erreur lors de l'envoi (Code: {response.status_code})")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")

    else:
        # Feedback has been sent
        st.markdown("---")
        st.success("✅ Merci pour votre retour ! Cela nous aide à améliorer notre modèle.")

        if st.button("🔄 Faire une nouvelle analyse", type="primary", use_container_width=True):
            reset_analysis()
            st.rerun()
