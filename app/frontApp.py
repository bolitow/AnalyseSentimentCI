# front.py
import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv('API_URL', 'https://projet-analyse-tweet-7c7caa373be8.herokuapp.com/predict')

# Page configuration
st.set_page_config(
    page_title="Analyse de Sentiment",
    page_icon="😊",
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

# Sidebar with information
with st.sidebar:
    st.header("À propos")
    st.info(
        """
        Cette application analyse le sentiment d'un texte et détermine s'il est positif ou négatif.

        Entrez simplement votre texte et cliquez sur 'Analyser' pour obtenir une prédiction.
        """
    )

    st.header("Exemples")
    st.markdown("""
    **Textes positifs:**
    - "J'adore ce produit, il est fantastique !"
    - "Excellente expérience, je recommande vivement."

    **Textes négatifs:**
    - "Je suis très déçu par ce service."
    - "Ce produit ne fonctionne pas du tout comme prévu."
    """)

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
        st.experimental_rerun()
with col2:
    if st.button("Exemple négatif"):
        text_input = "Je suis très déçu par ce service. Le produit ne fonctionne pas comme prévu et le support client est inexistant."
        st.session_state.text_input = text_input
        st.experimental_rerun()

# Prediction button
if st.button("Analyser le sentiment", type="primary"):
    if not text_input:
        st.warning("Veuillez entrer un texte à analyser.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                # Call API
                response = requests.post(API_URL, json={"text": text_input}, timeout=10)

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

                else:
                    st.error(f"Erreur lors de la prédiction (Code: {response.status_code}). Veuillez réessayer.")

            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API: {str(e)}")
                st.info("Vérifiez que le backend est démarré et accessible.")

# Footer
st.markdown("---")
st.markdown("<p class='info-text'>Développé avec Streamlit et Flask</p>", unsafe_allow_html=True)
