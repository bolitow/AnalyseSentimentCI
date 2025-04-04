# app/predict.py
import joblib


class SentimentPredictor:
    def __init__(self):
        # Chargement du modèle TF-IDF + Régression Logistique
        self.model = joblib.load('app/model/logistic_regression_model.pkl')
        self.vectorizer = joblib.load('app/model/tfidf_vectorizer.pkl')

    def preprocess(self, text):
        # Transformation TF-IDF
        return self.vectorizer.transform([text])

    def predict(self, text):
        # Prétraitement et prédiction
        processed = self.preprocess(text)
        probabilities = self.model.predict_proba(processed)[0]

        return {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1])
        }
