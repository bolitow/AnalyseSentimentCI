import os
from dotenv import load_dotenv  # pip install python-dotenv
import smtplib
import joblib
from email.mime.text import MIMEText

# Chargement des variables d'environnement
load_dotenv()


class SentimentPredictor:
    def __init__(self):
        # Chargement des modèles
        model_path = os.getenv('MODEL_PATH', 'app/model/logistic_regression_model.pkl')
        vectorizer_path = os.getenv('VECTORIZER_PATH', 'app/model/tfidf_vectorizer.pkl')

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        # Configuration email
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.email_from = os.getenv('EMAIL_FROM')
        self.email_to = os.getenv('EMAIL_TO')
        self.email_password = os.getenv('EMAIL_PASSWORD')

        # Compteur d'échecs consécutifs
        self.consecutive_failures = 0

    def preprocess(self, text):
        return self.vectorizer.transform([text])

    def predict(self, text, true_label=0):
        processed = self.preprocess(text)
        probabilities = self.model.predict_proba(processed)[0]
        prediction = self.model.predict(processed)[0]

        # Vérification de l'exactitude
        if true_label is not None:
            is_correct = (prediction == true_label)

            # Gestion des échecs consécutifs
            if not is_correct:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    self.send_alert_email()
            else:
                self.consecutive_failures = 0

        return {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1])
        }

    def send_alert_email(self):
        try:
            # Création du message
            msg = MIMEText("ALERTE : 3 prédictions incorrectes consécutives détectées !")
            msg['Subject'] = 'ALERTE MODELE'
            msg['From'] = self.email_from
            msg['To'] = self.email_to

            # Envoi via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                server.sendmail(self.email_from, [self.email_to], msg.as_string())

        except Exception as e:
            print(f"Erreur lors de l'envoi de l'email : {str(e)}")