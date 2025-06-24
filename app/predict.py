import os
import time
from dotenv import load_dotenv  # pip install python-dotenv
import smtplib
import joblib
import uuid
from email.mime.text import MIMEText

# Chargement des variables d'environnement
load_dotenv()


class SentimentPredictor:
    def __init__(self):
        # Chargement des modèles
        # Get the absolute path to the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Get model paths from environment variables or use default paths
        model_path = os.getenv('MODEL_PATH', 'app/model/logistic_regression_model.pkl')
        vectorizer_path = os.getenv('VECTORIZER_PATH', 'app/model/tfidf_vectorizer.pkl')

        # Convert to absolute paths if they are relative
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
        if not os.path.isabs(vectorizer_path):
            vectorizer_path = os.path.join(project_root, vectorizer_path)

        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

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

        # Stockage des prédictions récentes pour les décisions accept/reject
        self.recent_predictions = {}

    def preprocess(self, text):
        return self.vectorizer.transform([text])

    def predict(self, text, true_label=None):
        processed = self.preprocess(text)
        probabilities = self.model.predict_proba(processed)[0]
        prediction = self.model.predict(processed)[0]

        # Generate a unique ID for this prediction
        prediction_id = str(uuid.uuid4())

        # Store prediction data
        self.recent_predictions[prediction_id] = {
            "text": text,
            "prediction": prediction,
            "probabilities": probabilities,
            "timestamp": time.time()
        }

        # Vérification de l'exactitude seulement si true_label est fourni
        if true_label is not None:
            is_correct = prediction == true_label
            self.recent_predictions[prediction_id]["true_label"] = true_label
            self.recent_predictions[prediction_id]["is_correct"] = is_correct

            # Gestion des échecs consécutifs
            if not is_correct:
                self.consecutive_failures += 1
                print(f"Échec consécutif : {self.consecutive_failures}")  # Pour debug
                if self.consecutive_failures >= 3:
                    self.send_alert_email()
            else:
                self.consecutive_failures = 0

        result = {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1]),
            "prediction_id": prediction_id
        }

        return result

    def handle_decision(self, prediction_id, decision):
        """
        Handle user's accept/reject decision for a prediction

        Args:
            prediction_id (str): The unique ID of the prediction
            decision (str): Either 'accept' or 'reject'

        Returns:
            dict: Status of the operation
        """
        if prediction_id not in self.recent_predictions:
            return {"status": "error", "message": "Prediction ID not found"}

        prediction_data = self.recent_predictions[prediction_id]
        prediction_data["user_decision"] = decision

        if decision == "reject":
            self.consecutive_failures += 1
            print(f"Rejection received. Échec consécutif : {self.consecutive_failures}")

            if self.consecutive_failures >= 3:
                self.send_alert_email(is_rejection=True)
        else:  # decision == "accept"
            self.consecutive_failures = 0

        return {"status": "success", "message": f"Decision '{decision}' recorded"}

    def send_alert_email(self, is_rejection=False):
        try:
            # Création du message selon le type d'alerte
            if is_rejection:
                message_text = "ALERTE : 3 prédictions rejetées consécutivement par les utilisateurs !"
                subject = 'ALERTE MODELE - Rejets Utilisateurs'
            else:
                message_text = "ALERTE : 3 prédictions incorrectes consécutives détectées !"
                subject = 'ALERTE MODELE - Prédictions Incorrectes'

            msg = MIMEText(message_text)
            msg['Subject'] = subject
            msg['From'] = self.email_from
            msg['To'] = self.email_to

            # Envoi via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                server.sendmail(self.email_from, [self.email_to], msg.as_string())

            print(f"Email d'alerte envoyé: {subject}")

        except Exception as e:
            print(f"Erreur lors de l'envoi de l'email : {str(e)}")
