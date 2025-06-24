import os
import time
import logging
from dotenv import load_dotenv
import smtplib
import joblib
import uuid
from email.mime.text import MIMEText
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()


class SentimentPredictor:
    def __init__(self):
        # Chargement des modèles
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

        # Log email configuration (sans le mot de passe)
        logger.info(f"Email config - Server: {self.smtp_server}:{self.smtp_port}")
        logger.info(f"Email config - From: {self.email_from}, To: {self.email_to}")
        logger.info(f"Email config - Password configured: {'Yes' if self.email_password else 'No'}")

        # Compteur d'échecs consécutifs
        self.consecutive_failures = 0
        self.rejection_history = []  # Historique des rejets pour le debugging

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
            "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate for storage
            "prediction": prediction,
            "probabilities": probabilities,
            "timestamp": time.time()
        }

        # Clean old predictions (older than 1 hour)
        current_time = time.time()
        self.recent_predictions = {
            pid: data for pid, data in self.recent_predictions.items()
            if current_time - data["timestamp"] < 3600
        }

        result = {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1]),
            "prediction_id": prediction_id
        }

        return result

    def handle_decision(self, prediction_id, decision):
        """
        Handle user's accept/reject decision for a prediction
        """
        logger.info(f"Handling decision: {decision} for prediction {prediction_id}")

        if prediction_id not in self.recent_predictions:
            logger.warning(f"Prediction ID {prediction_id} not found in recent predictions")
            return {"status": "error", "message": "Prediction ID not found"}

        prediction_data = self.recent_predictions[prediction_id]
        prediction_data["user_decision"] = decision

        if decision == "reject":
            self.consecutive_failures += 1
            self.rejection_history.append({
                "prediction_id": prediction_id,
                "timestamp": datetime.now().isoformat(),
                "text_preview": prediction_data["text"][:50] + "..."
            })

            logger.warning(f"Rejection received. Consecutive failures: {self.consecutive_failures}")
            logger.info(f"Recent rejections: {len(self.rejection_history)} in history")

            if self.consecutive_failures >= 3:
                logger.error("ALERT: 3 consecutive rejections - sending email alert")
                self.send_alert_email(is_rejection=True)
                # Reset counter after sending alert
                self.consecutive_failures = 0
        else:  # decision == "accept"
            self.consecutive_failures = 0
            logger.info("Acceptance received. Consecutive failures reset to 0")

        return {
            "status": "success",
            "message": f"Decision '{decision}' recorded",
            "consecutive_failures": self.consecutive_failures
        }

    def send_alert_email(self, is_rejection=False):
        """Send alert email with better error handling and logging"""
        logger.info("Attempting to send alert email...")

        # Vérifier que toutes les configurations email sont présentes
        if not all([self.email_from, self.email_to, self.email_password]):
            logger.error("Email configuration incomplete. Cannot send alert.")
            logger.error(
                f"From: {bool(self.email_from)}, To: {bool(self.email_to)}, Password: {bool(self.email_password)}")
            return

        try:
            # Création du message selon le type d'alerte
            if is_rejection:
                message_text = f"""ALERTE : 3 prédictions rejetées consécutivement par les utilisateurs !

Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Derniers rejets :
"""
                for i, rejection in enumerate(self.rejection_history[-3:], 1):
                    message_text += f"\n{i}. {rejection['timestamp']} - {rejection['text_preview']}"

                subject = 'ALERTE MODELE - Rejets Utilisateurs'
            else:
                message_text = f"""ALERTE : 3 prédictions incorrectes consécutives détectées !

Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                subject = 'ALERTE MODELE - Prédictions Incorrectes'

            msg = MIMEText(message_text)
            msg['Subject'] = subject
            msg['From'] = self.email_from
            msg['To'] = self.email_to

            logger.info(f"Connecting to SMTP server {self.smtp_server}:{self.smtp_port}")

            # Envoi via SMTP avec timeout
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.set_debuglevel(1)  # Active le debug SMTP
                server.starttls()
                logger.info("TLS started successfully")

                server.login(self.email_from, self.email_password)
                logger.info("Login successful")

                server.sendmail(self.email_from, [self.email_to], msg.as_string())
                logger.info(f"Email sent successfully: {subject}")

            # Clear rejection history after successful email
            self.rejection_history = []

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP Authentication failed: {str(e)}")
            logger.error("Please check your email and app password configuration")
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error occurred: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error sending email: {str(e)}")
            logger.exception("Full traceback:")