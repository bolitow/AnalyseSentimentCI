# api.py
import logging
import os
import time
import traceback

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from app.predict import SentimentPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize predictor
predictor = SentimentPredictor()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'version': os.getenv('APP_VERSION', '1.0.0')
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            logger.warning("No JSON data in request")
            return jsonify({'error': 'Aucune donnée JSON fournie'}), 400

        if 'text' not in data:
            logger.warning("No 'text' field in request data")
            return jsonify({'error': 'Champ "text" manquant'}), 400

        text = data['text']

        # Log request (truncate long texts for logging)
        log_text = text[:100] + '...' if len(text) > 100 else text
        logger.info(f"Processing prediction request: '{log_text}'")

        # Make prediction
        result = predictor.predict(text)

        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Prediction completed in {processing_time:.4f} seconds")

        # Add metadata to response
        response_data = {
            **result,
            'metadata': {
                'processing_time_ms': round(processing_time * 1000),
                'timestamp': time.time()
            }
        }

        return jsonify(response_data)

    except Exception as e:
        # Log the full exception with traceback
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())

        return jsonify({
            'error': 'Une erreur est survenue lors du traitement de la demande',
            'details': str(e)
        }), 500

# Decision endpoint for accept/reject
@app.route('/decision', methods=['POST'])
def decision():
    start_time = time.time()

    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            logger.warning("No JSON data in request")
            return jsonify({'error': 'Aucune donnée JSON fournie'}), 400

        if 'prediction_id' not in data:
            logger.warning("No 'prediction_id' field in request data")
            return jsonify({'error': 'Champ "prediction_id" manquant'}), 400

        if 'decision' not in data:
            logger.warning("No 'decision' field in request data")
            return jsonify({'error': 'Champ "decision" manquant'}), 400

        prediction_id = data['prediction_id']
        decision = data['decision']

        # Validate decision value
        if decision not in ['accept', 'reject']:
            logger.warning(f"Invalid decision value: {decision}")
            return jsonify({'error': 'Valeur de décision invalide. Utilisez "accept" ou "reject"'}), 400

        # Log request
        logger.info(f"Processing decision request: prediction_id={prediction_id}, decision={decision}")

        # Handle decision
        result = predictor.handle_decision(prediction_id, decision)

        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Decision processed in {processing_time:.4f} seconds")

        # Add metadata to response
        response_data = {
            **result,
            'metadata': {
                'processing_time_ms': round(processing_time * 1000),
                'timestamp': time.time()
            }
        }

        return jsonify(response_data)

    except Exception as e:
        # Log the full exception with traceback
        logger.error(f"Error processing decision: {str(e)}")
        logger.error(traceback.format_exc())

        return jsonify({
            'error': 'Une erreur est survenue lors du traitement de la décision',
            'details': str(e)
        }), 500


# Ajoutez cet endpoint dans votre app.py après les autres endpoints

# Test email endpoint
@app.route('/test-email', methods=['GET'])
def test_email():
    """Endpoint pour tester la configuration email"""
    try:
        # Log la configuration (sans le mot de passe)
        logger.info("Testing email configuration...")
        config_info = {
            'smtp_server': predictor.smtp_server,
            'smtp_port': predictor.smtp_port,
            'email_from': predictor.email_from,
            'email_to': predictor.email_to,
            'password_configured': bool(predictor.email_password)
        }
        logger.info(f"Email config: {config_info}")

        # Forcer l'envoi d'un email de test
        predictor.consecutive_failures = 3
        predictor.send_alert_email(is_rejection=True)

        return jsonify({
            'status': 'success',
            'message': 'Test email sent (check logs)',
            'config': config_info
        })

    except Exception as e:
        logger.error(f"Test email failed: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'config': {
                'smtp_server': predictor.smtp_server,
                'email_from': predictor.email_from,
                'email_to': predictor.email_to,
                'password_configured': bool(predictor.email_password)
            }
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint non trouvé'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Méthode non autorisée'}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))

    # Log startup information
    logger.info(f"Starting API server on port {port}")

    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )
