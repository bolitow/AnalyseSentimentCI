# app.py - Version corrigée pour Azure Monitor
import logging
import os
import time
import traceback

# IMPORTANT: Configure Azure Monitor AVANT tous les autres imports
from dotenv import load_dotenv

load_dotenv()

# Configuration d'Azure Monitor - DOIT être fait en premier
from azure.monitor.opentelemetry import configure_azure_monitor

# Vérifier que la connection string est définie
connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
if connection_string:
    configure_azure_monitor(
        connection_string=connection_string,
        enable_live_metrics=True
    )
    print(f"Azure Monitor configuré avec connection string: {connection_string[:50]}...")
else:
    print("ATTENTION: APPLICATIONINSIGHTS_CONNECTION_STRING non définie - Azure Monitor désactivé")

# Maintenant importer Flask APRÈS la configuration Azure Monitor
from flask import Flask, request, jsonify
from app.predict import SentimentPredictor
from app.azure_monitor import AzureMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize predictor et Azure Monitor
predictor = SentimentPredictor()
azure_monitor = AzureMonitor()


# Rest of your code remains the same...
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'version': os.getenv('APP_VERSION', '1.0.0'),
        'monitoring': 'enabled' if azure_monitor.enabled else 'disabled',
        'connection_string_configured': bool(connection_string)
    })


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
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())

        azure_monitor.log_error(
            f"Prediction API error: {str(e)}",
            error_type='api_error',
            endpoint='/predict'
        )

        return jsonify({
            'error': 'Une erreur est survenue lors du traitement de la demande',
            'details': str(e)
        }), 500


# Autres routes...
@app.route('/decision', methods=['POST'])
def decision():
    start_time = time.time()

    try:
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

        if decision not in ['accept', 'reject']:
            logger.warning(f"Invalid decision value: {decision}")
            return jsonify({'error': 'Valeur de décision invalide. Utilisez "accept" ou "reject"'}), 400

        logger.info(f"Processing decision request: prediction_id={prediction_id}, decision={decision}")

        result = predictor.handle_decision(prediction_id, decision)

        processing_time = time.time() - start_time
        logger.info(f"Decision processed in {processing_time:.4f} seconds")

        response_data = {
            **result,
            'metadata': {
                'processing_time_ms': round(processing_time * 1000),
                'timestamp': time.time()
            }
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing decision: {str(e)}")
        logger.error(traceback.format_exc())

        azure_monitor.log_error(
            f"Decision API error: {str(e)}",
            error_type='api_error',
            endpoint='/decision'
        )

        return jsonify({
            'error': 'Une erreur est survenue lors du traitement de la décision',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting API server on port {port}")
    logger.info(f"Azure Monitor enabled: {azure_monitor.enabled}")
    logger.info(f"Connection string configured: {bool(connection_string)}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )