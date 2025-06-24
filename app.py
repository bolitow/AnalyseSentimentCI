import os
import logging
import time
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.flask import FlaskInstrumentor

from predict import SentimentPredictor
from azure_monitor import AzureMonitor

# Chargement .env
load_dotenv()

# Configuration Azure Monitor (traces, metrics, logs)
configure_azure_monitor(
    connection_string=os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING'),
    enable_metrics=True,
    enable_tracing=True,
    enable_logs=True,
    enable_live_metrics=True
)

# Setup Flask
app = Flask(__name__)
FlaskInstrumentor().instrument_app(app)

# Initialisation modules
predictor = SentimentPredictor()
azure_monitor = AzureMonitor()

# Logging de base
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'monitoring': 'enabled' if azure_monitor.enabled else 'disabled'
    })

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    try:
        data = request.get_json(force=True)
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Champ "text" manquant'}), 400

        result = predictor.predict(text)
        duration_ms = (time.time() - start) * 1000
        azure_monitor.log_performance('predict', duration_ms, text_length=len(text))

        return jsonify({**result, 'processing_time_ms': round(duration_ms, 2)})

    except Exception as e:
        logger.error(str(e))
        logger.debug(traceback.format_exc())
        azure_monitor.log_error(f"Predict error: {e}", error_type='predict')
        return jsonify({'error': str(e)}), 500

@app.route('/decision', methods=['POST'])
def decision():
    try:
        data = request.get_json(force=True)
        pid = data.get('prediction_id')
        decision = data.get('decision')
        if decision == 'reject':
            azure_monitor.log_rejection('', {'positive':0}, pid)
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(e)
        azure_monitor.log_error(f"Decision error: {e}", error_type='decision')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG')=='True')