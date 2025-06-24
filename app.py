import os
import logging
import time
import traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from app.predict import SentimentPredictor
from app.azure_monitor import AzureMonitor

# Chargement .env
load_dotenv()

# Configuration Azure Monitor (uniquement ici)
if os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING'):
    configure_azure_monitor(
        connection_string=os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING'),
        enable_live_metrics=True,
        enable_metrics=True,
        enable_tracing=True,
        enable_logs=True
    )

# Logging standard
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s'
)

app = Flask(__name__)
FlaskInstrumentor().instrument_app(app)

# Instanciation
azure_monitor = AzureMonitor()
predictor = SentimentPredictor()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'monitoring': 'enabled' if azure_monitor.enabled else 'disabled',
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    try:
        data = request.get_json()
        text = data.get('text')
        result = predictor.predict(text)
        duration_ms = (time.time() - start)*1000
        azure_monitor.log_performance('predict', duration_ms, text_length=len(text))
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error predict: {e}")
        logging.error(traceback.format_exc())
        azure_monitor.log_error(str(e), error_type='predict')
        return jsonify({'error': str(e)}), 500

@app.route('/decision', methods=['POST'])
def decision():
    try:
        data = request.get_json()
        pid = data.get('prediction_id')
        dec = data.get('decision')
        result = predictor.handle_decision(pid, dec)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error decision: {e}")
        azure_monitor.log_error(str(e), error_type='decision')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logging.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)