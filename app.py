# api.py
from flask import Flask, request, jsonify
from app.predict import SentimentPredictor

app = Flask(__name__)
predictor = SentimentPredictor()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Aucun texte fourni'}), 400

    result = predictor.predict(data['text'])
    return jsonify(result)


if __name__ == '__main__':
    import os

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)