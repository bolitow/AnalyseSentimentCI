# app/predict.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json


class SentimentPredictor:
    def __init__(self):
        # Chargement du modèle local
        self.model = load_model('app/model')

        # Chargement du tokenizer
        with open('app/model/tokenizer.json') as f:
            self.tokenizer = tokenizer_from_json(f.read())

    def preprocess(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        return pad_sequences(sequence, maxlen=100)

    def predict(self, text):
        processed = self.preprocess(text)
        prediction = self.model.predict(processed)[0]
        return {
            "negative": float(prediction[0]),
            "positive": float(prediction[1])
        }