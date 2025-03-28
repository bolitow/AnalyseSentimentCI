from app.predict import SentimentPredictor
import unittest

class TestPredict(unittest.TestCase):
    def test_prediction_format(self):
        predictor = SentimentPredictor()
        result = predictor.predict("I love this product")
        self.assertIn("positive", result)
        self.assertIsInstance(result["positive"], float)

if __name__ == "__main__":
    unittest.main()