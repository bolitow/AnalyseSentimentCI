import unittest
from unittest.mock import patch, MagicMock

from app.predict import SentimentPredictor


class TestPredict(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.predictor = SentimentPredictor()

    def test_prediction_format(self):
        """Test that prediction returns the expected format."""
        result = self.predictor.predict("I love this product")
        self.assertIn("positive", result)
        self.assertIn("negative", result)
        self.assertIsInstance(result["positive"], float)
        self.assertIsInstance(result["negative"], float)
        self.assertAlmostEqual(result["positive"] + result["negative"], 1.0, places=5)

    def test_positive_sentiment(self):
        """Test that positive text returns higher positive probability."""
        result = self.predictor.predict("I love this product, it's amazing!")
        self.assertGreater(result["positive"], result["negative"])

    def test_negative_sentiment(self):
        """Test that negative text returns higher negative probability."""
        result = self.predictor.predict("I hate this product, it's terrible!")
        self.assertGreater(result["negative"], result["positive"])

    def test_neutral_text(self):
        """Test with neutral text."""
        result = self.predictor.predict("This is a product.")
        self.assertIn("positive", result)
        self.assertIn("negative", result)

    def test_empty_text(self):
        """Test with empty text."""
        result = self.predictor.predict("")
        self.assertIn("positive", result)
        self.assertIn("negative", result)

    def test_consecutive_failures(self):
        """Test the consecutive failures counter."""
        # Reset counter
        self.predictor.consecutive_failures = 0

        # First incorrect prediction
        self.predictor.predict("positive text", true_label=0)  # Assuming 0 is negative
        self.assertEqual(self.predictor.consecutive_failures, 1)

        # Second incorrect prediction
        self.predictor.predict("another positive text", true_label=0)
        self.assertEqual(self.predictor.consecutive_failures, 2)

        # Correct prediction should reset counter
        with patch.object(self.predictor.model, 'predict') as mock_predict:
            mock_predict.return_value = [0]  # Mock to return the expected label
            self.predictor.predict("some text", true_label=0)
            self.assertEqual(self.predictor.consecutive_failures, 0)

    @patch('smtplib.SMTP')
    def test_alert_email(self, mock_smtp):
        """Test that alert email is sent after 3 consecutive failures."""
        # Setup mock
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Reset counter
        self.predictor.consecutive_failures = 0

        # Three incorrect predictions should trigger email
        for _ in range(3):
            self.predictor.predict("some text", true_label=0)

        # Check that email was sent
        mock_server.sendmail.assert_called_once()

if __name__ == "__main__":
    unittest.main()
