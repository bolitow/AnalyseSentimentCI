# app/azure_monitor.py - Version simplifiée
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)

# Configuration conditionnelle d'Azure Monitor
AZURE_MONITOR_ENABLED = os.getenv('AZURE_MONITOR_ENABLED', 'true').lower() == 'true'
CONNECTION_STRING_CONFIGURED = bool(os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING'))

if AZURE_MONITOR_ENABLED and CONNECTION_STRING_CONFIGURED:
    try:
        from opentelemetry import metrics, trace

        AZURE_IMPORTS_AVAILABLE = True
    except ImportError as e:
        logger.error(f"Failed to import OpenTelemetry dependencies: {e}")
        AZURE_IMPORTS_AVAILABLE = False
        AZURE_MONITOR_ENABLED = False
else:
    AZURE_IMPORTS_AVAILABLE = False


class AzureMonitor:
    """Gestionnaire pour Azure Application Insights avec OpenTelemetry simplifié"""

    def __init__(self):
        """
        Initialise Azure Monitor en utilisant les providers déjà configurés
        """
        self.connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')

        # Vérification des prérequis
        if not AZURE_MONITOR_ENABLED:
            logger.info("Azure Monitor is disabled by configuration")
            self.enabled = False
            self._setup_dummy_metrics()
            return

        if not CONNECTION_STRING_CONFIGURED:
            logger.warning("Azure Application Insights connection string not found. Monitoring disabled.")
            self.enabled = False
            self._setup_dummy_metrics()
            return

        if not AZURE_IMPORTS_AVAILABLE:
            logger.warning("Azure Monitor imports are not available. Monitoring disabled.")
            self.enabled = False
            self._setup_dummy_metrics()
            return

        try:
            # Utiliser les providers déjà configurés par configure_azure_monitor()
            self.meter = metrics.get_meter("sentiment_analysis", "1.0.0")
            self.tracer = trace.get_tracer("sentiment_analysis", "1.0.0")

            # Créer des métriques personnalisées
            self.prediction_counter = self.meter.create_counter(
                name="sentiment_predictions_total",
                description="Total number of sentiment predictions made",
                unit="1"
            )

            self.rejection_counter = self.meter.create_counter(
                name="sentiment_rejections_total",
                description="Total number of rejected predictions",
                unit="1"
            )

            self.processing_time_histogram = self.meter.create_histogram(
                name="sentiment_processing_duration_ms",
                description="Processing time for sentiment predictions in milliseconds",
                unit="ms"
            )

            # Historique des rejets pour la détection d'anomalies
            self.rejection_history = deque(maxlen=100)

            self.enabled = True
            logger.info("Azure Monitor metrics configured successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure Monitor: {e}")
            self.enabled = False
            self._setup_dummy_metrics()

    def _setup_dummy_metrics(self):
        """Configure des métriques factices quand Azure Monitor n'est pas disponible"""
        self.prediction_counter = DummyMetric()
        self.rejection_counter = DummyMetric()
        self.processing_time_histogram = DummyMetric()
        self.tracer = DummyTracer()
        self.rejection_history = deque(maxlen=100)

    def log_prediction(self, text: str, prediction: Dict[str, float], prediction_id: str):
        """Enregistre une prédiction dans Application Insights"""
        if not self.enabled:
            return

        try:
            predicted_sentiment = "positive" if prediction.get('positive', 0) > 0.5 else "negative"
            confidence = max(prediction.get('positive', 0), prediction.get('negative', 0))
            confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"

            # Enregistrer la métrique avec des attributs
            self.prediction_counter.add(1, {
                "sentiment": predicted_sentiment,
                "confidence_level": confidence_level,
                "model": "logistic_regression"
            })

            # Créer un span pour la prédiction
            with self.tracer.start_as_current_span("sentiment_prediction") as span:
                span.set_attributes({
                    "prediction.id": prediction_id,
                    "prediction.text_length": len(text),
                    "prediction.positive_score": prediction.get('positive', 0),
                    "prediction.negative_score": prediction.get('negative', 0),
                    "prediction.sentiment": predicted_sentiment,
                    "prediction.confidence": confidence,
                })

                # Log structuré pour Application Insights
                logger.info(
                    f"Sentiment prediction completed: {predicted_sentiment} ({confidence:.2%})",
                    extra={
                        'custom_dimensions': {
                            'event_type': 'prediction',
                            'prediction_id': prediction_id,
                            'text_length': len(text),
                            'positive_score': prediction.get('positive', 0),
                            'negative_score': prediction.get('negative', 0),
                            'predicted_sentiment': predicted_sentiment,
                            'confidence': confidence,
                            'confidence_level': confidence_level,
                            'model_version': '1.0'
                        }
                    }
                )

        except Exception as e:
            logger.error(f"Failed to log prediction to Azure Monitor: {e}")

    def log_rejection(self, text: str, prediction: Dict[str, float], prediction_id: str):
        """Enregistre un rejet dans Application Insights"""
        if not self.enabled:
            return

        try:
            predicted_sentiment = "positive" if prediction.get('positive', 0) > 0.5 else "negative"
            confidence = max(prediction.get('positive', 0), prediction.get('negative', 0))

            # Enregistrer la métrique de rejet
            self.rejection_counter.add(1, {
                "predicted_sentiment": predicted_sentiment,
                "model": "logistic_regression"
            })

            # Ajouter à l'historique
            rejection_time = datetime.now()
            self.rejection_history.append(rejection_time)

            # Log de rejet
            logger.warning(
                f"Prediction rejected: {predicted_sentiment} ({confidence:.2%})",
                extra={
                    'custom_dimensions': {
                        'event_type': 'rejection',
                        'prediction_id': prediction_id,
                        'text_length': len(text),
                        'positive_score': prediction.get('positive', 0),
                        'negative_score': prediction.get('negative', 0),
                        'predicted_sentiment': predicted_sentiment,
                        'confidence': confidence,
                        'timestamp': rejection_time.isoformat(),
                        'model_version': '1.0'
                    }
                }
            )

            # Vérifier si on doit déclencher une alerte
            self._check_rejection_threshold()

        except Exception as e:
            logger.error(f"Failed to log rejection to Azure Monitor: {e}")

    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Enregistre des métriques de performance"""
        if not self.enabled:
            return

        try:
            self.processing_time_histogram.record(duration_ms, {
                "operation": operation,
                "model": "logistic_regression"
            })
        except Exception as e:
            logger.error(f"Failed to log performance to Azure Monitor: {e}")

    def _check_rejection_threshold(self):
        """Vérifie si le seuil de rejets est dépassé (3 rejets en 5 minutes)"""
        if not self.enabled or len(self.rejection_history) < 3:
            return

        now = datetime.now()
        five_minutes_ago = now - timedelta(minutes=5)

        recent_rejections = [
            timestamp for timestamp in self.rejection_history
            if timestamp > five_minutes_ago
        ]

        if len(recent_rejections) >= 3:
            logger.critical(
                f"HIGH REJECTION RATE ALERT: {len(recent_rejections)} rejections in 5 minutes",
                extra={
                    'custom_dimensions': {
                        'event_type': 'alert',
                        'alert_type': 'high_rejection_rate',
                        'rejection_count': len(recent_rejections),
                        'time_window_minutes': 5,
                        'threshold': 3,
                        'timestamp': datetime.now().isoformat(),
                        'severity': 'critical'
                    }
                }
            )

    def log_error(self, error_message: str, error_type: str = 'general', **kwargs):
        """Enregistre une erreur dans Application Insights"""
        if not self.enabled:
            return

        try:
            logger.error(
                error_message,
                extra={
                    'custom_dimensions': {
                        'event_type': 'error',
                        'error_type': error_type,
                        'timestamp': datetime.now().isoformat(),
                        **kwargs
                    }
                }
            )
        except Exception as e:
            logger.error(f"Failed to log error to Azure Monitor: {e}")


# Classes factices pour quand Azure Monitor n'est pas disponible
class DummyMetric:
    def add(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass


class DummySpan:
    def set_attributes(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class DummyTracer:
    def start_as_current_span(self, name):
        return DummySpan()