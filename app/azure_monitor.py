# app/azure_monitor.py - Version corrigée
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)

# Configuration conditionnelle d'Azure Monitor
AZURE_MONITOR_ENABLED = os.getenv('AZURE_MONITOR_ENABLED', 'true').lower() == 'true'

if AZURE_MONITOR_ENABLED:
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        AZURE_IMPORTS_AVAILABLE = True
    except ImportError as e:
        logger.error(f"Failed to import Azure Monitor dependencies: {e}")
        AZURE_IMPORTS_AVAILABLE = False
        AZURE_MONITOR_ENABLED = False
else:
    AZURE_IMPORTS_AVAILABLE = False


class AzureMonitor:
    """Gestionnaire pour Azure Application Insights avec OpenTelemetry"""

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialise Azure Monitor

        Args:
            connection_string: Connection string Azure Application Insights
        """
        self.connection_string = connection_string or os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')

        # Vérification des prérequis
        if not AZURE_MONITOR_ENABLED:
            logger.info("Azure Monitor is disabled by configuration")
            self.enabled = False
            self._setup_dummy_metrics()
            return

        if not AZURE_IMPORTS_AVAILABLE:
            logger.warning("Azure Monitor imports are not available. Monitoring disabled.")
            self.enabled = False
            self._setup_dummy_metrics()
            return

        if not self.connection_string:
            logger.warning("Azure Application Insights connection string not found. Monitoring disabled.")
            self.enabled = False
            self._setup_dummy_metrics()
            return

        try:
            self._initialize_azure_monitor()
            self.enabled = True
            logger.info("Azure Application Insights initialized successfully with OpenTelemetry")
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

    def _initialize_azure_monitor(self):
        """Initialise Azure Monitor avec la méthode simplifiée et des métriques customs"""
        logger.info(f"Initializing Azure Monitor with connection string: {self.connection_string[:50]}...")

        # Utiliser la méthode configure_azure_monitor simplifiée
        configure_azure_monitor(
            connection_string=self.connection_string,
            enable_live_metrics=True
        )

        # Récupérer les providers configurés
        self.meter = metrics.get_meter("sentiment_analysis", "1.0.0")
        self.tracer = trace.get_tracer("sentiment_analysis", "1.0.0")

        # Créer des métriques personnalisées avec des noms explicites
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

        logger.info("Azure Monitor metrics configured successfully")

    def log_prediction(self, text: str, prediction: Dict[str, float], prediction_id: str):
        """
        Enregistre une prédiction dans Application Insights

        Args:
            text: Texte analysé
            prediction: Résultats de la prédiction
            prediction_id: ID unique de la prédiction
        """
        if not self.enabled:
            return

        try:
            # Déterminer le sentiment prédit
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
                    "prediction.confidence_level": confidence_level
                })

                # Log avec dimensions personnalisées pour Application Insights
                logger.info(
                    f"Sentiment prediction completed",
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
        """
        Enregistre un rejet (tweet mal prédit) dans Application Insights
        """
        if not self.enabled:
            return

        try:
            predicted_sentiment = "positive" if prediction.get('positive', 0) > 0.5 else "negative"
            confidence = max(prediction.get('positive', 0), prediction.get('negative', 0))
            confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"

            # Enregistrer la métrique de rejet
            self.rejection_counter.add(1, {
                "predicted_sentiment": predicted_sentiment,
                "confidence_level": confidence_level,
                "model": "logistic_regression"
            })

            # Ajouter à l'historique
            rejection_time = datetime.now()
            self.rejection_history.append(rejection_time)

            # Créer un span pour le rejet
            with self.tracer.start_as_current_span("sentiment_rejection") as span:
                span.set_attributes({
                    "rejection.prediction_id": prediction_id,
                    "rejection.text_length": len(text),
                    "rejection.predicted_sentiment": predicted_sentiment,
                    "rejection.confidence": confidence,
                    "rejection.confidence_level": confidence_level
                })

                # Log de rejet avec plus de détails
                logger.warning(
                    f"Prediction rejected by user",
                    extra={
                        'custom_dimensions': {
                            'event_type': 'rejection',
                            'prediction_id': prediction_id,
                            'rejected_text_preview': text[:100] + '...' if len(text) > 100 else text,
                            'text_length': len(text),
                            'positive_score': prediction.get('positive', 0),
                            'negative_score': prediction.get('negative', 0),
                            'predicted_sentiment': predicted_sentiment,
                            'confidence': confidence,
                            'confidence_level': confidence_level,
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
        """
        Enregistre des métriques de performance
        """
        if not self.enabled:
            return

        try:
            # Enregistrer la métrique de temps de traitement
            self.processing_time_histogram.record(duration_ms, {
                "operation": operation,
                "model": "logistic_regression"
            })

            # Log avec span
            with self.tracer.start_as_current_span(f"performance_{operation}") as span:
                span.set_attributes({
                    "performance.operation": operation,
                    "performance.duration_ms": duration_ms,
                    **kwargs
                })

        except Exception as e:
            logger.error(f"Failed to log performance to Azure Monitor: {e}")

    def _check_rejection_threshold(self):
        """
        Vérifie si le seuil de rejets est dépassé (3 rejets en 5 minutes)
        """
        if not self.enabled or len(self.rejection_history) < 3:
            return

        now = datetime.now()
        five_minutes_ago = now - timedelta(minutes=5)

        recent_rejections = [
            timestamp for timestamp in self.rejection_history
            if timestamp > five_minutes_ago
        ]

        if len(recent_rejections) >= 3:
            self._trigger_alert(len(recent_rejections))

    def _trigger_alert(self, rejection_count: int):
        """
        Déclenche une alerte dans Application Insights
        """
        try:
            with self.tracer.start_as_current_span("high_rejection_alert") as span:
                span.set_attributes({
                    "alert.type": "high_rejection_rate",
                    "alert.rejection_count": rejection_count,
                    "alert.time_window_minutes": 5,
                    "alert.threshold": 3
                })

                logger.critical(
                    f"HIGH REJECTION RATE ALERT: {rejection_count} rejections in 5 minutes",
                    extra={
                        'custom_dimensions': {
                            'event_type': 'alert',
                            'alert_type': 'high_rejection_rate',
                            'rejection_count': rejection_count,
                            'time_window_minutes': 5,
                            'threshold': 3,
                            'timestamp': datetime.now().isoformat(),
                            'severity': 'critical'
                        }
                    }
                )
        except Exception as e:
            logger.error(f"Failed to trigger alert in Azure Monitor: {e}")

    def log_error(self, error_message: str, error_type: str = 'general', **kwargs):
        """
        Enregistre une erreur dans Application Insights
        """
        if not self.enabled:
            return

        try:
            with self.tracer.start_as_current_span("error_occurred") as span:
                span.set_attributes({
                    "error.type": error_type,
                    "error.message": error_message,
                    **kwargs
                })

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

    def test_connection(self):
        """Test la connexion Azure Monitor en envoyant des métriques de test"""
        if not self.enabled:
            logger.warning("Azure Monitor not enabled, cannot test connection")
            return False

        try:
            # Envoyer une métrique de test
            test_counter = self.meter.create_counter(
                name="azure_monitor_connection_test",
                description="Test metric to verify Azure Monitor connection"
            )
            test_counter.add(1, {"test": "connection_check"})

            # Envoyer un log de test
            logger.info("Azure Monitor connection test", extra={
                'custom_dimensions': {
                    'event_type': 'connection_test',
                    'timestamp': datetime.now().isoformat(),
                    'test_status': 'success'
                }
            })

            return True
        except Exception as e:
            logger.error(f"Azure Monitor connection test failed: {e}")
            return False


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