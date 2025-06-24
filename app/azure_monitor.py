import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from collections import deque

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from azure.monitor.opentelemetry.exporter import (
    AzureMonitorLogExporter,
    AzureMonitorMetricExporter,
    AzureMonitorTraceExporter,
)
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


class AzureMonitor:
    """Gestionnaire pour Azure Application Insights avec OpenTelemetry"""

    def __init__(self, connection_string: str = None):
        """
        Initialise Azure Monitor

        Args:
            connection_string: Connection string Azure Application Insights
        """
        self.connection_string = connection_string or os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')

        if not self.connection_string:
            logger.warning("Azure Application Insights connection string not found. Monitoring disabled.")
            self.enabled = False
            return

        self.enabled = True

        # Configure resource
        resource = Resource.create({
            "service.name": "sentiment-analysis-api",
            "service.version": os.getenv('APP_VERSION', '1.0.0'),
        })

        # Configure metrics
        metric_reader = PeriodicExportingMetricReader(
            exporter=AzureMonitorMetricExporter(connection_string=self.connection_string),
            export_interval_millis=60000,  # Export every 60 seconds
        )

        metrics.set_meter_provider(MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        ))

        # Get meter
        self.meter = metrics.get_meter("sentiment_analysis")

        # Create custom metrics
        self.prediction_counter = self.meter.create_counter(
            name="prediction_count",
            description="Number of predictions made",
            unit="predictions"
        )

        self.rejection_counter = self.meter.create_counter(
            name="rejection_count",
            description="Number of rejected predictions",
            unit="rejections"
        )

        self.processing_time_histogram = self.meter.create_histogram(
            name="processing_time",
            description="Processing time for predictions",
            unit="ms"
        )

        # Configure tracing
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer_provider = trace.get_tracer_provider()
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                AzureMonitorTraceExporter(connection_string=self.connection_string)
            )
        )
        self.tracer = trace.get_tracer("sentiment_analysis")

        # Configure logging
        self.azure_logger = logging.getLogger('azure_telemetry')
        self.azure_logger.setLevel(logging.INFO)
        # Add a stream handler for local logging
        stream_handler = logging.StreamHandler()
        self.azure_logger.addHandler(stream_handler)

        # Create Azure Monitor exporter for logs
        # Note: We'll use the tracer and metrics we've already set up
        # instead of trying to use AzureMonitorLogExporter directly as a handler

        # Historique des rejets pour la détection d'anomalies
        self.rejection_history = deque(maxlen=100)

        logger.info("Azure Application Insights initialized successfully with OpenTelemetry")

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

        with self.tracer.start_as_current_span("log_prediction") as span:
            # Enregistrer la métrique
            self.prediction_counter.add(1, {
                "sentiment": "positive" if prediction.get('positive', 0) > 0.5 else "negative",
                "confidence_level": "high" if max(prediction.get('positive', 0),
                                                  prediction.get('negative', 0)) > 0.8 else "low"
            })

            # Ajouter des attributs au span
            span.set_attributes({
                "prediction.id": prediction_id,
                "text.length": len(text),
                "prediction.positive_score": prediction.get('positive', 0),
                "prediction.negative_score": prediction.get('negative', 0),
                "prediction.sentiment": "positive" if prediction.get('positive', 0) > 0.5 else "negative",
                "prediction.confidence": max(prediction.get('positive', 0), prediction.get('negative', 0))
            })

            # Log personnalisé
            logger.info(
                f"Prediction made: {prediction_id}",
                extra={
                    'custom_dimensions': {
                        'prediction_id': prediction_id,
                        'text_length': len(text),
                        'positive_score': prediction.get('positive', 0),
                        'negative_score': prediction.get('negative', 0),
                        'predicted_sentiment': 'positive' if prediction.get('positive', 0) > 0.5 else 'negative',
                        'confidence': max(prediction.get('positive', 0), prediction.get('negative', 0))
                    }
                }
            )

    def log_rejection(self, text: str, prediction: Dict[str, float], prediction_id: str):
        """
        Enregistre un rejet (tweet mal prédit) dans Application Insights

        Args:
            text: Texte du tweet mal prédit
            prediction: Prédiction qui a été rejetée
            prediction_id: ID de la prédiction
        """
        if not self.enabled:
            return

        with self.tracer.start_as_current_span("log_rejection") as span:
            # Enregistrer la métrique avec attributs
            self.rejection_counter.add(1, {
                "predicted_sentiment": "positive" if prediction.get('positive', 0) > 0.5 else "negative",
                "confidence_level": "high" if max(prediction.get('positive', 0),
                                                  prediction.get('negative', 0)) > 0.8 else "low"
            })

            # Ajouter à l'historique
            rejection_time = datetime.now()
            self.rejection_history.append(rejection_time)

            # Ajouter des attributs au span
            span.set_attributes({
                "rejection.prediction_id": prediction_id,
                "rejection.text_length": len(text),
                "rejection.positive_score": prediction.get('positive', 0),
                "rejection.negative_score": prediction.get('negative', 0),
                "rejection.predicted_sentiment": "positive" if prediction.get('positive', 0) > 0.5 else "negative",
                "rejection.confidence": max(prediction.get('positive', 0), prediction.get('negative', 0))
            })

            # Log d'avertissement pour les rejets
            logger.warning(
                f"Prediction rejected: {prediction_id}",
                extra={
                    'custom_dimensions': {
                        'prediction_id': prediction_id,
                        'rejected_text': text[:500],
                        'text_length': len(text),
                        'positive_score': prediction.get('positive', 0),
                        'negative_score': prediction.get('negative', 0),
                        'predicted_sentiment': 'positive' if prediction.get('positive', 0) > 0.5 else 'negative',
                        'confidence': max(prediction.get('positive', 0), prediction.get('negative', 0)),
                        'timestamp': rejection_time.isoformat()
                    }
                }
            )

            # Vérifier si on doit déclencher une alerte
            self._check_rejection_threshold()

    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """
        Enregistre des métriques de performance

        Args:
            operation: Nom de l'opération
            duration_ms: Durée en millisecondes
            **kwargs: Propriétés supplémentaires
        """
        if not self.enabled:
            return

        # Enregistrer la métrique de temps de traitement
        self.processing_time_histogram.record(duration_ms, {
            "operation": operation
        })

        with self.tracer.start_as_current_span(f"performance_{operation}") as span:
            span.set_attributes({
                "performance.operation": operation,
                "performance.duration_ms": duration_ms,
                **kwargs
            })

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
        with self.tracer.start_as_current_span("alert_triggered") as span:
            span.set_attributes({
                "alert.type": "high_rejection_rate",
                "alert.rejection_count": rejection_count,
                "alert.time_window": "5_minutes",
                "alert.threshold": 3
            })

            logger.critical(
                f"ALERT: High rejection rate detected! {rejection_count} rejections in 5 minutes",
                extra={
                    'custom_dimensions': {
                        'alert_type': 'high_rejection_rate',
                        'rejection_count': rejection_count,
                        'time_window': '5_minutes',
                        'threshold': 3,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            )

    def log_error(self, error_message: str, error_type: str = 'general', **kwargs):
        """
        Enregistre une erreur dans Application Insights
        """
        if not self.enabled:
            return

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
                        'error_type': error_type,
                        'timestamp': datetime.now().isoformat(),
                        **kwargs
                    }
                }
            )
