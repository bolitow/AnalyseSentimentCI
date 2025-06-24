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
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource
        from azure.monitor.opentelemetry.exporter import (
            AzureMonitorMetricExporter,
            AzureMonitorTraceExporter,
        )
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

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
        """Initialise Azure Monitor avec gestion d'erreurs"""
        # Configure resource
        resource = Resource.create({
            "service.name": os.getenv('OTEL_SERVICE_NAME', 'sentiment-analysis-api'),
            "service.version": os.getenv('APP_VERSION', '1.0.0'),
        })

        # Configure metrics avec timeout
        try:
            metric_exporter = AzureMonitorMetricExporter(
                connection_string=self.connection_string
            )
            metric_reader = PeriodicExportingMetricReader(
                exporter=metric_exporter,
                export_interval_millis=60000,  # Export every 60 seconds
            )

            metrics.set_meter_provider(MeterProvider(
                resource=resource,
                metric_readers=[metric_reader]
            ))
        except Exception as e:
            logger.error(f"Failed to configure metrics: {e}")
            raise

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

        # Configure tracing avec timeout
        try:
            trace.set_tracer_provider(TracerProvider(resource=resource))
            tracer_provider = trace.get_tracer_provider()

            trace_exporter = AzureMonitorTraceExporter(
                connection_string=self.connection_string
            )

            tracer_provider.add_span_processor(
                BatchSpanProcessor(trace_exporter)
            )
            self.tracer = trace.get_tracer("sentiment_analysis")
        except Exception as e:
            logger.error(f"Failed to configure tracing: {e}")
            raise

        # Historique des rejets pour la détection d'anomalies
        self.rejection_history = deque(maxlen=100)

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
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

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

        try:
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
        except Exception as e:
            logger.error(f"Failed to log rejection: {e}")

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

        try:
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
        except Exception as e:
            logger.error(f"Failed to log performance: {e}")

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
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")

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
                            'error_type': error_type,
                            'timestamp': datetime.now().isoformat(),
                            **kwargs
                        }
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log error: {e}")


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