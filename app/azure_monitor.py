import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from collections import deque

from opentelemetry import metrics, trace

logger = logging.getLogger(__name__)

class AzureMonitor:
    """
    Client simplifié pour Application Insights via OpenTelemetry.
    Nécessite que configure_azure_monitor soit appelé en amont.
    """
    def __init__(self):
        self.connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if not self.connection_string:
            logger.warning("APPLICATIONINSIGHTS_CONNECTION_STRING non configurée. Monitoring désactivé.")
            self.enabled = False
            return
        self.enabled = True

        # Récupère le meter et le tracer configurés par configure_azure_monitor
        self.meter = metrics.get_meter("sentiment_analysis")
        self.tracer = trace.get_tracer("sentiment_analysis")

        # Création des métriques personnalisées
        self.prediction_counter = self.meter.create_counter(
            name="prediction_count",
            description="Nombre de prédictions effectuées",
            unit="1"
        )
        self.rejection_counter = self.meter.create_counter(
            name="rejection_count",
            description="Nombre de prédictions rejetées",
            unit="1"
        )
        self.processing_time_histogram = self.meter.create_histogram(
            name="processing_time",
            description="Temps de traitement en ms",
            unit="ms"
        )

        # Historique des rejets pour détection d'alerte
        self.rejection_history = deque(maxlen=100)

        logger.info("AzureMonitor initialisé. Métriques et traces activées.")

    def log_prediction(self, text: str, prediction: Dict[str, Any], prediction_id: str) -> None:
        if not self.enabled:
            return
        with self.tracer.start_as_current_span("log_prediction") as span:
            # Incrémente compteur
            self.prediction_counter.add(1, {
                "sentiment": "positive" if prediction.get('positive', 0) > 0.5 else "negative",
                "confidence": round(prediction.get('positive', 0), 3)
            })
            # Attributs de trace
            span.set_attributes({
                "prediction.id": prediction_id,
                "text.length": len(text)
            })

    def log_rejection(self, text: str, prediction: Dict[str, Any], prediction_id: str) -> None:
        if not self.enabled:
            return
        with self.tracer.start_as_current_span("log_rejection") as span:
            self.rejection_counter.add(1, {
                "sentiment": "positive" if prediction.get('positive', 0) > 0.5 else "negative",
                "confidence": round(prediction.get('positive', 0), 3)
            })
            span.set_attributes({
                "rejection.id": prediction_id,
                "text.length": len(text)
            })
            # Historique
            now = datetime.utcnow()
            self.rejection_history.append(now)
            # Vérification seuil 3 rejets en 5 min
            self._check_rejection_threshold()

    def log_performance(self, operation: str, duration_ms: float, **attrs) -> None:
        if not self.enabled:
            return
        # Enregistre histogramme
        self.processing_time_histogram.record(duration_ms, {"operation": operation})
        with self.tracer.start_as_current_span(f"perf_{operation}") as span:
            span.set_attributes({"duration_ms": duration_ms, **attrs})

    def log_error(self, message: str, error_type: str = 'general', **attrs) -> None:
        if not self.enabled:
            return
        with self.tracer.start_as_current_span("error_occurred") as span:
            span.set_attributes({"error.type": error_type, **attrs})
        logger.error(message)

    def _check_rejection_threshold(self) -> None:
        if len(self.rejection_history) < 3:
            return
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=5)
        recent = [t for t in self.rejection_history if t >= window_start]
        if len(recent) >= 3:
            # Trace event d'alerte
            with self.tracer.start_as_current_span("alert_rejections") as span:
                span.set_attributes({"count": len(recent)})
            logger.critical(f"ALERTE: {len(recent)} rejets en 5 minutes")