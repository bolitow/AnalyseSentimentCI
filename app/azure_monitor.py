import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from collections import deque

from opentelemetry import metrics, trace
from azure.monitor.opentelemetry.exporter import (
    AzureMonitorLogExporter,
    AzureMonitorMetricExporter,
    AzureMonitorTraceExporter,
)
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

class AzureMonitor:
    """
    Client simplifié pour Application Insights via OpenTelemetry.
    Nécessite que configure_azure_monitor soit appelé en amont.
    """

    def __init__(self):
        connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
        if not connection_string:
            logger.warning("Azure Monitor non configuré: APPLICATIONINSIGHTS_CONNECTION_STRING manquant.")
            self.enabled = False
            return
        self.enabled = True

        # Ajoute les readers/exporters si nécessaire
        # Les providers ont déjà été configurés par configure_azure_monitor
        # Pour s'assurer que metrics exportent, on peut ajouter ceci si besoin:
        meter = metrics.get_meter_provider().get_meter(__name__)
        self.prediction_counter = meter.create_counter(
            name="prediction_count",
            description="Nombre de prédictions effectuées",
            unit="1"
        )
        self.rejection_counter = meter.create_counter(
            name="rejection_count",
            description="Nombre de rejets",
            unit="1"
        )
        self.processing_time_histogram = meter.create_histogram(
            name="processing_time",
            description="Temps de traitement des prédictions",
            unit="ms"
        )

        # Configure le logger pour logs personnalisés
        app_logger = logging.getLogger()
        exporter = AzureMonitorLogExporter(connection_string=connection_string)
        exporter.set_level(logging.INFO)
        app_logger.addHandler(exporter)
        logger.info("AzureMonitor initialisé")

    def log_prediction(self, text: str, prediction: Dict[str, Any], prediction_id: str):
        if not self.enabled:
            return
        # métrique
        self.prediction_counter.add(1, {
            "prediction.id": prediction_id,
            "sentiment": 'positive' if prediction.get('positive',0)>0.5 else 'negative'
        })
        # span
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("log_prediction") as span:
            span.set_attribute("prediction.id", prediction_id)
            span.set_attribute("text.length", len(text))

    def log_rejection(self, text: str, prediction: Dict[str, Any], prediction_id: str):
        if not self.enabled:
            return
        self.rejection_counter.add(1, {
            "prediction.id": prediction_id
        })
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("log_rejection") as span:
            span.set_attribute("rejection.id", prediction_id)

    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        if not self.enabled:
            return
        self.processing_time_histogram.record(duration_ms, {"operation": operation})
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"perf_{operation}") as span:
            span.set_attribute("duration_ms", duration_ms)

    def log_error(self, message: str, error_type: str = 'general', **kwargs):
        if not self.enabled:
            return
        logger.error(message)
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("log_error") as span:
            span.set_attribute("error.type", error_type)
            span.set_attribute("error.message", message)