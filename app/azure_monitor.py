import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from collections import deque

from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure import metrics_exporter
from opencensus.stats import aggregation as aggregation_module
from opencensus.stats import measure as measure_module
from opencensus.stats import stats as stats_module
from opencensus.stats import view as view_module
from opencensus.tags import tag_map as tag_map_module
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer

logger = logging.getLogger(__name__)


class AzureMonitor:
    """Gestionnaire pour Azure Application Insights"""

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

        # Initialiser le logger Azure
        self.azure_logger = logging.getLogger('azure_telemetry')
        self.azure_logger.setLevel(logging.INFO)
        self.azure_logger.addHandler(AzureLogHandler(connection_string=self.connection_string))

        # Initialiser le tracer pour les traces distribuées
        self.tracer = Tracer(
            exporter=AzureExporter(connection_string=self.connection_string),
            sampler=ProbabilitySampler(1.0)
        )

        # Initialiser les métriques
        self.stats = stats_module.stats
        self.view_manager = self.stats.view_manager

        # Configurer l'exportateur de métriques
        exporter = metrics_exporter.new_metrics_exporter(
            connection_string=self.connection_string
        )
        self.view_manager.register_exporter(exporter)

        # Créer les mesures personnalisées
        self.prediction_count_measure = measure_module.MeasureInt(
            "prediction_count", "Nombre de prédictions", "predictions"
        )
        self.rejection_count_measure = measure_module.MeasureInt(
            "rejection_count", "Nombre de rejets", "rejections"
        )

        # Créer les vues pour les métriques
        prediction_view = view_module.View(
            "prediction_count_view",
            "Nombre de prédictions par minute",
            [],
            self.prediction_count_measure,
            aggregation_module.CountAggregation()
        )

        rejection_view = view_module.View(
            "rejection_count_view",
            "Nombre de rejets par minute",
            [],
            self.rejection_count_measure,
            aggregation_module.CountAggregation()
        )

        # Enregistrer les vues
        self.view_manager.register_view(prediction_view)
        self.view_manager.register_view(rejection_view)

        # Historique des rejets pour la détection d'anomalies
        self.rejection_history = deque(maxlen=100)  # Garde les 100 derniers rejets

        logger.info("Azure Application Insights initialized successfully")

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

        with self.tracer.span(name='log_prediction') as span:
            # Enregistrer la métrique
            mmap = self.stats.stats_recorder.new_measurement_map()
            mmap.measure_int_put(self.prediction_count_measure, 1)
            mmap.record()

            # Log personnalisé avec propriétés
            properties = {
                'prediction_id': prediction_id,
                'text_length': len(text),
                'positive_score': prediction.get('positive', 0),
                'negative_score': prediction.get('negative', 0),
                'predicted_sentiment': 'positive' if prediction.get('positive', 0) > 0.5 else 'negative',
                'confidence': max(prediction.get('positive', 0), prediction.get('negative', 0))
            }

            self.azure_logger.info(
                f"Prediction made: {prediction_id}",
                extra={
                    'custom_dimensions': properties
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

        with self.tracer.span(name='log_rejection') as span:
            # Enregistrer la métrique
            mmap = self.stats.stats_recorder.new_measurement_map()
            mmap.measure_int_put(self.rejection_count_measure, 1)
            mmap.record()

            # Ajouter à l'historique
            rejection_time = datetime.now()
            self.rejection_history.append(rejection_time)

            # Propriétés détaillées pour l'analyse
            properties = {
                'prediction_id': prediction_id,
                'rejected_text': text[:500],  # Limiter la taille
                'text_length': len(text),
                'positive_score': prediction.get('positive', 0),
                'negative_score': prediction.get('negative', 0),
                'predicted_sentiment': 'positive' if prediction.get('positive', 0) > 0.5 else 'negative',
                'confidence': max(prediction.get('positive', 0), prediction.get('negative', 0)),
                'timestamp': rejection_time.isoformat()
            }

            # Log d'avertissement pour les rejets
            self.azure_logger.warning(
                f"Prediction rejected: {prediction_id} - Text: {text[:100]}...",
                extra={
                    'custom_dimensions': properties
                }
            )

            # Vérifier si on doit déclencher une alerte
            self._check_rejection_threshold()

    def _check_rejection_threshold(self):
        """
        Vérifie si le seuil de rejets est dépassé (3 rejets en 5 minutes)
        et déclenche une alerte si nécessaire
        """
        if not self.enabled or len(self.rejection_history) < 3:
            return

        # Obtenir l'heure actuelle et le seuil de 5 minutes
        now = datetime.now()
        five_minutes_ago = now - timedelta(minutes=5)

        # Compter les rejets des 5 dernières minutes
        recent_rejections = [
            timestamp for timestamp in self.rejection_history
            if timestamp > five_minutes_ago
        ]

        if len(recent_rejections) >= 3:
            # Déclencher une alerte critique
            self._trigger_alert(len(recent_rejections))

    def _trigger_alert(self, rejection_count: int):
        """
        Déclenche une alerte dans Application Insights

        Args:
            rejection_count: Nombre de rejets dans la fenêtre de temps
        """
        properties = {
            'alert_type': 'high_rejection_rate',
            'rejection_count': rejection_count,
            'time_window': '5_minutes',
            'threshold': 3,
            'timestamp': datetime.now().isoformat()
        }

        # Log critique pour déclencher les alertes Azure
        self.azure_logger.critical(
            f"ALERT: High rejection rate detected! {rejection_count} rejections in 5 minutes",
            extra={
                'custom_dimensions': properties
            }
        )

    def log_error(self, error_message: str, error_type: str = 'general', **kwargs):
        """
        Enregistre une erreur dans Application Insights

        Args:
            error_message: Message d'erreur
            error_type: Type d'erreur
            **kwargs: Propriétés supplémentaires
        """
        if not self.enabled:
            return

        properties = {
            'error_type': error_type,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        self.azure_logger.error(
            error_message,
            extra={
                'custom_dimensions': properties
            }
        )

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

        properties = {
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        self.azure_logger.info(
            f"Performance metric: {operation} took {duration_ms}ms",
            extra={
                'custom_dimensions': properties
            }
        )