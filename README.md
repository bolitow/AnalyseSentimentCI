# AnalyseSentimentCI
# Analyse de Sentiment

Application d'analyse de sentiment qui détermine si un texte exprime un sentiment positif ou négatif.

## Fonctionnalités

- **Analyse de sentiment** : Détermination de la polarité (positive/négative) d'un texte
- **Interface utilisateur intuitive** : Interface web conviviale pour soumettre des textes
- **API REST** : Endpoint pour l'intégration avec d'autres applications
- **Surveillance de performance** : Métriques de performance et endpoint de santé
- **Alertes par email** : Notification en cas de prédictions incorrectes consécutives

## Architecture

L'application est composée de deux parties principales :

1. **Backend (API Flask)** :
   - Endpoint de prédiction `/predict`
   - Endpoint de santé `/health`
   - Gestion des erreurs et logging
   - Métriques de performance

2. **Frontend (Streamlit)** :
   - Interface utilisateur intuitive
   - Visualisation des résultats
   - Exemples prédéfinis
   - Gestion des erreurs

## Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Configuration

Créez un fichier `.env` dans le dossier `app/` avec les variables suivantes :

```
# Configuration SMTP (pour les alertes)
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
EMAIL_FROM=your-email@example.com
EMAIL_TO=alerts@example.com
EMAIL_PASSWORD=your-password

# Modèles
MODEL_PATH=app/model/logistic_regression_model.pkl
VECTORIZER_PATH=app/model/tfidf_vectorizer.pkl

# API Configuration
API_URL=http://localhost:5000/predict
APP_VERSION=1.1.0
FLASK_DEBUG=False
```

## Démarrage

### Backend (API)

```bash
python app.py
```

L'API sera accessible à l'adresse : http://localhost:5000

### Frontend (Interface utilisateur)

```bash
streamlit run app/frontApp.py
```

L'interface sera accessible à l'adresse : http://localhost:8501

## Tests

Exécutez les tests avec pytest :

```bash
pytest
```

Pour générer un rapport de couverture :

```bash
pytest --cov=app
```

## Déploiement

L'application est configurée pour être déployée sur Heroku. Utilisez le fichier `Procfile` inclus.

## Améliorations récentes

- Interface utilisateur améliorée avec une meilleure expérience utilisateur
- Ajout de tests complets et d'une couverture de code
- Correction du bug de faux positifs dans la détection d'erreurs
- Ajout de métriques de performance et de logging
- Ajout d'un endpoint de santé pour la surveillance
- Optimisation des dépendances

## Licence

Ce projet est sous licence MIT.