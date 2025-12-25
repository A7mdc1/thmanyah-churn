# Thmanyah Customer Churn Prediction System

## 1. Project Overview

This system predicts customer churn for a music streaming platform using event log data. The challenge lies in defining churn from implicit user behavior patterns rather than explicit cancellations, addressing severe class imbalance (77% churn rate), and preventing temporal data leakage in time-series event data.

Key technical challenges include proper churn definition from subscription downgrades and inactivity patterns, feature engineering with temporal cutoffs to prevent future information leakage, class imbalance handling via class weights and PR-AUC optimization, and continuous monitoring for data and concept drift detection.

---

## 2. End-to-End System Architecture

The system implements a complete MLOps pipeline:

**Data ingestion** (`data.py`) → **Feature engineering** (`feature_engineering.py`) → **Model training** (`model.py`) → **MLflow tracking** (integrated) → **Model artifacts** (serialized models) → **FastAPI inference** (`api.py`) → **Monitoring** (`monitoring.py`, `monitor_job.py`) → **Retraining** (`retraining.py`)

- **Data loader**: Processes JSON event logs, handles encoding issues, basic preprocessing
- **Feature engineer**: Creates 67+ temporal, behavioral, and subscription features with data leakage prevention
- **Model trainer**: Time-based train/test splits, handles class imbalance, evaluates with business metrics
- **API service**: FastAPI with single/batch prediction endpoints, input validation, risk scoring
- **Drift monitor**: Population Stability Index (PSI) for data drift, proxy metrics for concept drift
- **Retraining pipeline**: Schedule-based, performance-based, and drift-triggered model updates

---

## 3. Data & Churn Definition

### Input Data Format
JSON event logs with user interactions: `{"userId": "123", "ts": "2018-10-01T00:00:00Z", "page": "NextSong", "level": "paid", ...}`

### Churn Definition Challenges
- No explicit cancellation events for most users
- Subscription level changes (paid ↔ free) complicate binary classification
- Need to distinguish temporary inactivity from true churn
- Business impact varies by user subscription tier

### Implemented Churn Criteria
A user is considered churned if they meet **any** of these criteria:

1. **Explicit Churn**: "Submit Downgrade" events (68.1% of churned users)
2. **Inactivity Churn**: >30 consecutive days inactive before dataset end (8.8% of churned users)
3. **Subscription Churn**: Downgrade from paid to free + subsequent inactivity (rare pattern)

### Business Rationale
- **30-day threshold**: Balances false positives with actionable timeframe for intervention
- **Explicit signals prioritized**: Clear user intent to cancel subscription
- **Subscription transitions**: Paid→free often precedes full churn, making it a leading indicator
- **Time-based approach**: Prevents data leakage by using only historical information

---

## 4. Class Imbalance Handling

### Class Distribution
- **Churned**: 77% (174/226 users)
- **Active**: 23% (52/226 users)

### Why Accuracy is Inappropriate
With 77% churn rate, a naive classifier predicting "churn" for all users achieves 77% accuracy but provides zero business value. The cost of missing true churns (false negatives) far exceeds incorrectly flagging active users (false positives).

### Techniques Used
- **Class weights**: `class_weight='balanced'` in scikit-learn models automatically adjusts for imbalance
- **Evaluation metrics**: PR-AUC (primary), ROC-AUC, F1-score, Recall@80%/90% precision

### Metric Selection Rationale
- **PR-AUC**: Focuses on precision-recall tradeoff, more informative than ROC-AUC for imbalanced data
- **Recall@K**: Business-relevant metric for campaign capacity constraints (e.g., can only target 20% of users)
- **F1-score**: Balances precision and recall for threshold selection

---

## 5. Feature Engineering & Data Leakage Prevention

### Feature Categories (67+ features total)
- **Temporal**: days_since_registration, days_since_last_event, activity_span
- **Behavioral**: total_events, songs_played, thumbs_up/down, session_patterns
- **Subscription**: current_level_paid, downgrade_events, subscription_changes
- **Engagement**: thumbs_up_ratio, add_to_playlist_events, help_page_visits
- **Trends**: activity_decline_7d/14d/30d, recent_activity_ratios

### Data Leakage Prevention
**Temporal cutoff**: All features use `prediction_cutoff` parameter to ensure no future information
**Time-based splits**: Train/test split by `days_since_registration` rather than random sampling
**Feature validation**: Each feature is computed using only data available at prediction time

### Time-Based Train/Test Splitting
Split at 80th percentile of `days_since_registration` to simulate real-world temporal constraints where newer users are predicted based on patterns from earlier users.

---

## 6. Model Training & Evaluation

### Models Implemented
- **Logistic Regression** (best performer): Simple, interpretable, class_weight='balanced'
- **Gradient Boosting**: Handles non-linear patterns, feature interactions
- **Random Forest**: Robust to outliers, provides feature importance

### Model Selection
**Best model**: Logistic Regression with StandardScaler
- **PR-AUC**: 98.2%
- **ROC-AUC**: 91.6%
- **F1-Score**: 86.2%
- **Precision**: 100%
- **Recall**: 75.7%
- **Features**: 69 engineered features

### Business Metrics
- **Recall@90% Precision**: 72% (high-confidence predictions)
- **Recall@80% Precision**: 75% (broader campaign targeting)
- **Error Analysis**: 0 false positives, 9 false negatives

---

## 7. FastAPI Service Architecture

### Endpoints
- `POST /predict`: Single user prediction with risk scoring
- `POST /predict/batch`: Batch predictions (up to 1000 users)
- `GET /health`: Service health check
- `GET /model/info`: Model metadata and performance

### Input Validation
Pydantic models enforce feature schema:
```python
class UserFeatures(BaseModel):
    userId: str
    days_since_registration: float
    total_events: float
    songs_played: float
    current_level_paid: int  # 0 or 1
    # ... additional features
```

### Output Format
```json
{
  "userId": "123",
  "churn_probability": 0.0052,
  "churn_prediction": false,
  "risk_level": "low",
  "prediction_timestamp": "2025-12-25T14:46:54.082564"
}
```

---

## 8. MLflow Integration

### Experiment Tracking
- **Experiment**: "thmanyah-churn"
- **Runs**: Each model training session logged with parameters, metrics, artifacts
- **Model registry**: Best models saved with versioning

### Logged Information
- **Parameters**: model_type, train_size, test_size, split_strategy, class_balancing
- **Metrics**: pr_auc, roc_auc, f1_score, precision, recall, recall_at_90_precision
- **Artifacts**: trained model pipeline, feature importance plots

### Model Versioning
**Current model**: `best_churn_model_logistic_regression.pkl`
**Metadata**: `models/model_info.json` with 69 feature list and performance benchmarks
**Features**: 69 engineered features including temporal, behavioral, subscription, and demographic attributes

---

## 9. Monitoring System

### 9.1 Data Drift Detection
**Method**: Population Stability Index (PSI) on engineered features
**Features monitored**: Top 10 numeric features by importance
**Threshold**: PSI > 0.1 triggers drift alert
**Frequency**: Continuous monitoring via `monitor_job.py`

Persisted drift reports in `reports/drift_report_{timestamp}.json`

### 9.2 Concept Drift Detection
**Challenge**: True churn labels arrive with 30+ day delay
**Approach**: Proxy metrics for performance degradation
- Event rate changes (sudden drops in user activity)
- Subscription level distribution shifts
- Behavior pattern anomalies

**Rationale**: In production, concept drift detection must operate without ground truth labels. Activity patterns provide early signals of model degradation.

### Monitoring Commands
```bash
# Run monitoring cycle
python -m src.monitor_job

# Check performance logs
cat logs/performance_log.json
```

---

## 10. Retraining Strategy

### Retraining Triggers
1. **Schedule-based**: Every 7 days minimum
2. **Performance-based**: >10% degradation in proxy metrics
3. **Data drift-based**: PSI > 0.1 on key features

### Incremental Data Handling
- **Data validation**: Check minimum volume, quality thresholds
- **Feature consistency**: Ensure new data matches training schema
- **Model comparison**: Only deploy if new model outperforms current by >2% PR-AUC

### Model Deployment Gating
- Performance validation on holdout set
- A/B testing capability (infrastructure ready)
- Automated rollback if performance degrades

```bash
# Trigger retraining
python -m src.retraining
```

---

## 11. Project Structure

```
thmanyah-churn/
├── src/
│   ├── data.py              # Event log loading and preprocessing
│   ├── churn_definition.py  # Churn criteria and labeling logic
│   ├── feature_engineering.py # 67+ features with leakage prevention
│   ├── model.py             # Training pipeline with time-based splits
│   ├── api.py               # FastAPI service with batch support
│   ├── monitoring.py        # Drift detection classes
│   ├── monitor_job.py       # Continuous monitoring job
│   ├── retraining.py        # Automated retraining pipeline
│   └── config.py            # Centralized path configuration
├── notebooks/
│   └── 01_data_exploration.ipynb # Comprehensive EDA
├── data/                    # JSON event logs (gitignored)
├── models/                  # Trained model artifacts
├── logs/                    # Performance and monitoring logs
├── reports/                 # Drift detection reports
├── configs/                 # YAML configuration files
├── pyproject.toml           # Python dependencies (uv)
├── Makefile                 # Common commands
├── Dockerfile               # Container configuration
└── README.md                # This file
```

---

## 12. How to Run the System

### Installation
```bash
# Install uv and sync dependencies
make install
```

### Train Model
```bash
make train
```

### Launch MLflow UI
```bash
uv run mlflow ui --port 5000
# Browse to http://localhost:5000
```

### Start API Service
```bash
make run
# or: uv run uvicorn src.api:app --reload --port 8000
```

### Run Monitoring
```bash
make monitor
# or: uv run python -m src.monitor_job
```

### Trigger Retraining
```bash
uv run python -m src.retraining
```

### Docker Deployment
```bash
make docker-build
make docker-run
# Service available at http://localhost:8000
```

---

## 13. Design Decisions & Trade-offs

### PSI vs Other Drift Metrics
**Chosen**: Population Stability Index (PSI)
**Rationale**: Industry standard for production monitoring, interpretable thresholds, works well with binned features
**Trade-off**: Less sensitive to subtle distributional changes compared to Kolmogorov-Smirnov test

### Proxy Concept Drift vs Labeled Drift
**Chosen**: Event rate and behavior pattern changes as drift proxies
**Rationale**: True churn labels arrive 30+ days late, making real-time concept drift detection impossible
**Trade-off**: Proxies may miss subtle model degradation that only appears in prediction accuracy

### Class Weights vs Resampling
**Chosen**: Scikit-learn's `class_weight='balanced'`
**Rationale**: Maintains original data distribution, computationally efficient, well-integrated
**Trade-off**: Less control compared to custom SMOTE/undersampling approaches

### Time-Based Split vs Random Split
**Chosen**: Time-based split by `days_since_registration`
**Rationale**: Prevents data leakage, mimics real-world deployment constraints
**Trade-off**: Smaller effective training set, potential for temporal bias in model evaluation

---
