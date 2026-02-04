# User Behavior Intelligence Platform (Churn)

Production-grade churn prediction system built on real retail transaction data. It includes ingestion, feature store, model training, explainability, monitoring, retraining, an API, and a product dashboard.

## Data Source
This project uses the UCI "Online Retail II" dataset (real e-commerce transactions). The ingestion pipeline downloads it programmatically via `ucimlrepo`.

## Quickstart
1. Start infrastructure.

```bash
docker compose up -d db prometheus grafana pushgateway
```

2. Create a virtual environment and install dependencies.

```bash
make setup
```

3. Fetch and clean data.

```bash
make fetch
make clean
```

4. Build features and load into the feature store.

```bash
make features
make load-db
```

5. Train the churn model and compute SHAP explanations.

```bash
make train
make explain
```

6. Run the API and dashboard.

```bash
make api
make dashboard
```

7. Run drift monitoring and retraining checks.

```bash
make monitor
make retrain
```

## API Endpoints
- `POST /predict`
- `POST /explain`
- `GET /health`
- `GET /metrics`

### Example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_ids": [12347, 12348]}'
```

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 12347}'
```

## Dashboard
- Streamlit dashboard: http://localhost:8501
- Grafana: http://localhost:3000 (Prometheus datasource is auto-provisioned)
- Prometheus: http://localhost:9090

## Architecture
- Ingestion: `src/ingest/`
- Feature engineering: `src/features/`
- Model training and explainability: `src/train/`
- API serving: `src/serve/`
- Monitoring: `src/monitor/`
- Retraining: `src/retrain/`
- Dashboard: `dashboard/app.py`

## Notes
- The churn label is defined as no purchases in the last `LABEL_WINDOW_DAYS`.
- Features are derived from a rolling history window of `FEATURE_WINDOW_DAYS`.
- PostgreSQL is used as the feature store and prediction store.
- Prometheus + Grafana provide monitoring for drift and API usage.
