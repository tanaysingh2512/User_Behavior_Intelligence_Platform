.PHONY: setup fetch clean features load-db train explain monitor retrain api dashboard

PYTHON ?= python3

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

fetch:
	$(PYTHON) -m src.ingest.fetch_uci_online_retail

clean:
	$(PYTHON) -m src.ingest.clean_retail

features:
	$(PYTHON) -m src.features.build_features

load-db:
	$(PYTHON) -m src.ingest.load_transactions
	$(PYTHON) -m src.features.feature_store

train:
	$(PYTHON) -m src.train.train_model

explain:
	$(PYTHON) -m src.train.explain

monitor:
	$(PYTHON) -m src.monitor.monitor

retrain:
	$(PYTHON) -m src.retrain.retrain

api:
	uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py
