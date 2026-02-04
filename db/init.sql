CREATE TABLE IF NOT EXISTS transactions (
    invoice_no TEXT,
    stock_code TEXT,
    description TEXT,
    quantity INTEGER,
    invoice_date TIMESTAMP,
    unit_price NUMERIC,
    customer_id INTEGER,
    country TEXT,
    is_cancelled BOOLEAN,
    total_price NUMERIC
);

CREATE TABLE IF NOT EXISTS features (
    customer_id INTEGER,
    last_invoice_date TIMESTAMP,
    first_invoice_date TIMESTAMP,
    invoices_count INTEGER,
    total_items NUMERIC,
    total_spend NUMERIC,
    unique_items INTEGER,
    avg_unit_price NUMERIC,
    active_days INTEGER,
    recency_days NUMERIC,
    tenure_days NUMERIC,
    avg_basket_size NUMERIC,
    spend_per_invoice NUMERIC,
    frequency_per_day NUMERIC,
    churn_label INTEGER,
    cutoff_date TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    churn_proba DOUBLE PRECISION,
    churn_label INTEGER,
    model_version TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drift_metrics (
    metric_id SERIAL PRIMARY KEY,
    feature_name TEXT,
    psi DOUBLE PRECISION,
    ks DOUBLE PRECISION,
    computed_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_features_customer ON features (customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_customer ON predictions (customer_id);
