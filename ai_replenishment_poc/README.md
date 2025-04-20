# FMCG

┌──────────────────────┐
│     Data Sources     │
│ (CSV, API, S3, etc.) │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│   Data Ingestion     │ ←─ Pandas / PySpark
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Feature Engineering  │ ←─ Time features, holidays, promotions
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ ML Forecasting Model │ ←─ XGBoost / RandomForest / Prophet
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Replenishment Logic  │ ←─ Reorder point, EOQ, safety stock
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Reporting Dashboard  │ ←─ Streamlit / Flask
└──────────────────────┘


ai_replenishment_poc/
│
├── data/                      # For raw CSVs or simulated data
├── output/                    # Model outputs / replenishment plans
├── utils/
│   └── holidays.py            # Add local holiday logic here
│
├── data_ingestion.py
├── feature_engineering.py
├── forecast_model.py
├── replenishment.py
├── app.py                     # Optional dashboard (e.g., Streamlit/Flask)
├── config.yaml                # Config settings
└── main.py                    # Orchestrates all modules

ai_replenishment_poc/
├── app.py
├── templates/
│   └── dashboard.html
├── data/
│   └── sample_sales.csv
├── output/
├── data_ingestion.py
├── feature_engineering.py
├── forecast_model.py
├── replenishment.py
└── main.py