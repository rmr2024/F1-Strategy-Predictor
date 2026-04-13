# F1-DRS: Race Strategy Predictor

F1-DRS predicts when Formula 1 drivers will pit based on telemetry, tyre state, and race context. It helps teams anticipate pit windows and optimize race strategy.

## Why It Matters

In F1, timing a pit stop can mean the difference between winning and finishing off the podium. Tyre degradation, fuel load, and track position all influence when a driver needs to box. This project uses machine learning to read patterns in race data and predict pit stops before they happen.

## How It Works

1. **Data Collection** — Load race telemetry from FastF1 (2021–2023 seasons)
2. **Feature Engineering** — Extract tyre life, stint progress, pace delta, fuel load, and track status
3. **Model Training** — XGBoost classifier trained on historical pit patterns
4. **Prediction** — Real-time pit window forecasting via Streamlit dashboard

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## Project Structure

```
F1-DRS/
├── src/               # Core modules
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
├── models/            # Trained model files
├── data/              # Raw/cached race data
├── app.py             # Streamlit dashboard
└── requirements.txt
```

## Requirements

- Python 3.9+
- FastF1, pandas, numpy, xgboost, streamlit, plotly, shap, joblib