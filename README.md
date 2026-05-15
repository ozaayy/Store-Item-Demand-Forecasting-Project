# Store Item Demand Forecasting with LSTM

End-to-end deep learning project that forecasts daily store-item sales using a Long Short-Term Memory (LSTM) neural network, with feature engineering and an interactive Streamlit deployment.

---

## Project Overview

**Business problem.** Retail chains need accurate short-term demand forecasts at the store-item level to optimize inventory, reduce stockouts, and minimize waste. This project builds a deep learning model that predicts daily sales for individual store-item combinations using historical transaction data.

**Approach.** A sequence-to-one LSTM architecture trained on engineered time-series features. The model captures seasonality, weekly patterns, and store-level effects, and is wrapped in a Streamlit web application for interactive forecasting.

**Deployment.** Fully reproducible: a saved model artifact, persisted scaler objects, and a single-command Streamlit launch.

---

## Technical Stack

- **Language:** Python 3.x
- **Deep Learning:** TensorFlow / Keras (LSTM)
- **Data Processing:** Pandas, NumPy
- **Feature Engineering:** Custom lag, rolling-window, and calendar features
- **Scaling:** Scikit-learn MinMaxScaler (persisted as `scalerX.save` / `scalerY.save` via joblib)
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit

---

## Feature Engineering

The model relies on engineered features rather than raw sequences alone:

- **Lag features** — sales values at t-1, t-7, t-30 to capture short-term and seasonal dependencies
- **Rolling statistics** — moving averages and standard deviations over multiple windows
- **Calendar features** — day-of-week, month, quarter, year, holiday flags
- **Store / item encoding** — categorical encoding of store and item identifiers
- **Trend decomposition** — separating long-term trend from seasonal noise

Feature engineering proved more impactful than model depth — a recurring pattern in time-series forecasting.

---

## Model Architecture

A stacked LSTM with dropout regularization:

```
Input(sequence_length, n_features)
  → LSTM(units=64, return_sequences=True)
  → Dropout(0.2)
  → LSTM(units=32)
  → Dropout(0.2)
  → Dense(units=16, activation='relu')
  → Dense(units=1)
```

- **Loss:** Mean Squared Error
- **Optimizer:** Adam
- **Validation:** Time-based train/validation split (no random shuffling — leakage-safe)

---

## Results

The model produces directionally accurate forecasts and learns weekly/seasonal patterns from the engineered features. Forecast plots and evaluation metrics are available in the deployed Streamlit app.

> *Performance metrics are dataset-dependent; the codebase is designed for reproducibility and experimentation rather than a single benchmark claim.*

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/ozaayy/Store-Item-Demand-Forecasting-Project.git
cd Store-Item-Demand-Forecasting-Project

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

The app loads the trained model and pre-fitted scalers, then accepts user input for store-item combinations to produce demand forecasts.

---

## Repository Structure

```
.
├── app.py                          # Streamlit entry point
├── demand_forecasting_app.py       # Core forecasting logic and model loading
├── scalerX.save                    # Persisted MinMaxScaler for features
├── scalerY.save                    # Persisted MinMaxScaler for target
├── pages/                          # Streamlit multi-page assets
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Key Learnings

- **Feature engineering > model complexity** for tabular time-series problems
- **Leakage-safe validation** (time-based splits) is non-negotiable in forecasting
- **Scaler persistence** is critical for inference parity between training and production
- **Streamlit** is a fast path from notebook to a stakeholder-facing demo

---

## About the Author

Built as part of a deep learning practitioner track (Miuul Data Science Bootcamp). My professional background is in large-scale program management across payments, banking and pension platforms — this project reflects ongoing interest in applying ML to forecasting and operational decision-making problems.

- **LinkedIn:** [linkedin.com/in/ozaayy](https://linkedin.com/in/ozaayy)
- **Website:** [www.ozanayyildiz.com](https://www.ozanayyildiz.com)

---

## License

This project is released under the MIT License. See `LICENSE` for details.
