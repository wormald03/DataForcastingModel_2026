# Time Series Forecaster

A machine learning project that forecasts daily temperatures using a Random Forest Regressor, built in Python with a clean modular pipeline and an interactive Streamlit web app.

**[Try the live demo here](http://192.168.0.33:8501)**

---

## Project Overview

This project demonstrates a complete end-to-end machine learning forecasting pipeline — from raw data ingestion through to model evaluation, visualisation, and a deployed interactive web application. The model is trained on the Delhi Daily Climate dataset and predicts daily mean temperature using engineered lag features, rolling statistics, and calendar-based signals.

The project is structured with clean separation of concerns across multiple modules, making it easy to swap datasets, features, or models independently.

---

## Features

- Modular pipeline split across dedicated source files
- Feature engineering with lag values, rolling statistics, and calendar signals
- Random Forest Regressor with configurable hyperparameters
- Evaluation with MAE and RMSE metrics
- Interactive Streamlit web app — upload any CSV and forecast any column
- Actual vs Predicted visualisation rendered in the browser
- Feature importance chart showing which signals the model relied on most
- Download predictions as a CSV directly from the app

---

## Results

Trained and evaluated on the Delhi Daily Climate dataset (2013–2017):

| Metric | Score |
|--------|-------|
| MAE    | 1.17°C |
| RMSE   | 1.55°C |

On average the model predicts daily temperature to within **1.17°C** — comparable to professional short-range weather forecasting benchmarks.

---

## Project Structure

```
forecasting_project/
│
├── data/                        # Raw dataset (not on Git)
├── notebooks/
│   └── exploration.ipynb        # Exploratory data analysis
├── src/
│   ├── data_loader.py           # Data ingestion and cleaning
│   ├── evaluate.py              # Evaluation metrics and plots
│   ├── features.py              # Feature engineering
│   └── model.py                 # Model training and splitting
├── outputs/
│   └── plots/                   # Saved forecast visualisations
├── app.py                       # Streamlit web application
├── main.py                      # CLI pipeline entry point
├── requirements.txt             # Project dependencies
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/time-series-forecaster.git
cd time-series-forecaster
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your dataset
Download the Delhi Daily Climate dataset from Kaggle and place `DailyDelhiClimateTrain.csv` inside the `data/` folder.

### 5a. Run the CLI pipeline
```bash
python main.py
```

### 5b. Run the Streamlit app
```bash
streamlit run app.py
```

---

## Streamlit App

The web app allows anyone to use the forecasting pipeline without touching any code:

- **Upload** any time series CSV
- **Select** the date column and target column to forecast
- **Adjust** test set size and number of trees via sliders
- **View** the full time series, actual vs predicted chart, and feature importance
- **Download** predictions as a CSV

---

## Features Used

| Feature | Description |
|---|---|
| `lag_1` | Previous day's value |
| `lag_7` | Value from 7 days ago |
| `rolling_mean_7` | 7-day rolling average |
| `rolling_std_7` | 7-day rolling standard deviation |
| `month` | Month of year (1–12) — captures annual seasonality |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) — captures weekly patterns |

---

## Model

- **Algorithm:** Random Forest Regressor
- **Library:** scikit-learn
- **Trees:** 100 estimators
- **Train/Test Split:** 80% / 20% (chronological — no shuffling to prevent data leakage)
- **Evaluation Metrics:** MAE, RMSE

---

## Future Improvements

- [ ] Add XGBoost as an alternative model and compare performance
- [ ] Implement time series cross-validation
- [ ] Add multivariate forecasting using humidity and wind speed as additional inputs
- [ ] Explore ARIMA and Facebook Prophet
- [ ] Add confidence intervals to the forecast plot

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas | Data manipulation |
| numpy | Numerical computing |
| scikit-learn | ML model and evaluation |
| matplotlib | Visualisation |
| Streamlit | Interactive web application |
| statsmodels | Classical time series reference |

---

## Dataset

**Delhi Daily Climate** — daily weather data for Delhi from January 2013 to April 2017, containing mean temperature, humidity, wind speed, and mean pressure.

Source: [Kaggle — sumanthvrao/daily-climate-time-series-data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)

> The raw data file is not tracked in this repository. Download it from Kaggle and place it in the `data/` folder.

---

## Author

**Ewan**
[GitHub](https://github.com/wormald03) · [LinkedIn](https://linkedin.com/in/ewan-wormald)