# Time Series Forecasting with Machine Learning

A machine learning project that forecasts time series data using a
Random Forest Regressor, built in Python with a modular pipeline.

---

## Project Overview

This project demonstrates an end-to-end machine learning forecasting
pipeline — from raw data ingestion through to model evaluation and
visualisation. The goal is to predict future values in a time series
dataset using lag features, rolling statistics, and calendar-based signals.

---

## Project Structure
```
forecasting_project/
│  
├── data/                   # Raw dataset (not tracked by Git)  
├── notes/                  # Exploratory analysis notebooks  
├── src/  
│   ├── data_loader.py      # Data ingestion and cleaning  
│   ├── evaluate.py         # Evaluation metrics and plots  
│   ├── features.py         # Feature engineering  
│   └── model.py            # Model training  
├── outputs/  
│   └── plots/              # Saved forecast visualisations  
├── main.py                 # Main pipeline entry point  
├── requirements.txt        # Project dependencies  
└── README.md               # Project documentation  
```
---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/wormald03/DataForcastingModel_2026.git
cd DataForcastingModel_2026
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your dataset
Place your CSV file inside the `data/` folder and ensure it contains
a `date` column and a target value column.

### 5. Run the pipeline
```bash
python main.py
```

---

## Features Used

| Feature | Description |
|---|---|
| `lag_1` | Previous day's value |
| `lag_7` | Value from 7 days ago |
| `rolling_mean_7` | 7-day rolling average |
| `rolling_std_7` | 7-day rolling standard deviation |
| `month` | Month of year (1–12) |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) |

---

## Model

- **Algorithm:** Random Forest Regressor
- **Library:** scikit-learn
- **Train/Test Split:** 80% / 20% (chronological, no shuffling)
- **Evaluation Metrics:** MAE, RMSE

---

## Results

| Metric | Score |
|---|-------|
| MAE | 1.17  |
| RMSE | 1.55  |

> Results have been updated

---

## Future Improvements

- Experiment with API implementation
- Add cross-validation using time series splits
- Build a Streamlit dashboard for interactive forecasting

---

## Tech Stack

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- statsmodels

---

## Author

**Ewan Wormald**  
[GitHub](https://github.com/wormald03) · 
[LinkedIn](https://linkedin.com/in/ewan-wormald)