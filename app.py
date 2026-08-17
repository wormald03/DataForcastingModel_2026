import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page config
st.set_page_config(page_title="Time Series Forecaster", layout="wide")
st.title("📈 Time Series Forecaster")
st.markdown("Upload a CSV, pick a column, and forecast it with a Random Forest model.")

# Sidebar controls
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
date_col = st.sidebar.text_input("Date column name", value="date")
target_col = st.sidebar.text_input("Target column to forecast", value="meantemp")
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2)
n_estimators = st.sidebar.slider("Number of trees", 50, 500, 100, step=50)

# Main logic
if uploaded_file is not None:
    # Load
    df = pd.read_csv(uploaded_file)
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))

    st.subheader("Full Time Series")
    st.line_chart(df[target_col])

    # Feature engineering
    from src.features import engineer_features
    df = engineer_features(df, target_col=target_col)

    # Train
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    feature_cols = [
        "lag_1", "lag_7",
        "rolling_mean_7", "rolling_std_7",
        "month", "day_of_week"
    ]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")

    # Forecast chart
    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test.values, label="Actual", linewidth=1.5)
    ax.plot(predictions, label="Predicted", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set_xlabel("Days (test period)")
    ax.set_ylabel(target_col)
    st.pyplot(fig)

    # Feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

    # Download predictions
    st.subheader("Download Predictions")
    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    }, index=y_test.index)
    csv = results_df.to_csv().encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file in the sidebar to get started.")