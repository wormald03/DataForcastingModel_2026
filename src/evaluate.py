import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os


def evaluate_model(model, X_test, y_test):
    os.makedirs("outputs/plots", exist_ok=True)  # creates folder if it doesn't exist

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.savefig("outputs/plots/forecast_plot.png")
    plt.show()