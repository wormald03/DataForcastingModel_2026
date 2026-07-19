from src.data_loader import load_data
from src.features import engineer_features
from src.model import train_model
from src.evaluate import evaluate_model

df = load_data("data/DailyDelhiClimateTrain.csv")
df = engineer_features(df, target_col="meantemp")
model, X_test, y_test = train_model(df, target_col="meantemp")

# 4. Evaluate
evaluate_model(model, X_test, y_test)
