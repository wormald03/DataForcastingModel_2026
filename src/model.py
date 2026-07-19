from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_model(df, target_col):
    feature_cols = [
        "lag_1", "lag_7",
        "rolling_mean_7", "rolling_std_7",
        "month", "day_of_week"
    ]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test