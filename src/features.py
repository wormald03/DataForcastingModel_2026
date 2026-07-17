def engineer_features(df, target_col):
    df["lag_1"] = df[target_col].shift(1)
    df["lag_7"] = df[target_col].shift(7)
    df["rolling_mean_7"] = df[target_col].rolling(window=7).mean()
    df["rolling_std_7"] = df[target_col].rolling(window=7).std()
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df.dropna(inplace=True)
    return df