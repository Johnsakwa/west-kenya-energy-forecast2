"""Train XGBoost models for connections and demand."""
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import joblib

from src.build_features import build_features, feature_columns

HOLDOUT_MONTHS = 6


def time_split(df, months=HOLDOUT_MONTHS):
    cutoff = df["date"].max() - pd.DateOffset(months=months)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test


def train_one(df, target, model_path):
    feats = feature_columns(target)
    train, test = time_split(df)

    print(f"\n=== {target} ===")
    print(f"Train rows: {len(train)} | Test rows: {len(test)}")
    print(f"Train dates: {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"Test dates:  {test['date'].min().date()} to {test['date'].max().date()}")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(train[feats], train[target])

    preds = model.predict(test[feats])
    mape = mean_absolute_percentage_error(test[target], preds)
    print(f"Holdout MAPE: {mape:.2%}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feats, "mape": mape}, model_path)
    print(f"Saved model to {model_path}")
    return mape


def main():
    df = pd.read_csv("data/raw/synthetic.csv", parse_dates=["date"])
    feat = build_features(df)
    print(f"Feature rows: {len(feat)}")

    mape_conn = train_one(feat, "new_connections", "models/connections.pkl")
    mape_demand = train_one(feat, "kwh_per_connection", "models/demand.pkl")

    print("\n=== Summary ===")
    print(f"Connections model MAPE: {mape_conn:.2%}")
    print(f"Demand model MAPE:      {mape_demand:.2%}")


if __name__ == "__main__":
    main()