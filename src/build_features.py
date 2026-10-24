"""Build lag, rolling, and calendar features for forecasting."""
import pandas as pd

LAGS = [1, 2, 3, 6, 12]
ROLLING = [3, 6, 12]
TARGETS = ["new_connections", "kwh_per_connection"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["region", "date"]).reset_index(drop=True)

    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    for target in TARGETS:
        g = df.groupby("region")[target]
        for lag in LAGS:
            df[f"{target}_lag{lag}"] = g.shift(lag)
        for w in ROLLING:
            df[f"{target}_roll{w}"] = g.transform(
                lambda s, w=w: s.shift(1).rolling(w).mean()
            )

    return df.dropna().reset_index(drop=True)


def feature_columns(target: str) -> list:
    cols = ["month", "quarter", "year"]
    cols += [f"{target}_lag{l}" for l in LAGS]
    cols += [f"{target}_roll{w}" for w in ROLLING]
    return cols


if __name__ == "__main__":
    df = pd.read_csv("data/raw/synthetic.csv", parse_dates=["date"])
    print(f"Input rows: {len(df)}")

    feat = build_features(df)
    print(f"After feature build: {len(feat)} rows")
    print(f"Columns: {list(feat.columns)}")
    print("\nFirst 3 rows:")
    print(feat.head(3))

    print("\nFeature columns for 'new_connections':")
    print(feature_columns("new_connections"))