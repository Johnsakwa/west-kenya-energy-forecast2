"""Generate 12-month forecasts under Base / Accelerated / Constrained scenarios."""
import pandas as pd
import joblib

from src.build_features import build_features, feature_columns

SCENARIOS = {
    "base":        {"connections_mult": 1.00, "demand_mult": 1.00},
    "accelerated": {"connections_mult": 1.25, "demand_mult": 1.10},
    "constrained": {"connections_mult": 0.85, "demand_mult": 0.95},
}

HORIZON = 12


def forecast(df, horizon=HORIZON):
    conn_bundle = joblib.load("models/connections.pkl")
    dem_bundle = joblib.load("models/demand.pkl")
    conn_model = conn_bundle["model"]
    dem_model = dem_bundle["model"]
    conn_feats = feature_columns("new_connections")
    dem_feats = feature_columns("kwh_per_connection")

    all_results = []

    for scenario, mult in SCENARIOS.items():
        print(f"\nForecasting scenario: {scenario}")
        history = df.copy()

        for h in range(1, horizon + 1):
            next_date = history["date"].max() + pd.DateOffset(months=1)
            last_month = history.sort_values("date").groupby("region").tail(1).copy()
            last_month["date"] = next_date

            combined = pd.concat([history, last_month], ignore_index=True)
            feats = build_features(combined)
            step = feats[feats["date"] == next_date].copy()

            if len(step) == 0:
                print(f"  Step {h}: no rows produced, stopping")
                break

            step["new_connections_pred"] = (
                conn_model.predict(step[conn_feats]) * mult["connections_mult"]
            )
            step["kwh_per_conn_pred"] = (
                dem_model.predict(step[dem_feats]) * mult["demand_mult"]
            )
            step["total_connections_pred"] = (
                step["total_connections"]
                + step["new_connections_pred"]
            )
            step["total_kwh_pred"] = (
                step["total_connections_pred"] * step["kwh_per_conn_pred"]
            )
            step["scenario"] = scenario
            step["horizon_month"] = h

            keep = [
                "date", "region", "scenario", "horizon_month",
                "new_connections_pred", "kwh_per_conn_pred",
                "total_connections_pred", "total_kwh_pred",
            ]
            step = step[keep].copy()
            all_results.append(step)
            history = pd.concat([history, step], ignore_index=True)

    return pd.concat(all_results, ignore_index=True)


def main():
    df = pd.read_csv("data/raw/synthetic.csv", parse_dates=["date"])
    print(f"History rows: {len(df)}")
    print(f"Last date in history: {df['date'].max().date()}")

    out = forecast(df)
    out.to_csv("data/processed/forecast_scenarios.csv", index=False)
    print(f"\nWrote {len(out)} forecast rows to data/processed/forecast_scenarios.csv")

    summary = out.groupby(["scenario", "region"]).agg(
        new_connections_total=("new_connections_pred", "sum"),
        total_kwh_end=("total_kwh_pred", "last"),
    ).round(1)
    print("\n=== Summary (12-month totals) ===")
    print(summary)


if __name__ == "__main__":
    main()