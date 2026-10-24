"""Generate synthetic KPLC-style data for Western Kenya."""
import numpy as np
import pandas as pd
from pathlib import Path

REGIONS = ["Kisumu", "Kakamega", "Vihiga", "Siaya", "Busia", "Bungoma"]
BASE = {"Kisumu": 42000, "Kakamega": 38000, "Vihiga": 15000,
        "Siaya": 18000, "Busia": 16000, "Bungoma": 22000}

def main():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for region in REGIONS:
        base = BASE[region]
        for i, d in enumerate(dates):
            trend = 1 + 0.012 * i
            noise = 1 + rng.normal(0, 0.02)
            total = int(base * trend * noise)
            new = int(total * 0.018 * (1 + rng.normal(0, 0.15)))
            kwh_pc = round(55 * (1 + rng.normal(0, 0.05)), 2)
            rows.append({
                "date": d, "region": region,
                "total_connections": total,
                "new_connections": max(new, 0),
                "kwh_per_connection": kwh_pc,
                "total_kwh": round(total * kwh_pc, 1),
            })
    df = pd.DataFrame(rows).sort_values(["region", "date"]).reset_index(drop=True)
    out = Path("data/raw/synthetic.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(df.head())

if __name__ == "__main__":
    main()