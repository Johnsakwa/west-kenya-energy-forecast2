"""Plot the 12-month scenario forecast."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_scenarios(csv="data/processed/forecast_scenarios.csv",
                   out="docs/forecast_scenarios.png"):
    df = pd.read_csv(csv, parse_dates=["date"])
    agg = df.groupby(["date", "scenario"])["total_kwh_pred"].sum().unstack()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "base": "#1f77b4",
        "accelerated": "#2ca02c",
        "constrained": "#d62728",
    }
    for s in agg.columns:
        ax.plot(agg.index, agg[s], label=s.title(),
                color=colors.get(s), marker="o", linewidth=2)

    ax.set_title("West Kenya - Total Monthly Demand Forecast (12-mo)", fontsize=13)
    ax.set_ylabel("kWh / month")
    ax.set_xlabel("Month")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    import os
    os.makedirs("docs", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_scenarios()