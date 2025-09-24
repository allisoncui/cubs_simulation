# analyze_panel.py
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-dir",
    type=str,
    default=None,
    help="Path to a specific run directory under cubs_simulation/data. "
         "If omitted, uses cubs_simulation/data/latest."
)
args = parser.parse_args()

base_data_dir = Path("cubs_simulation/data")
run_dir = Path(args.run_dir) if args.run_dir else (base_data_dir / "latest")
run_dir = run_dir.resolve()

# Validate run folder & panel.parquet BEFORE creating any subfolders
panel_path = run_dir / "panel.parquet"
if not run_dir.exists() or not panel_path.exists():
    # Do nothing if invalid; exit cleanly
    print(f"(info) No valid run folder found at {run_dir} (expected {panel_path}). Skipping analysis.")
    sys.exit(0)

# Create subfolders only for a valid run
analytics_dir = run_dir / "analytics"
figs_dir = run_dir / "figs"
analytics_dir.mkdir(parents=True, exist_ok=True)
figs_dir.mkdir(parents=True, exist_ok=True)

# load data
df = pd.read_parquet(panel_path)
if not np.issubdtype(df["dt"].dtype, np.datetime64):
    df["dt"] = pd.to_datetime(df["dt"])

# derive a cycle key (one row/day, summarize by person-cycle)
if "billing_cycle" not in df.columns:
    df["billing_cycle"] = df["dt"].dt.to_period("M").astype(str)

# revolver indicator by day -> per-cycle status
cycle = (
    df.groupby(["id", "billing_cycle"], as_index=False)
      .agg(
          revolver_cycle=("is_revolving", "max"),
          mean_consumption=("consumption", "mean"),
          mean_payment=("payment_to_card", "mean"),
          mean_interest=("interest_charge", "mean"),
          mean_late_fee=("late_fee", "mean"),
      )
      .sort_values(["id", "billing_cycle"])
)

# summary
n_rows = len(df)
n_people = df["id"].nunique()
n_cycles = len(cycle)
frac_revolver_cycles = cycle["revolver_cycle"].mean()
daily_means = df[["consumption", "payment_to_card", "interest_charge", "late_fee"]].mean()

print("BASIC SUMMARY")
print(f"  n_rows: {n_rows}")
print(f"  n_people: {n_people}")
print(f"  n_cycles: {n_cycles}")
print(f"  frac_revolver_cycles: {frac_revolver_cycles}")
print(f"  daily_mean_consumption: {daily_means['consumption']}")
print(f"  daily_mean_payment_to_card: {daily_means['payment_to_card']}")
print(f"  daily_mean_interest_charge: {daily_means['interest_charge']}")
print(f"  daily_mean_late_fee: {daily_means['late_fee']}")

# transition probs P(R_{t+h}=1 | R_t)
cycle["R"] = cycle["revolver_cycle"].astype(int)

def horizon_probs(cyc, max_h=12):
    out = []
    for r_now in (0, 1):
        row = []
        for h in range(1, max_h+1):
            mask = cyc["R"] == r_now
            future = cyc.groupby("id")["R"].shift(-h)
            row.append(float(np.nanmean(future[mask])))
        out.append(row)
    return np.array(out)

R_h = horizon_probs(cycle, max_h=12)
trans_table = pd.DataFrame(
    R_h,
    index=pd.Index([0, 1], name="current_R"),
    columns=[f"h{i}" for i in range(1, 13)]
)
print("\nP(R_{t+h}=1 | R_t) (rows current 0/1, cols h1..h12)")
print(trans_table)

# persistence tables by past-12 behavior
def add_past12_count(cyc):
    cyc = cyc.copy()
    cyc["past12_revolver_count"] = (
        cyc.groupby("id")["R"]
           .rolling(12, min_periods=1)
           .sum()
           .shift(1)
           .reset_index(level=0, drop=True)
    ).fillna(0).astype(int)
    return cyc

cycle = add_past12_count(cycle)

def conditional_by_past12(cyc, condition_R, max_h=12):
    sub = cyc[cyc["R"] == condition_R].copy()
    results = []
    for k in range(0, 13):
        rows_k = sub[sub["past12_revolver_count"] == k]
        line = []
        for h in range(1, max_h+1):
            future = cyc.groupby("id")["R"].shift(-h)
            val = float(np.nanmean(future.loc[rows_k.index]))
            line.append(val)
        results.append(line if len(rows_k) else [np.nan]*max_h)
    return pd.DataFrame(results,
                        index=pd.Index(range(13), name="past12_revolver_count"),
                        columns=[f"h{i}" for i in range(1, 13)])

table_T = conditional_by_past12(cycle, condition_R=0, max_h=12)
table_R = conditional_by_past12(cycle, condition_R=1, max_h=12)

print("\nTable (condition on current = Transactor, rows past12 count 0..12):")
print(table_T)
print("\nTable (condition on current = Revolver, rows past12 count 0..12):")
print(table_R)

# save analytics to specific run
trans_table.to_parquet(analytics_dir / "transition_probs.parquet", index=True)
table_T.to_parquet(analytics_dir / "cond_table_current_T.parquet", index=True)
table_R.to_parquet(analytics_dir / "cond_table_current_R.parquet", index=True)

# heatmaps
def save_heatmap(df_heat, title, fname):
    plt.figure()
    arr = df_heat.to_numpy(dtype=float)
    plt.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")

    # Add text annotations for each cell
    n_rows, n_cols = arr.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = arr[i, j]
            if not np.isnan(val):  # skip NaN cells
                plt.text(
                    j, i, f"{val:.2f}",        # format with 2 decimal places
                    ha="center", va="center",
                    color="white" if val > 0.5 else "black",  # contrast text
                    fontsize=7
                )

    plt.title(title)
    plt.xlabel("horizon (1..12)")
    plt.ylabel(df_heat.index.name or "")
    plt.colorbar(label="Probability of Revolver")
    plt.tight_layout()
    plt.savefig(figs_dir / fname, dpi=150)
    plt.close()

save_heatmap(trans_table, "P(R_{t+h}=1 | R_t)", "heatmap_transition.png")
save_heatmap(table_T, "P(R_{t+h}=1) | current=Transactor, by past12 count", "heatmap_T.png")
save_heatmap(table_R, "P(R_{t+h}=1) | current=Revolver, by past12 count", "heatmap_R.png")

# metadata
meta = {
    "run_dir": str(run_dir),
    "panel_path": str(panel_path),
    "n_rows": int(n_rows),
    "n_people": int(n_people),
    "n_cycles": int(n_cycles),
    "frac_revolver_cycles": float(frac_revolver_cycles),
    "daily_mean_consumption": float(daily_means["consumption"]),
    "daily_mean_payment_to_card": float(daily_means["payment_to_card"]),
    "daily_mean_interest_charge": float(daily_means["interest_charge"]),
    "daily_mean_late_fee": float(daily_means["late_fee"]),
}
(run_dir / "analysis_meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nSaved analytics to {analytics_dir} and heatmaps to {figs_dir}")
