"""
anova_sweep.py — Full factorial ANOVA sweep for Blood Logistics Sim (tiered)
=============================================================================
Variables swept (4 levels each = 256 runs):
  - casualty_rate:    10, 50, 80, 110  (cas/day/node)
  - resupply_hrs:     24, 72, 150, 200 (hours)
  - t1_pct:           3, 5, 7, 10      (% of casualties)
  - wbb_rate:         4, 8, 10, 12     (units/hr)

Outcomes captured:
  - fill_rate
  - cumulative_deaths
  - unmet_total
  - surgery_wait_deaths
  - t2_pcc_deaths
  - first_failure_time (hrs)
  - surgeries_completed
  - evac_count
  - FWB_from_DD + FWB_from_WIA (adjunct augmentation)

Outputs:
  - anova_results.csv   — raw results, one row per run
  - anova_summary.csv   — main effects table (mean outcome by factor level)
  - anova_stats.txt     — ANOVA F-statistics and p-values (scipy)

Usage:
  python anova_sweep.py
  python anova_sweep.py --workers 4   # limit CPU cores
"""

import numpy as np
import pandas as pd
import itertools
import argparse
import time
from multiprocessing import Pool, cpu_count
from simulation_tiered import Simulation, run_single_simulation_oo, haversine

# ── Sweep parameters ──────────────────────────────────────────────────────────
CASUALTY_RATES  = [10, 50, 80, 110]   # cas/day/node
RESUPPLY_HRS    = [24, 72, 150, 200]  # hours between resupply flights
T1_PCTS         = [3, 5, 7, 10]       # % of casualties who are T1
WBB_RATES       = [4, 8, 10, 12]      # WBB generation rate (units/hr)

# ── Fixed parameters (doctrinal LSCO baseline) ───────────────────────────────
N_NODES     = 5
T_MAX       = 24 * 30   # 720 hrs = 30 days
DT          = 0.1
BLACKOUTS   = 2         # standard LSCO: 2 blackout windows
T1_UNITS    = 24        # registry median
RESUPPLY_DELAY = 3.0

BLACKOUT_WINDOWS = [
    (24 * 5,  24 * 8),   # Day 5-8
    (24 * 18, 24 * 21),  # Day 18-21
]

lat_lon = np.array([
    [49.9935, 36.2304],
    [48.5862, 38.0000],
    [47.8388, 35.1396],
    [48.4647, 35.0462],
    [46.6354, 32.6169],
])

def build_travel_matrix(n):
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = haversine(
                    lat_lon[i][0], lat_lon[i][1],
                    lat_lon[j][0], lat_lon[j][1])
                mat[i, j] = dist / 278 + 0.167
    return mat

travel_time_matrix = build_travel_matrix(N_NODES)


def make_config(casualty_rate, resupply_hrs, t1_pct, wbb_rate):
    N = N_NODES
    time_arr = np.arange(0, T_MAX, DT)

    # Blackout windows only if BLACKOUTS > 0
    bk_windows = BLACKOUT_WINDOWS[:BLACKOUTS]

    config = {
        "N":        N,
        "T_max":    T_MAX,
        "dt":       DT,
        "time":     time_arr,
        "seed":     42,

        # Blood stock
        "INIT_STOCK": {
            "RBC": 60, "FFP": 40, "PLT": 20,
            "CRYO": 20, "WBB": 60, "CAS": 0
        },
        "MAX_STORAGE":  500,
        "B_max":        120,

        # Casualty / tier params
        "casualty_rate": casualty_rate / 24.0,  # convert to per-hour
        "CASUALTY_TIERS": {
            "T1": {"fraction": t1_pct / 100.0, "units_needed": T1_UNITS, "surgical": True},
            "T2": {"fraction": 0.12,            "units_needed": 6.0,      "surgical": False},
            "T3": {"fraction": max(0.0, 1.0 - t1_pct / 100.0 - 0.12), "units_needed": 0.0, "surgical": False},
        },

        # Resupply
        "interval_hours":  resupply_hrs,
        "resupply_delay":  RESUPPLY_DELAY,
        "blackout_windows": bk_windows,

        # WBB / donors
        "wbb_rate":         wbb_rate,
        "N_total":          100,
        "PCT_KIA":          0.25,
        "pct_kia_suitable": 0.10,
        "pct_wia_stable":   0.15,
        "UNITS_PER_WIA":    1.0,
        "UNITS_PER_DD":     3.0,
        "FWB_PROCESSING_DELAY": 1.0,

        # Expiry
        "EXPIRY_WBB": 24,
        "EXPIRY_PLT": 120,
        "tau":        56 * 24,
        "tau_RBC":    42 * 24,
        "tau_FFP":    12 * 24,
        "tau_CRYO":    6 * 24,

        # Thresholds
        "CRITICAL_THRESHOLD": {"RBC": 10, "FFP": 5,  "PLT": 1, "WBB": 5,  "CRYO": 3},
        "SAFE_THRESHOLD":     {"RBC": 20, "FFP": 10, "PLT": 2, "WBB": 10, "CRYO": 6},
        "PLT_PER_PACKET":     1,

        # Surgical subsystem
        "OR_DURATION":        2.25,
        "MEAN_TIME_TO_DEATH": 4.0,
        "PACU_EXIT_TIMES":    6.0,
        "PCC_SURVIVAL_HOURS": 12.0,
        "POSTOP_MAX":         8,
        "PCC_MAX":            8,
        "DCS_KITS":           30,
        "KITS_PER_RESUPPLY":  10,
        "KIT_REPROCESS_TIME": 2.0,
        "PARTIAL_TX_MORTALITY": 0.50,
        "T2_PCC_SURVIVAL_HOURS": 72.0,
        "CASEVAC_CAPACITY":   4,

        # Surgeon model
        "SURGEONS_PER_NODE":  2,
        "MAX_SURGEON_HOURS":  12.0,

        # Redistribution
        "redistribution_check_interval": 6.0,

        # Travel
        "travel_time_matrix": travel_time_matrix,
        "t_setup": np.zeros(N),
    }
    return config


def run_one(params):
    casualty_rate, resupply_hrs, t1_pct, wbb_rate = params
    config  = make_config(casualty_rate, resupply_hrs, t1_pct, wbb_rate)
    results = run_single_simulation_oo(config)
    return {
        # Factor levels
        "casualty_rate": casualty_rate,
        "resupply_hrs":  resupply_hrs,
        "t1_pct":        t1_pct,
        "wbb_rate":      wbb_rate,
        # Outcomes
        "fill_rate":             results.get("fill_rate", np.nan),
        "cumulative_deaths":     results.get("cumulative_deaths", np.nan),
        "unmet_total":           results.get("unmet_total", np.nan),
        "surgery_wait_deaths":   results.get("surgery_wait_deaths", np.nan),
        "t2_pcc_deaths":         results.get("t2_pcc_deaths", np.nan),
        "first_failure_time":    results.get("first_failure_time", np.nan),
        "surgeries_completed":   results.get("surgeries_completed", np.nan),
        "evac_count":            results.get("evac_count", np.nan),
        "adjunct_units":         results.get("FWB_from_DD", 0) + results.get("FWB_from_WIA", 0),
        "FWB_from_WBB":          results.get("FWB_from_WBB", np.nan),
        "surgeon_shortage_hours": results.get("surgeon_shortage_hours", np.nan),
    }


def run_anova_stats(df, outcomes, factors):
    """Compute one-way ANOVA F and p for each factor x outcome combination."""
    from scipy import stats
    rows = []
    for outcome in outcomes:
        for factor in factors:
            groups = [df[df[factor] == lvl][outcome].dropna().values
                      for lvl in sorted(df[factor].unique())]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                continue
            F, p = stats.f_oneway(*groups)
            # Eta-squared (effect size) = SS_between / SS_total
            grand_mean = df[outcome].mean()
            ss_total   = ((df[outcome] - grand_mean) ** 2).sum()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            eta2       = ss_between / ss_total if ss_total > 0 else np.nan
            rows.append({
                "outcome": outcome,
                "factor":  factor,
                "F":       round(F, 3),
                "p":       round(p, 6),
                "eta2":    round(eta2, 4),
                "significant": "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            })
    return pd.DataFrame(rows).sort_values(["outcome", "eta2"], ascending=[True, False])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help="Number of parallel workers (default: all CPUs)")
    args = parser.parse_args()

    # Build full factorial parameter list
    param_grid = list(itertools.product(
        CASUALTY_RATES, RESUPPLY_HRS, T1_PCTS, WBB_RATES
    ))
    print(f"Full factorial sweep: {len(param_grid)} runs")
    print(f"Variables: casualty_rate × resupply_hrs × t1_pct × wbb_rate")
    print(f"Workers: {args.workers}")
    print("-" * 50)

    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(run_one, param_grid)
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv("anova_results.csv", index=False)
    print(f"\nDone in {elapsed:.1f}s — saved anova_results.csv ({len(df)} rows)")

    # ── Main effects summary ──────────────────────────────────────────────────
    outcomes = [
        "fill_rate", "cumulative_deaths", "unmet_total",
        "surgery_wait_deaths", "t2_pcc_deaths",
        "surgeries_completed", "evac_count",
    ]
    factors = ["casualty_rate", "resupply_hrs", "t1_pct", "wbb_rate"]

    summary_rows = []
    for factor in factors:
        for lvl in sorted(df[factor].unique()):
            sub = df[df[factor] == lvl]
            row = {"factor": factor, "level": lvl}
            for o in outcomes:
                row[f"{o}_mean"] = round(sub[o].mean(), 2)
                row[f"{o}_std"]  = round(sub[o].std(),  2)
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("anova_summary.csv", index=False)
    print("Saved anova_summary.csv")

    # ── ANOVA stats ───────────────────────────────────────────────────────────
    try:
        stats_df = run_anova_stats(df, outcomes, factors)
        stats_df.to_csv("anova_stats.csv", index=False)

        # Pretty print to terminal
        print("\n── ANOVA Results (sorted by effect size η²) ──────────────────")
        print(f"{'Outcome':<25} {'Factor':<20} {'F':>8} {'p':>10} {'η²':>8} {'Sig':>5}")
        print("-" * 80)
        for _, r in stats_df.iterrows():
            print(f"{r['outcome']:<25} {r['factor']:<20} {r['F']:>8.1f} {r['p']:>10.4f} {r['eta2']:>8.4f} {r['significant']:>5}")

        print("\nSaved anova_stats.csv")
    except ImportError:
        print("scipy not installed — skipping ANOVA stats. Run: pip install scipy")

    # ── Quick summary to terminal ─────────────────────────────────────────────
    print("\n── Mean outcomes by casualty rate ────────────────────────────────")
    print(df.groupby("casualty_rate")[["fill_rate","cumulative_deaths","unmet_total"]].mean().round(1).to_string())
    print("\n── Mean outcomes by resupply interval ────────────────────────────")
    print(df.groupby("resupply_hrs")[["fill_rate","cumulative_deaths","unmet_total"]].mean().round(1).to_string())
