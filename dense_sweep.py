"""
dense_sweep.py — Dense asynchronous parameter sweep for Blood Logistics Sim
=============================================================================
Parameter grids:
  - casualty_rate:  10 → 110  step 10   (11 levels)
  - resupply_hrs:   24 → 200  step 12   (15 levels: 24, 36, … 192)
  - t1_pct:         3, 5, 7, 10         ( 4 levels — unchanged)
  - wbb_rate:       2.0 → 10.0 step 0.5 (17 levels)

Total: 11 × 15 × 4 × 17 = 11,220 runs

Features:
  - True async dispatch via asyncio + ProcessPoolExecutor
  - Checkpoint file written every --checkpoint-every N completed runs (default 500)
  - --resume flag picks up where a previous run left off
  - Live progress bar with ETA (tqdm)
  - Final CSV + ANOVA stats identical in format to anova_sweep.py

Usage:
  python dense_sweep.py                          # all CPUs, fresh run
  python dense_sweep.py --workers 8              # limit to 8 cores
  python dense_sweep.py --resume                 # resume from checkpoint
  python dense_sweep.py --checkpoint-every 200   # save more frequently
  python dense_sweep.py --out results/dense       # custom output prefix
"""

import asyncio
import itertools
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

from simulation_tiered import run_single_simulation_oo, haversine

# ── Dense sweep parameters ────────────────────────────────────────────────────
CASUALTY_RATES = list(range(10, 111, 10))                    # 11 levels: 10…110
RESUPPLY_HRS   = list(range(24, 201, 12))                    # 15 levels: 24…192
T1_PCTS        = [3, 5, 7, 10]                              #  4 levels (unchanged)
WBB_RATES      = [round(v, 1) for v in np.arange(2.0, 10.5, 0.5)]  # 17 levels: 2.0…10.0

# ── Fixed / doctrinal parameters ──────────────────────────────────────────────
N_NODES        = 5
T_MAX          = 24 * 30   # 720 hrs = 30 days
DT             = 0.1
BLACKOUTS      = 2
T1_UNITS       = 24
RESUPPLY_DELAY = 3.0

BLACKOUT_WINDOWS = [
    (24 * 5,  24 * 8),
    (24 * 18, 24 * 21),
]

LAT_LON = np.array([
    [49.9935, 36.2304],
    [48.5862, 38.0000],
    [47.8388, 35.1396],
    [48.4647, 35.0462],
    [46.6354, 32.6169],
])


def _build_travel_matrix():
    n = len(LAT_LON)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = haversine(LAT_LON[i][0], LAT_LON[i][1],
                                 LAT_LON[j][0], LAT_LON[j][1])
                mat[i, j] = dist / 278 + 0.167
    return mat


# Build once at module level — pickling plain ndarray is cheap and avoids
# re-computing in every worker.
TRAVEL_MATRIX = _build_travel_matrix()


# ── Worker function (runs in subprocess) ──────────────────────────────────────

def _make_config(casualty_rate, resupply_hrs, t1_pct, wbb_rate):
    N = N_NODES
    time_arr = np.arange(0, T_MAX, DT)
    bk_windows = BLACKOUT_WINDOWS[:BLACKOUTS]

    return {
        "N":        N,
        "T_max":    T_MAX,
        "dt":       DT,
        "time":     time_arr,
        "seed":     42,

        "INIT_STOCK": {
            "RBC": 60, "FFP": 40, "PLT": 20,
            "CRYO": 20, "WBB": 60, "CAS": 0,
        },
        "MAX_STORAGE":  500,
        "B_max":        120,

        "casualty_rate": casualty_rate / 24.0,
        "CASUALTY_TIERS": {
            "T1": {"fraction": t1_pct / 100.0, "units_needed": T1_UNITS, "surgical": True},
            "T2": {"fraction": 0.12,            "units_needed": 6.0,      "surgical": False},
            "T3": {"fraction": max(0.0, 1.0 - t1_pct / 100.0 - 0.12),
                   "units_needed": 0.0, "surgical": False},
        },

        "interval_hours":  resupply_hrs,
        "resupply_delay":  RESUPPLY_DELAY,
        "blackout_windows": bk_windows,

        "wbb_rate":         wbb_rate,
        "N_total":          100,
        "PCT_KIA":          0.25,
        "pct_kia_suitable": 0.10,
        "pct_wia_stable":   0.15,
        "UNITS_PER_WIA":    1.0,
        "UNITS_PER_DD":     3.0,
        "FWB_PROCESSING_DELAY": 1.0,

        "EXPIRY_WBB": 24,
        "EXPIRY_PLT": 120,
        "tau":        56 * 24,
        "tau_RBC":    42 * 24,
        "tau_FFP":    12 * 24,
        "tau_CRYO":    6 * 24,

        "CRITICAL_THRESHOLD": {"RBC": 10, "FFP": 5,  "PLT": 1, "WBB": 5,  "CRYO": 3},
        "SAFE_THRESHOLD":     {"RBC": 20, "FFP": 10, "PLT": 2, "WBB": 10, "CRYO": 6},
        "PLT_PER_PACKET":     1,

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

        "SURGEONS_PER_NODE":  2,
        "MAX_SURGEON_HOURS":  12.0,

        "redistribution_check_interval": 6.0,

        "travel_time_matrix": TRAVEL_MATRIX,
        "t_setup": np.zeros(N),
    }


def run_one(params):
    """Top-level picklable worker function — executed in a subprocess."""
    casualty_rate, resupply_hrs, t1_pct, wbb_rate = params
    config  = _make_config(casualty_rate, resupply_hrs, t1_pct, wbb_rate)
    results = run_single_simulation_oo(config)
    return {
        "casualty_rate":          casualty_rate,
        "resupply_hrs":           resupply_hrs,
        "t1_pct":                 t1_pct,
        "wbb_rate":               wbb_rate,
        "fill_rate":              results.get("fill_rate",             np.nan),
        "cumulative_deaths":      results.get("cumulative_deaths",     np.nan),
        "unmet_total":            results.get("unmet_total",           np.nan),
        "surgery_wait_deaths":    results.get("surgery_wait_deaths",   np.nan),
        "t2_pcc_deaths":          results.get("t2_pcc_deaths",         np.nan),
        "first_failure_time":     results.get("first_failure_time",    np.nan),
        "surgeries_completed":    results.get("surgeries_completed",   np.nan),
        "evac_count":             results.get("evac_count",            np.nan),
        "adjunct_units":          results.get("FWB_from_DD", 0) + results.get("FWB_from_WIA", 0),
        "FWB_from_WBB":           results.get("FWB_from_WBB",          np.nan),
        "surgeon_shortage_hours": results.get("surgeon_shortage_hours", np.nan),
    }


# ── ANOVA stats (identical to anova_sweep.py) ─────────────────────────────────

def run_anova_stats(df, outcomes, factors):
    from scipy import stats
    rows = []
    for outcome in outcomes:
        for factor in factors:
            groups = [
                df[df[factor] == lvl][outcome].dropna().values
                for lvl in sorted(df[factor].unique())
            ]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                continue
            F, p = stats.f_oneway(*groups)
            grand_mean = df[outcome].mean()
            ss_total   = ((df[outcome] - grand_mean) ** 2).sum()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            eta2       = ss_between / ss_total if ss_total > 0 else np.nan
            rows.append({
                "outcome":     outcome,
                "factor":      factor,
                "F":           round(F, 3),
                "p":           round(p, 6),
                "eta2":        round(eta2, 4),
                "significant": "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")),
            })
    return pd.DataFrame(rows).sort_values(["outcome", "eta2"], ascending=[True, False])


# ── Async sweep engine ─────────────────────────────────────────────────────────

async def run_sweep(param_grid, completed_params, workers, checkpoint_every,
                    checkpoint_path, out_prefix):
    """Dispatch all runs asynchronously; checkpoint periodically."""
    try:
        from tqdm.asyncio import tqdm as atqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("tqdm not installed — install with: pip install tqdm  (progress bar disabled)")

    completed_set = {tuple(p) for p in completed_params}
    pending = [p for p in param_grid if tuple(p) not in completed_set]

    total     = len(param_grid)
    done_so_far = total - len(pending)

    print(f"Total runs:     {total}")
    print(f"Already done:   {done_so_far}")
    print(f"Remaining:      {len(pending)}")
    print(f"Workers:        {workers}")
    print(f"Checkpoint every {checkpoint_every} runs → {checkpoint_path}")
    print("-" * 60)

    loop      = asyncio.get_event_loop()
    rows      = []        # new rows accumulated since last checkpoint
    all_rows  = []        # every row (for final write)
    n_done    = 0

    # Load existing results if resuming
    existing_csv = f"{out_prefix}_results.csv"
    if os.path.exists(existing_csv) and done_so_far > 0:
        existing_df = pd.read_csv(existing_csv)
        all_rows = existing_df.to_dict("records")
        print(f"Loaded {len(all_rows)} existing rows from {existing_csv}")

    t_start = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            loop.run_in_executor(executor, run_one, p): p
            for p in pending
        }

        if use_tqdm:
            pbar = atqdm(total=total, initial=done_so_far,
                         desc="Sweep", unit="run", dynamic_ncols=True)

        pending_futures = set(futures.keys())

        while pending_futures:
            done_batch, pending_futures = await asyncio.wait(
                pending_futures, return_when=asyncio.FIRST_COMPLETED
            )

            for fut in done_batch:
                try:
                    row = fut.result()
                except Exception as exc:
                    params = futures[fut]
                    print(f"\n[ERROR] params={params}: {exc}")
                    row = {
                        "casualty_rate": params[0], "resupply_hrs": params[1],
                        "t1_pct": params[2], "wbb_rate": params[3],
                        "error": str(exc),
                    }

                rows.append(row)
                all_rows.append(row)
                n_done += 1

                if use_tqdm:
                    pbar.update(1)
                    elapsed = time.time() - t_start
                    rate    = n_done / elapsed if elapsed > 0 else 0
                    eta_s   = (len(pending) - n_done) / rate if rate > 0 else float("inf")
                    pbar.set_postfix(
                        rate=f"{rate:.1f}/s",
                        ETA=f"{eta_s/3600:.1f}h" if eta_s < float("inf") else "?",
                    )

                # Checkpoint
                if n_done % checkpoint_every == 0:
                    _write_checkpoint(checkpoint_path, all_rows, [futures[f] for f in pending_futures])
                    # Write partial CSV so results are always accessible
                    pd.DataFrame(all_rows).to_csv(existing_csv, index=False)
                    elapsed = time.time() - t_start
                    total_done = done_so_far + n_done
                    print(f"\n[checkpoint] {total_done}/{total} done — "
                          f"{elapsed/3600:.2f}h elapsed — saved {existing_csv}")

        if use_tqdm:
            pbar.close()

    return all_rows


def _write_checkpoint(path, completed_rows, remaining_params):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            "completed_rows":    completed_rows,
            "remaining_params":  remaining_params,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)   # atomic rename


def _load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dense async parameter sweep for blood logistics sim"
    )
    parser.add_argument("--workers",          type=int,  default=cpu_count(),
                        help="Parallel worker processes (default: all CPUs)")
    parser.add_argument("--checkpoint-every", type=int,  default=500,
                        help="Save checkpoint every N completed runs (default: 500)")
    parser.add_argument("--checkpoint",       type=str,  default="dense_sweep_checkpoint.pkl",
                        help="Checkpoint file path (default: dense_sweep_checkpoint.pkl)")
    parser.add_argument("--resume",           action="store_true",
                        help="Resume from an existing checkpoint file")
    parser.add_argument("--out",              type=str,  default="dense_sweep",
                        help="Output file prefix (default: dense_sweep)")
    args = parser.parse_args()

    # Print grid summary
    print("=" * 60)
    print("Dense sweep parameter grid")
    print("=" * 60)
    print(f"  casualty_rate : {len(CASUALTY_RATES)} levels  {CASUALTY_RATES}")
    print(f"  resupply_hrs  : {len(RESUPPLY_HRS)} levels  {RESUPPLY_HRS}")
    print(f"  t1_pct        : {len(T1_PCTS)} levels  {T1_PCTS}")
    print(f"  wbb_rate      : {len(WBB_RATES)} levels  {WBB_RATES}")
    total = len(CASUALTY_RATES) * len(RESUPPLY_HRS) * len(T1_PCTS) * len(WBB_RATES)
    print(f"  TOTAL RUNS    : {total}")
    print("=" * 60)

    # Full factorial grid
    param_grid = list(itertools.product(CASUALTY_RATES, RESUPPLY_HRS, T1_PCTS, WBB_RATES))

    # Resume?
    completed_params = []
    if args.resume:
        if not os.path.exists(args.checkpoint):
            print(f"[resume] Checkpoint file not found: {args.checkpoint}")
            print("         Starting a fresh run.")
        else:
            ckpt = _load_checkpoint(args.checkpoint)
            completed_rows  = ckpt["completed_rows"]
            completed_params = [
                (r["casualty_rate"], r["resupply_hrs"], r["t1_pct"], r["wbb_rate"])
                for r in completed_rows
                if "error" not in r
            ]
            print(f"[resume] Loaded checkpoint: {len(completed_params)} completed runs")

    t_wall_start = time.time()

    all_rows = asyncio.run(run_sweep(
        param_grid       = param_grid,
        completed_params = completed_params,
        workers          = args.workers,
        checkpoint_every = args.checkpoint_every,
        checkpoint_path  = args.checkpoint,
        out_prefix       = args.out,
    ))

    elapsed = time.time() - t_wall_start
    print(f"\nAll runs complete in {elapsed/3600:.2f}h ({elapsed:.0f}s)")

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    results_csv = f"{args.out}_results.csv"
    df.to_csv(results_csv, index=False)
    print(f"Saved {results_csv}  ({len(df)} rows)")

    # ── Main effects summary ──────────────────────────────────────────────────
    outcomes = [
        "fill_rate", "cumulative_deaths", "unmet_total",
        "surgery_wait_deaths", "t2_pcc_deaths",
        "surgeries_completed", "evac_count",
    ]
    factors = ["casualty_rate", "resupply_hrs", "t1_pct", "wbb_rate"]

    # Only include rows with no error for stats
    df_clean = df[~df.get("error", pd.Series(dtype=str)).notna()].copy() \
               if "error" in df.columns else df.copy()

    summary_rows = []
    for factor in factors:
        for lvl in sorted(df_clean[factor].unique()):
            sub = df_clean[df_clean[factor] == lvl]
            row = {"factor": factor, "level": lvl}
            for o in outcomes:
                row[f"{o}_mean"] = round(sub[o].mean(), 2)
                row[f"{o}_std"]  = round(sub[o].std(),  2)
            summary_rows.append(row)

    summary_csv = f"{args.out}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv}")

    # ── ANOVA stats ───────────────────────────────────────────────────────────
    try:
        stats_df  = run_anova_stats(df_clean, outcomes, factors)
        stats_csv = f"{args.out}_stats.csv"
        stats_df.to_csv(stats_csv, index=False)

        print("\n── ANOVA Results (sorted by effect size η²) ──────────────────")
        print(f"{'Outcome':<25} {'Factor':<20} {'F':>8} {'p':>10} {'η²':>8} {'Sig':>5}")
        print("-" * 80)
        for _, r in stats_df.iterrows():
            print(f"{r['outcome']:<25} {r['factor']:<20} {r['F']:>8.1f} "
                  f"{r['p']:>10.4f} {r['eta2']:>8.4f} {r['significant']:>5}")

        print(f"\nSaved {stats_csv}")
    except ImportError:
        print("scipy not installed — skipping ANOVA stats. Run: pip install scipy")

    # ── Quick terminal summary ────────────────────────────────────────────────
    print("\n── Mean outcomes by casualty rate ─────────────────────────────────")
    print(df_clean.groupby("casualty_rate")[
        ["fill_rate", "cumulative_deaths", "unmet_total"]].mean().round(1).to_string())

    print("\n── Mean outcomes by resupply interval ─────────────────────────────")
    print(df_clean.groupby("resupply_hrs")[
        ["fill_rate", "cumulative_deaths", "unmet_total"]].mean().round(1).to_string())

    print("\n── Mean outcomes by wbb_rate ───────────────────────────────────────")
    print(df_clean.groupby("wbb_rate")[
        ["fill_rate", "cumulative_deaths", "unmet_total"]].mean().round(1).to_string())

    # Clean up checkpoint on successful completion
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print(f"\nCheckpoint {args.checkpoint} removed (run complete).")


if __name__ == "__main__":
    main()
