# server.py — Flask backend for LSCO Blood Logistics UI
# ============================================================
# Run with:  python server.py
# Listens on http://localhost:5000
#
# Endpoints:
#   POST /run    — accepts JSON config, runs one simulation, returns results
#   GET  /health — liveness check
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback

from simulation_tiered import run_single_simulation_oo, haversine

app = Flask(__name__)
CORS(app)   # allow requests from the UI (different port)

# ── Node network (fixed — 5-node Ukraine hotspot layout) ─────────
N = 5
LAT_LON = np.array([
    [49.9935, 36.2304],   # Kharkiv
    [48.5862, 38.0000],   # Luhansk region
    [47.8388, 35.1396],   # Zaporizhzhia
    [48.4647, 35.0462],   # Dnipro
    [46.6354, 32.6169],   # Kherson
])

def build_travel_time_matrix():
    ttm = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = haversine(LAT_LON[i][0], LAT_LON[i][1],
                                 LAT_LON[j][0], LAT_LON[j][1])
                ttm[i, j] = dist / 278 + 0.167
    return ttm

TRAVEL_TIME_MATRIX = build_travel_time_matrix()

# ── Base config (fixed params not exposed in the UI) ─────────────
def build_config(params):
    """
    Merges UI-supplied params with fixed base config.
    All numpy arrays are rebuilt here so the sim gets exactly what it needs.
    """
    t1_frac  = params["t1Pct"] / 100
    t2_frac  = 0.12
    t3_frac  = max(0.0, round(1.0 - t1_frac - t2_frac, 6))
    T_max    = 24 * 30
    dt       = 0.1

    # Blackout windows — each 72h of contested airspace
    # Cumulative: each level adds a window to the previous set
    # Day 5-8   (t=120-192h)  — early disruption
    # Day 18-21 (t=432-504h)  — mid-campaign
    # Day 10-13 (t=240-312h)  — early-mid overlap
    # Day 24-27 (t=576-648h)  — late, hits exhausted stocks hardest
    _all_windows = [(120, 192), (432, 504), (240, 312), (576, 648)]
    n_blackouts = int(params.get("blackouts", 0))
    blackout_windows = _all_windows[:n_blackouts]

    config = {
        # Horizon
        "N":     N,
        "T_max": T_max,
        "dt":    dt,
        "time":  list(np.arange(0, T_max, dt)),   # converted to list for JSON safety

        # Blood stock
        "INIT_STOCK": {
            "RBC":  params["initRBC"],
            "FFP":  params["initFFP"],
            "PLT":  20,
            "CRYO": 20,
            "WBB":  params["initWBB"],
            "CAS":  0,
        },

        # MTP packet sizes
        "RBC_PER_PACKET":      4,
        "FFP_PER_PACKET":      4,
        "PLT_PER_PACKET":      1,
        "WB_UNITS_PER_PACKET": 4,
        "PACKETS_PER_PATIENT": 2,

        # Thresholds
        "CRITICAL_THRESHOLD": {"RBC": 10, "FFP": 5, "PLT": 1, "WBB": 5,  "CRYO": 3},
        "SAFE_THRESHOLD":     {"RBC": 20, "FFP": 10, "PLT": 2, "WBB": 10, "CRYO": 6},
        "MAX_STORAGE": 200,
        "B_max":       120,

        # Donor pool
        "N_total": 100,
        "tau":     56 * 24,

        # Casualty / FWB
        "PCT_KIA":       0.25,
        "UNITS_PER_DD":  5.0,
        "UNITS_PER_WIA": 1.0,

        # Shelf lives
        "EXPIRY_WBB": 24,
        "EXPIRY_PLT": 120,
        "tau_RBC":    42 * 24,
        "tau_FFP":    12 * 24,
        "tau_CRYO":    6 * 24,

        # Redistribution
        "redistribution_check_interval": 6.0,
        "travel_time_matrix": TRAVEL_TIME_MATRIX,

        # Casualty tiers
        "CASUALTY_TIERS": {
            "T1": {"fraction": t1_frac,  "units_needed": params["t1Units"], "surgical": True},
            "T2": {"fraction": t2_frac,  "units_needed": 6.0,               "surgical": False},
            "T3": {"fraction": t3_frac,  "units_needed": 0.0,               "surgical": False},
        },

        # Surgical
        "DCS_KITS":            params["dcsKits"],
        "KITS_PER_RESUPPLY":   10,
        "KIT_RECYCLABLE_RATE": 0.50,
        "KIT_REPROCESS_TIME":  2.0,
        "OR_DURATION":         2.25,
        "MEAN_TIME_TO_DEATH":  params["mttd"],
        "POSTOP_MAX":          8,
        "PACU_EXIT_TIMES":     params["pacuLos"],
        "PCC_SURVIVAL_HOURS":  12.0,
        "PCC_MORTALITY_RATE":  0.03,
        "SURGEONS_PER_NODE":   params.get("surgeonsPerNode", 2),
        "MAX_SURGEON_HOURS":   params.get("maxSurgHours", 12),

        # FWB processing
        "FWB_PROCESSING_DELAY": 1.0,

        # Logistics
        "interval_hours":   params["resupplyHrs"],
        "resupply_delay":   3.0,
        "blackout_windows": blackout_windows,
        "t_setup":          np.zeros(N),

        # Sweep params
        "casualty_rate":    params["casualtyRate"] / 24,
        "wbb_rate":         params["wbbRate"],
        "pct_kia_suitable": params["kiaEligible"] / 100,
        "pct_wia_stable":   params["wiaStable"] / 100,
    }

    # Convert numpy arrays to plain Python for JSON serialization later
    config["t_setup"] = config["t_setup"].tolist()
    config["travel_time_matrix"] = config["travel_time_matrix"].tolist()
    config["time"] = list(np.arange(0, T_max, dt))

    # Re-attach as numpy where the sim needs it
    config["t_setup"]              = np.array(config["t_setup"])
    config["travel_time_matrix"]   = np.array(config["travel_time_matrix"])
    config["time"]                 = np.arange(0, T_max, dt)

    return config


def sanitize(results):
    """Convert numpy scalars/arrays to plain Python for JSON."""
    out = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


# ── Routes ────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/run", methods=["POST"])
def run_simulation():
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "No JSON body received"}), 400

        config  = build_config(params)
        results = run_single_simulation_oo(config)
        return jsonify(sanitize(results))

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 55)
    print("  LSCO Blood Logistics Sim Server")
    print(f"  http://0.0.0.0:{port}")
    print("  POST /run  — run a simulation")
    print("  GET  /health — liveness check")
    print(f"  Auth: {'TOKEN SET' if ACCESS_TOKEN else 'DISABLED (local dev)'}")
    print("=" * 55)
    app.run(host="0.0.0.0", debug=False, port=port)
