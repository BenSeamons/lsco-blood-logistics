import numpy as np

# =============================================================================
# simulation.py — Tiered Casualty Model (v3)
# =============================================================================
#
# KEY CHANGES FROM v2 (deterministic baseline):
#
#  [A] Casualty tiers replace flat CASUALTY_TO_MTP_RATIO
#      T1 (5%):  massive hemorrhage, DCS candidate, 24 units over OR duration
#      T2 (12%): moderate hemorrhage, component therapy only, 6 units at arrival
#      T3 (83%): no blood needed
#      Source: McWhirter et al. DODTR 2024 (Role 2, 15,581 encounters)
#              T1 = massive+supermassive (5%), T2 = submassive (11% → 12% rounded),
#              T3 = non-recipients (83%)
#
#  [B] T1 blood demand is OR-coupled
#      Blood is drawn continuously during the operative window at a fixed
#      drain rate = T1_UNITS_TOTAL / or_duration (units/hr), mirroring
#      DCR practice where resuscitation runs concurrently with DCS.
#      Blood is NOT consumed at casualty arrival — only when surgery begins.
#      This couples surgical throughput and blood supply as they are in reality.
#
#  [C] T2 blood demand uses a fractional accumulator
#      T2 patients draw blood at arrival using component therapy (no surgery).
#      Demand is accumulated fractionally so no patients are silently lost
#      at small dt.
#
#  [D] FWB collection processing delay
#      Collected FWB enters wbb_queues at age = FWB_PROCESSING_DELAY (hrs)
#      rather than age 0, preventing same-step collection-and-use.
#
#  [E] Kit resupply is incremental, not a full reset
#      Each resupply flight delivers KITS_PER_RESUPPLY kits (capped at DCS_KITS max).
#      kits_in_reprocessing is NOT cleared on resupply.
#
#  [F] PCC mortality uses time-in-PCC threshold
#      Patients in PCC die if time_in_pcc >= PCC_SURVIVAL_HOURS.
#      More clinically interpretable than the probability accumulator.
#
#  [G] Component product expiration (first-order)
#      RBC, FFP, CRYO now expire via E = X/tau each step, matching
#      the Modeling.md ODE formulation.
#
#  [H] Redistribution respects blackout windows
#      Transfers are not scheduled during blackouts and are rescheduled
#      (not dropped) if a blackout is active at planned arrival time.
#
# Unchanged from v2:
#  - All stochastic calls remain deterministic
#  - Resupply schedule, blackout windows, flight delay all deterministic
#  - FIFO WBB and PLT age queues preserved
#  - Donor recovery (56-day tau) preserved
#  - Full/partial/unmet MTP logging preserved
# =============================================================================


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def generate_casualties(N, t, rate_per_hour, dt, config):
    """
    Deterministic tiered casualty generation.

    Returns a dict with total CAS, T1, T2, T3 counts per node.
    T1 = surgical / massive hemorrhage  (5%  of total, McWhirter DODTR)
    T2 = moderate hemorrhage             (12% of total)
    T3 = no blood needed                 (83% of total)

    Surge period days 5-10 doubles baseline rate.
    """
    tiers = config.get("CASUALTY_TIERS", {
        "T1": {"fraction": 0.05},
        "T2": {"fraction": 0.12},
        "T3": {"fraction": 0.83},
    })
    baseline_rate = rate_per_hour
    surge_rate    = rate_per_hour * 2
    rate = surge_rate if (24 * 5 < t < 24 * 10) else baseline_rate
    total = rate * dt

    # casualty_rate is per-node — 50 cas/day means 50 arriving at EACH Role II
    # Total theater casualty burden = rate * N_nodes
    return {
        "CAS": np.full(N, total),
        "T1":  np.full(N, total * tiers["T1"]["fraction"]),
        "T2":  np.full(N, total * tiers["T2"]["fraction"]),
        "T3":  np.full(N, total * tiers["T3"]["fraction"]),
    }


def withdraw_from_queue(queue, needed):
    """FIFO withdrawal from an age-tagged queue."""
    withdrawn = 0.0
    new_queue = []
    for age, qty in queue:
        if withdrawn >= needed:
            new_queue.append([age, qty])
            continue
        take = min(qty, needed - withdrawn)
        withdrawn += take
        if qty > take:
            new_queue.append([age, qty - take])
    return new_queue, withdrawn


def due_this_step(now, prev, event_time):
    return (prev < event_time) and (event_time <= now)


def in_blackout(t, blackout_windows):
    return any(start <= t <= end for start, end in blackout_windows)


def get_flight_delay(i, j, travel_time_matrix):
    """Fixed 15% overhead on base travel time (deterministic)."""
    base_time = travel_time_matrix[i, j]
    if not np.isfinite(base_time):
        return np.inf
    return base_time * 1.15


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------
# Simulation class
# ------------------------------------------------------------------

class Simulation:
    def __init__(self, config):
        if "seed" in config:
            np.random.seed(config["seed"])

        self.config   = config
        self.dt       = config["dt"]
        self.interval_hours   = config["interval_hours"]
        self.casualty_rate    = config["casualty_rate"]
        self.wbb_rate         = config["wbb_rate"]
        self.pct_kia_suitable = config["pct_kia_suitable"]
        self.pct_wia_stable   = config["pct_wia_stable"]

        # Tier parameters (with registry-grounded defaults)
        tiers = config.get("CASUALTY_TIERS", {
            "T1": {"fraction": 0.05, "units_needed": 24.0, "surgical": True},
            "T2": {"fraction": 0.12, "units_needed": 6.0,  "surgical": False},
            "T3": {"fraction": 0.83, "units_needed": 0.0,  "surgical": False},
        })
        self.t1_units = tiers["T1"]["units_needed"]  # consumed over OR duration
        self.t2_units = tiers["T2"]["units_needed"]  # consumed at arrival
        self.or_duration = config.get("OR_DURATION", 2.25)  # hrs, fixed DCS time

        # FWB processing delay — collected blood not available until this age
        self.fwb_processing_delay = config.get("FWB_PROCESSING_DELAY", 1.0)  # hrs

        # Component shelf-life constants (first-order expiration, hrs)
        self.tau_RBC  = config.get("tau_RBC",  42 * 24)   # 42 days
        self.tau_FFP  = config.get("tau_FFP",  12 * 24)   # 12 days (thawed)
        self.tau_CRYO = config.get("tau_CRYO",  6 * 24)   # 6 days (thawed)

        # ── DCS / Surgical subsystem — PER NODE ──────────────────
        # Each node is an independent FRSD with its own ORs, PACU, PCC,
        # kit supply, and surgery queue. Blood draws only from that node.
        N_nodes = config["N"]

        self.postop_exit_duration = config.get("PACU_EXIT_TIMES", 6.0)
        self.pcc_max              = config.get("PCC_MAX", 8.0)
        self.postop_max           = config.get("POSTOP_MAX", 8.0)
        self.mean_time_to_death   = config.get("MEAN_TIME_TO_DEATH", 4.0)
        self.pcc_survival_hours   = config.get("PCC_SURVIVAL_HOURS", 12.0)

        # 2 ORs per node — each FRSD operates independently
        self.ORs = [
            [
                {"busy_until": 0.0, "waiting": False, "patient": None},
                {"busy_until": 0.0, "waiting": False, "patient": None},
            ]
            for _ in range(N_nodes)
        ]

        # ── Surgeon availability model ────────────────────────────────
        # Each FRSD has SURGEONS_PER_NODE surgeons (default 2 — doctrinal).
        # Each surgeon has a MAX_SURGEON_HOURS rolling 24h ops tempo limit.
        # Surgeon tracks:
        #   hours_today  — cumulative OR hours in current 24h rolling window
        #   window_start — when their current 24h window began
        #   available    — False if currently in a case (busy_until)
        # A surgeon is considered rested when hours_today < MAX_SURGEON_HOURS.
        # Window resets 24h after window_start (rolling, not calendar).
        n_surgeons          = config.get("SURGEONS_PER_NODE", 2)
        max_surgeon_hours   = config.get("MAX_SURGEON_HOURS", 12.0)
        self.max_surgeon_hours = max_surgeon_hours
        self.surgeon_shortage_hours = 0.0  # cumulative hrs OR idle due to no surgeon
        self.surgeons = [
            [
                {"hours_today": 0.0, "window_start": 0.0, "busy_until": 0.0}
                for _ in range(n_surgeons)
            ]
            for _ in range(N_nodes)
        ]

        # Per-node queues and pools
        self.surgery_queue = [[] for _ in range(N_nodes)]
        self.post_op_pool  = [[] for _ in range(N_nodes)]
        self.pcc_pool      = [[] for _ in range(N_nodes)]

        # Kit economy — per node (each FRSD has its own kit stock)
        kits_total              = config.get("DCS_KITS", 30)
        kits_per_node           = kits_total // N_nodes
        self.dcs_kits_available   = [kits_per_node] * N_nodes
        self.kit_reprocess_time   = config.get("KIT_REPROCESS_TIME", 2.0)
        self.kits_in_reprocessing = [[] for _ in range(N_nodes)]

        # Global counters (summed across nodes for output)
        self.surgeries_completed    = 0
        self.surgeries_started      = 0
        self.surgery_wait_deaths    = 0
        self.postop_overflow_deaths = 0
        self.pcc_deaths             = 0
        self.pcc_entries            = 0
        self.postop_entries         = 0
        self.postop_exits           = 0
        self.pacu_deaths            = 0   # died in PACU before evac (exceeds pcc_survival_hours)

        # ── Outcome-based mortality counters ─────────────────────────
        # Assumptions (configurable):
        #   - No transfusion (T1 or T2 unmet demand) → 100% mortality
        #   - Partial transfusion (T2 partial tx)    → 50% mortality (deterministic)
        #   - Full MTP                               → 0% additional mortality here
        #     (post-op deaths captured separately via pacu_deaths / pcc_deaths)
        self.PARTIAL_TX_MORTALITY   = config.get("PARTIAL_TX_MORTALITY", 0.50)
        self.deaths_no_blood_t2     = 0   # T2 unmet — no transfusion at all
        self.deaths_partial_tx      = 0   # T2 partial — died despite some blood
        self.deaths_no_blood_t1     = 0   # T1 unmet — OR couldn't start (blood prereq)

        # ── T2 PCC survival window (Bellamy 1984, Fig. 1) ────────────
        # T2 patients who receive any blood enter PCC after treatment.
        # Their survival clock reflects the sepsis/MOF curve from Bellamy:
        # hemorrhage controlled by transfusion, but without definitive
        # surgical care and antibiotics, peritonitis/empyema/soft-tissue
        # infection drives a secondary mortality peak at 1-3 days.
        # Default 72h = sepsis inflection for abdominal/thoracic wounds.
        # Configurable down to 48h (fast peritonitis) or up to 168h
        # (slow soft-tissue infection, extremity wounds).
        # Ref: Bellamy RF. Mil Med. 1984;149(2):55-62. Table III, Fig. 1.
        self.t2_pcc_survival_hours  = config.get("T2_PCC_SURVIVAL_HOURS", 72.0)
        self.t2_pcc_deaths          = 0   # T2 died in PCC before CASEVAC
        self.t2_pcc_entries         = 0   # T2 patients who entered PCC
        self.t2_pcc_pool            = [[] for _ in range(N_nodes)]
        self.or_blocked_time        = 0.0
        self.or_operating_time      = 0.0
        self.pacu_full_time         = 0.0
        self.kit_stockouts          = 0
        self.evac_count             = 0   # total patients evacuated

        # ── CASEVAC capacity (LSCO assumption) ───────────────────────
        # No dedicated MEDEVAC in LSCO due to contested airspace.
        # Evacuation occurs on returning resupply flights only (CASEVAC).
        # A utility UH-60 configured for CASEVAC carries 4 litter patients
        # (seats removed per TM 1-1520-237-10; compare HH-60M MEDEVAC = 6).
        # TODO: add opportunistic MEDEVAC sorties as a config option for
        #       permissive/semi-permissive airspace scenarios.
        self.casevac_capacity       = config.get("CASEVAC_CAPACITY", 4)

        # ── Blood inventory ───────────────────────────────────────
        N = config["N"]
        self.B_state = {k: np.full(N, v, dtype=float)
                        for k, v in config["INIT_STOCK"].items()}
        self.B_state["CAS"].fill(0)

        self.NR_state = np.full(N, config["N_total"], dtype=float)
        self.NU_state = np.zeros(N, dtype=float)

        self.wbb_queues = [[] for _ in range(N)]
        self.plt_queues = [[] for _ in range(N)]
        self.nu_queues  = [[] for _ in range(N)]
        for i in range(N):
            if config["INIT_STOCK"]["WBB"] > 0:
                self.wbb_queues[i].append([0.0, config["INIT_STOCK"]["WBB"]])
            if config["INIT_STOCK"]["PLT"] > 0:
                self.plt_queues[i].append([0.0, config["INIT_STOCK"]["PLT"]])

        self.cumulative_wbb_generated = np.zeros(N)
        self.FWB_from_WIA             = np.zeros(N)
        self.FWB_from_DD              = np.zeros(N)

        # ── Logging ───────────────────────────────────────────────
        n_steps = len(config["time"])
        self.unmet_demand_log = np.zeros((N, n_steps))
        self.full_mtp_log     = np.zeros((N, n_steps))
        self.partial_tx_log   = np.zeros((N, n_steps))
        # Unit-level tracking for fill rate = units_delivered / units_demanded
        self.units_delivered  = 0.0
        self.units_demanded   = 0.0

        # Fractional donor accumulators — same pattern as T1/T2 backlogs
        # int() truncation at dt=0.1 kills all donor events (0.02 donors/step → 0)
        self._dd_backlog  = np.zeros(N_nodes)  # KIA direct donation
        self._wia_backlog = np.zeros(N_nodes)  # WIA whole blood

        # T2 demand accumulator (fractional patients, per node)
        self._t2_backlog = np.zeros(N)

        # ── Logistics schedules ───────────────────────────────────
        self.blackout_windows = config.get("blackout_windows", [])

        resupply_delay = config.get("resupply_delay", 3.0)
        self.resupply_schedule = [[] for _ in range(N)]
        for i in range(N):
            t = 0
            while t < config["T_max"]:
                arrival = t + resupply_delay
                # If arrival falls in a blackout, push to blackout_end + delay
                for bk_start, bk_end in self.blackout_windows:
                    if bk_start <= arrival <= bk_end:
                        arrival = bk_end + resupply_delay
                        break
                if arrival < config["T_max"] and arrival not in self.resupply_schedule[i]:
                    self.resupply_schedule[i].append(arrival)
                t += self.interval_hours

        self.redistribution_events    = []
        self.last_redistribution_check = -12.0

    # ----------------------------------------------------------------
    # [B] T1 surgical casualty generation — stores total_units on patient
    # ----------------------------------------------------------------
    def _generate_t1_surgical_casualties(self, t, casualties):
        """
        Per-node T1 backlog: each node accumulates its own fractional
        casualties and queues patients into that node's surgery queue.
        Blood will be drawn exclusively from that node's inventory.
        """
        N = self.config["N"]
        if not hasattr(self, "_t1_backlog"):
            self._t1_backlog = [0.0] * N

        for i in range(N):
            self._t1_backlog[i] += casualties["T1"][i]
            n = int(self._t1_backlog[i])
            self._t1_backlog[i] -= n
            for _ in range(n):
                # Patient queued at their arrival node — blood draws from node i only
                self.surgery_queue[i].append({
                    "arrival":          t,
                    "deadline":         t + self.mean_time_to_death,
                    "total_units":      self.t1_units,
                    "blood_drain_rate": None,   # set when surgery starts
                    "units_remaining":  self.t1_units,
                    "node":             i,
                })

    # ----------------------------------------------------------------
    # [B] OR processing — per node, per OR table
    # Each node has 2 ORs. An OR at node i draws blood only from node i.
    # ----------------------------------------------------------------
    def _process_or(self, t):
        dt    = self.config["dt"]
        N     = self.config["N"]
        t_idx = int(round(t / dt))

        for i in range(N):
            # Return any reprocessed kits to this node
            still_reprocessing = []
            for kit in self.kits_in_reprocessing[i]:
                if t >= kit["ready_time"]:
                    self.dcs_kits_available[i] = min(
                        self.dcs_kits_available[i] + 1,
                        self.config.get("DCS_KITS", 30) // N
                    )
                else:
                    still_reprocessing.append(kit)
            self.kits_in_reprocessing[i] = still_reprocessing

            for OR in self.ORs[i]:

                # ── CASE 1: OR busy — draw blood from node i ─────
                if t < OR["busy_until"]:
                    self.or_operating_time += dt

                    patient = OR["patient"]
                    if patient is not None and patient["blood_drain_rate"] is not None:
                        units_needed = patient["blood_drain_rate"] * dt
                        units_needed = min(units_needed, patient["units_remaining"])
                        if units_needed > 0:
                            patient["units_remaining"] -= units_needed
                            t_idx_local = t_idx

                            # Draw from node i only — WBB first, then RBC, then FFP
                            original_needed = units_needed
                            self.units_demanded += units_needed
                            while units_needed > 1e-6:
                                wbb_avail = sum(q for _, q in self.wbb_queues[i])
                                if wbb_avail >= 1.0:
                                    draw = min(units_needed, wbb_avail)
                                    self.wbb_queues[i], withdrawn = withdraw_from_queue(
                                        self.wbb_queues[i], draw)
                                    self.B_state["WBB"][i] -= withdrawn
                                    units_needed -= withdrawn
                                    self.units_delivered += withdrawn
                                    self.full_mtp_log[i, t_idx_local] += withdrawn / self.t1_units
                                elif self.B_state["RBC"][i] >= 1.0:
                                    draw = min(units_needed, self.B_state["RBC"][i])
                                    self.B_state["RBC"][i] -= draw
                                    units_needed -= draw
                                    self.units_delivered += draw
                                    self.partial_tx_log[i, t_idx_local] += draw / self.t1_units
                                elif self.B_state["FFP"][i] >= 1.0:
                                    draw = min(units_needed, self.B_state["FFP"][i])
                                    self.B_state["FFP"][i] -= draw
                                    units_needed -= draw
                                    self.units_delivered += draw
                                    self.partial_tx_log[i, t_idx_local] += draw / self.t1_units
                                else:
                                    self.unmet_demand_log[i, t_idx_local] += units_needed / self.t1_units
                                    break
                    continue

                # Surgery just finished → enter waiting state
                if OR["patient"] is not None and not OR["waiting"]:
                    OR["waiting"] = True

                # ── CASE 2: OR finished, offload patient ──────────
                if OR["waiting"]:
                    patient = OR["patient"]

                    if t > patient["deadline"]:
                        self.postop_overflow_deaths += 1
                        OR["patient"] = None
                        OR["waiting"] = False
                        continue

                    if len(self.post_op_pool[i]) < self.postop_max:
                        self.post_op_pool[i].append({
                            "entry": t,
                            "exit":  t + self.postop_exit_duration,
                            "patient": patient,
                        })
                        self.postop_entries += 1
                    elif len(self.pcc_pool[i]) < self.pcc_max:
                        self.pcc_pool[i].append({"entry": t, "patient": patient})
                        self.pcc_entries += 1
                    else:
                        self.or_blocked_time += dt
                        continue  # OR stays blocked

                    self.surgeries_completed += 1
                    OR["patient"]    = None
                    OR["waiting"]    = False
                    OR["busy_until"] = t
                    continue

                # ── CASE 3: OR idle — pull from this node's queue ─
                # Purge dead patients from this node's queue
                alive = []
                for p in self.surgery_queue[i]:
                    if t > p["deadline"] + 1e-6:
                        self.surgery_wait_deaths += 1
                    else:
                        alive.append(p)
                self.surgery_queue[i] = alive

                if not self.surgery_queue[i]:
                    continue
                if self.dcs_kits_available[i] <= 0:
                    self.kit_stockouts += 1
                    continue

                # ── Blood prerequisite check (Option A) ──────────────
                blood_avail = (
                    sum(q for _, q in self.wbb_queues[i])
                    + self.B_state["RBC"][i]
                    + self.B_state["FFP"][i]
                )
                if blood_avail < 1.0:
                    continue  # OR stays idle — no blood to operate

                # ── Surgeon availability check ────────────────────────
                # Find a surgeon who (a) is not currently in a case and
                # (b) has hours remaining in their rolling 24h window.
                # Rolling window: resets 24h after window_start.
                assigned_surgeon = None
                for surg in self.surgeons[i]:
                    # Roll the window if 24h have passed since it opened
                    if t - surg["window_start"] >= 24.0:
                        surg["hours_today"]  = 0.0
                        surg["window_start"] = t
                    # Surgeon is free if their last case has finished
                    surgeon_free = t >= surg["busy_until"]
                    hours_left   = self.max_surgeon_hours - surg["hours_today"]
                    if surgeon_free and hours_left >= self.or_duration:
                        assigned_surgeon = surg
                        break

                if assigned_surgeon is None:
                    # No surgeon available — OR stays idle, patient ages toward deadline
                    self.surgeon_shortage_hours += dt
                    continue

                patient = self.surgery_queue[i].pop(0)
                self.surgeries_started += 1

                # Commit surgeon to this case
                assigned_surgeon["busy_until"]  = t + self.or_duration
                assigned_surgeon["hours_today"] += self.or_duration

                self.dcs_kits_available[i] -= 1
                self.kits_in_reprocessing[i].append(
                    {"ready_time": t + self.kit_reprocess_time}
                )

                patient["blood_drain_rate"] = self.t1_units / self.or_duration

                OR["busy_until"] = t + self.or_duration
                OR["patient"]    = patient
                OR["waiting"]    = False

    # ----------------------------------------------------------------
    # Post-op / PACU — per node
    #
    # LSCO CASEVAC MODEL:
    # Patients remain in PACU until evacuated on a returning resupply
    # flight. No dedicated MEDEVAC assumed (contested airspace).
    # Evacuation is triggered by _process_evac() on each resupply arrival.
    # Patients who exceed pcc_survival_hours before evac die in PACU.
    # PACU overflow (> postop_max) goes to PCC; PCC overflow blocks the OR.
    # ----------------------------------------------------------------
    def _process_postop(self, t):
        N = self.config["N"]
        for i in range(N):
            # Kill patients who have been in PACU too long without evac
            still_in = []
            for p in self.post_op_pool[i]:
                time_in_pacu = t - p["entry"]
                if time_in_pacu >= self.pcc_survival_hours:
                    self.pacu_deaths += 1
                else:
                    still_in.append(p)
            self.post_op_pool[i] = still_in

            # Refill PACU from PCC when space opens
            while len(self.post_op_pool[i]) < self.postop_max and self.pcc_pool[i]:
                pcc_patient = self.pcc_pool[i].pop(0)
                self.post_op_pool[i].append({
                    "entry":   pcc_patient["entry"],   # preserve original entry time
                    "patient": pcc_patient,
                })
                self.postop_entries += 1

            if len(self.post_op_pool[i]) >= self.postop_max:
                self.pacu_full_time += self.dt

    # ----------------------------------------------------------------
    # CASEVAC evacuation — fires on each resupply arrival
    # Evacuates up to casevac_capacity patients per flight, FIFO.
    # Called from the resupply block in _step().
    # ----------------------------------------------------------------
    def _process_evac(self, t, node_i):
        """
        Evacuate patients on the returning resupply flight.
        Capacity = 4 litter patients (utility UH-60 CASEVAC config).
        LSCO assumption: no dedicated MEDEVAC — contested airspace.

        Priority order (clinical acuity):
          1. T1 post-op (PACU) — most critical, post-surgical
          2. T2 PCC — monitored but deteriorating on sepsis clock

        Both pools compete for the same 4 slots per flight.
        FIFO within each priority tier.
        """
        slots = self.casevac_capacity

        # Priority 1: T1 post-op PACU patients
        remaining_t1 = []
        for p in self.post_op_pool[node_i]:
            if slots > 0:
                self.postop_exits += 1
                self.evac_count   += 1
                slots -= 1
            else:
                remaining_t1.append(p)
        self.post_op_pool[node_i] = remaining_t1

        # Priority 2: T2 PCC patients (remaining slots)
        remaining_t2 = []
        for p in self.t2_pcc_pool[node_i]:
            if slots > 0:
                self.evac_count += 1
                slots -= 1
            else:
                remaining_t2.append(p)
        self.t2_pcc_pool[node_i] = remaining_t2

    # ----------------------------------------------------------------
    # [F] PCC mortality — per node
    #
    # Two populations in PCC:
    #   pcc_pool    — T1 post-op PACU overflow (rare in continuous flow)
    #   t2_pcc_pool — T2 submassive hemorrhage post-treatment
    #
    # T2 PCC clock = Bellamy (1984) sepsis/MOF curve:
    #   hemorrhage controlled by transfusion, but peritonitis/empyema/
    #   soft-tissue infection drives secondary mortality at 1-3 days
    #   without definitive surgical care or antibiotics.
    # ----------------------------------------------------------------
    def _process_pcc(self, t):
        N = self.config["N"]
        for i in range(N):
            # T1 overflow PCC (PACU overflow — structurally rare)
            still_in = []
            for p in self.pcc_pool[i]:
                if t - p["entry"] >= self.pcc_survival_hours:
                    self.pcc_deaths += 1
                else:
                    still_in.append(p)
            self.pcc_pool[i] = still_in

            # T2 PCC — sepsis/MOF clock (Bellamy 1984)
            still_in_t2 = []
            for p in self.t2_pcc_pool[i]:
                if t - p["entry"] >= self.t2_pcc_survival_hours:
                    self.t2_pcc_deaths += 1
                else:
                    still_in_t2.append(p)
            self.t2_pcc_pool[i] = still_in_t2

    # ----------------------------------------------------------------
    # Main step
    # ----------------------------------------------------------------
    def _step(self, t):
        N     = self.config["N"]
        dt    = self.dt
        t_prev = t - dt
        t_idx  = int(round(t / dt))

        # ── Redistribution planning ───────────────────────────────
        # [H] Only schedule transfers when NOT in a blackout
        if t - self.last_redistribution_check >= self.config["redistribution_check_interval"]:
            if not in_blackout(t, self.blackout_windows):
                needy_nodes = {k: [] for k in ["RBC", "FFP", "PLT", "CRYO", "WBB"]}
                for i in range(N):
                    cur_wbb = sum(q for _, q in self.wbb_queues[i])
                    cur_plt = sum(q for _, q in self.plt_queues[i])
                    if self.B_state["RBC"][i]  < self.config["CRITICAL_THRESHOLD"]["RBC"]:   needy_nodes["RBC"].append(i)
                    if self.B_state["FFP"][i]  < self.config["CRITICAL_THRESHOLD"]["FFP"]:   needy_nodes["FFP"].append(i)
                    if cur_plt                  < self.config["CRITICAL_THRESHOLD"]["PLT"]:   needy_nodes["PLT"].append(i)
                    if self.B_state["CRYO"][i] < self.config["CRITICAL_THRESHOLD"]["CRYO"]:  needy_nodes["CRYO"].append(i)
                    if cur_wbb                  < self.config["CRITICAL_THRESHOLD"]["WBB"]:   needy_nodes["WBB"].append(i)

                for k in ["RBC", "FFP", "PLT", "CRYO", "WBB"]:
                    for needy in needy_nodes[k]:
                        best = -1; best_t = np.inf
                        for donor in range(N):
                            if donor == needy: continue
                            if not np.isfinite(self.config["travel_time_matrix"][donor, needy]): continue
                            if k == "WBB":
                                donor_stock = sum(q for _, q in self.wbb_queues[donor])
                            elif k == "PLT":
                                donor_stock = sum(q for _, q in self.plt_queues[donor])
                            else:
                                donor_stock = self.B_state[k][donor]
                            ok = donor_stock > self.config["SAFE_THRESHOLD"][k]
                            if ok and self.config["travel_time_matrix"][donor, needy] < best_t:
                                best = donor; best_t = self.config["travel_time_matrix"][donor, needy]

                        if best != -1:
                            if k == "WBB":
                                needy_stock  = sum(q for _, q in self.wbb_queues[needy])
                                donor_surplus = sum(q for _, q in self.wbb_queues[best]) - self.config["SAFE_THRESHOLD"][k]
                            elif k == "PLT":
                                needy_stock  = sum(q for _, q in self.plt_queues[needy])
                                donor_surplus = sum(q for _, q in self.plt_queues[best]) - self.config["SAFE_THRESHOLD"][k]
                            else:
                                needy_stock  = self.B_state[k][needy]
                                donor_surplus = self.B_state[k][best] - self.config["SAFE_THRESHOLD"][k]

                            qty = max(0.0, min(
                                self.config["CRITICAL_THRESHOLD"][k] - needy_stock,
                                donor_surplus))
                            if qty <= 0: continue

                            delay = get_flight_delay(best, needy, self.config["travel_time_matrix"])
                            planned_arrival = t + delay

                            # [H] If planned arrival falls inside a blackout, push to end of that window
                            if in_blackout(planned_arrival, self.blackout_windows):
                                for start, end in self.blackout_windows:
                                    if start <= planned_arrival <= end:
                                        planned_arrival = end + get_flight_delay(
                                            best, needy, self.config["travel_time_matrix"])
                                        break

                            self.redistribution_events.append(
                                (best, needy, k, qty, planned_arrival))

            self.last_redistribution_check = t

        # ── Casualty generation ───────────────────────────────────
        casualties = generate_casualties(N, t, self.casualty_rate, dt, self.config)

        # T1 → surgery queue
        self._generate_t1_surgical_casualties(t, casualties)

        # T2 → component blood demand (fractional accumulator)
        self._t2_backlog += casualties["T2"]
        t2_patients  = self._t2_backlog.astype(int)
        self._t2_backlog -= t2_patients

        # ── Opportunistic FWB collection ──────────────────────────
        # [D] Blood enters queue at processing delay age, not age 0
        # Fractional accumulator pattern — int() truncation at dt=0.1
        # kills all events (0.02 donors/step → 0 every step).
        DD_UNITS_PER_DONOR = 3.0  # mean of uniform(2,4), McWhirter consistent
        for i in range(N):
            cas_i = casualties["CAS"][i]
            if cas_i <= 0:
                continue

            # KIA donors — exclude those who needed MTP
            kia_total       = cas_i * self.config["PCT_KIA"]
            t1_i            = casualties["T1"][i]
            kia_not_needing = max(0.0, kia_total - t1_i)
            self._dd_backlog[i]  += self.pct_kia_suitable * kia_not_needing

            # Stable WIA donors — not KIA, not T1/T2
            stable_wia      = max(0.0, cas_i - kia_total - t1_i - casualties["T2"][i])
            self._wia_backlog[i] += self.pct_wia_stable * stable_wia

            # Flush whole donors when accumulator >= 1
            dd_donors  = int(self._dd_backlog[i])
            self._dd_backlog[i]  -= dd_donors
            wia_donors = int(self._wia_backlog[i])
            self._wia_backlog[i] -= wia_donors

            dd_units  = dd_donors  * DD_UNITS_PER_DONOR
            wia_units = wia_donors * self.config["UNITS_PER_WIA"]

            if dd_units > 0:
                self.wbb_queues[i].append([self.fwb_processing_delay, dd_units])
                self.B_state["WBB"][i] += dd_units
                self.FWB_from_DD[i]    += dd_units
            if wia_units > 0:
                self.wbb_queues[i].append([self.fwb_processing_delay, wia_units])
                self.B_state["WBB"][i]  += wia_units
                self.FWB_from_WIA[i]    += wia_units

        # ── Age, expire products ──────────────────────────────────
        for i in range(N):
            # WBB and PLT: FIFO age queues
            self.wbb_queues[i] = [[a + dt, q] for a, q in self.wbb_queues[i]
                                  if a + dt <= self.config["EXPIRY_WBB"]]
            self.plt_queues[i] = [[a + dt, q] for a, q in self.plt_queues[i]
                                  if a + dt <= self.config["EXPIRY_PLT"]]
            self.B_state["WBB"][i] = sum(q for _, q in self.wbb_queues[i])
            self.B_state["PLT"][i] = sum(q for _, q in self.plt_queues[i])
            self.B_state["CAS"][i] += casualties["CAS"][i]

            # [G] First-order component expiration (Modeling.md: E = X/tau)
            self.B_state["RBC"][i]  -= (dt / self.tau_RBC)  * self.B_state["RBC"][i]
            self.B_state["FFP"][i]  -= (dt / self.tau_FFP)  * self.B_state["FFP"][i]
            self.B_state["CRYO"][i] -= (dt / self.tau_CRYO) * self.B_state["CRYO"][i]

        # ── Apply redistribution arrivals ─────────────────────────
        for event in list(self.redistribution_events):
            frm, to, k, qty, t_arr = event
            if due_this_step(t, t_prev, t_arr):
                if k == "PLT":
                    self.plt_queues[to].append([0.0, qty])
                    self.plt_queues[frm], _ = withdraw_from_queue(self.plt_queues[frm], qty)
                elif k == "WBB":
                    self.wbb_queues[to].append([0.0, qty])
                    self.wbb_queues[frm], _ = withdraw_from_queue(self.wbb_queues[frm], qty)
                else:
                    self.B_state[k][to]  += qty
                    self.B_state[k][frm]  = max(0.0, self.B_state[k][frm] - qty)
                self.B_state["WBB"][frm] = sum(q for _, q in self.wbb_queues[frm])
                self.B_state["PLT"][frm] = sum(q for _, q in self.plt_queues[frm])
                self.redistribution_events.remove(event)

        # ── T2 demand fulfillment (component therapy at arrival) ──
        # One iteration per T2 patient. Each patient needs t2_units (6u) total:
        #   half RBC, half FFP, plus PLT if available.
        for i in range(N):
            n_patients = self._t2_backlog  # fractional accumulator (shared scalar — see below)
            # Use per-step integer patient count
            n_pts = int(t2_patients[i])
            frac  = t2_patients[i] - n_pts
            # handle fractional patient via backlog (reuse existing _t2_backlog mechanism)

            for _ in range(n_pts):
                need     = self.t2_units          # 6 units per patient
                rbc_need = need / 2               # 3u RBC
                ffp_need = need / 2               # 3u FFP
                rbc_ok   = self.B_state["RBC"][i] >= rbc_need
                ffp_ok   = self.B_state["FFP"][i] >= ffp_need
                plt_ok   = sum(q for _, q in self.plt_queues[i]) >= self.config["PLT_PER_PACKET"]

                self.units_demanded += need  # always demand full 6u per T2 patient

                if rbc_ok and ffp_ok and plt_ok:
                    self.B_state["RBC"][i] -= rbc_need
                    self.B_state["FFP"][i] -= ffp_need
                    self.plt_queues[i], _used = withdraw_from_queue(
                        self.plt_queues[i], self.config["PLT_PER_PACKET"])
                    self.units_delivered += need
                    self.full_mtp_log[i, t_idx] += 1
                    # Full MTP → enters PCC; sepsis clock starts (Bellamy Fig. 1)
                    self.t2_pcc_pool[i].append({"entry": t, "tx": "full"})
                    self.t2_pcc_entries += 1
                elif rbc_ok and ffp_ok:
                    self.B_state["RBC"][i] -= rbc_need
                    self.B_state["FFP"][i] -= ffp_need
                    self.units_delivered += rbc_need + ffp_need
                    self.partial_tx_log[i, t_idx] += 1
                    self.deaths_partial_tx += self.PARTIAL_TX_MORTALITY   # 50% die immediately
                    # Surviving 50% enter PCC with same sepsis clock
                    self.t2_pcc_pool[i].append({"entry": t, "tx": "partial"})
                    self.t2_pcc_entries += 1
                elif rbc_ok:
                    draw = min(rbc_need, self.B_state["RBC"][i])
                    self.B_state["RBC"][i] -= draw
                    self.units_delivered += draw
                    self.partial_tx_log[i, t_idx] += 1
                    self.deaths_partial_tx += self.PARTIAL_TX_MORTALITY   # 50% die immediately
                    # Surviving 50% enter PCC
                    self.t2_pcc_pool[i].append({"entry": t, "tx": "partial"})
                    self.t2_pcc_entries += 1
                else:
                    self.unmet_demand_log[i, t_idx] += 1
                    self.deaths_no_blood_t2 += 1   # no transfusion → 100% mortality, never reaches PCC
            # fractional remainder accumulates in existing _t2_backlog (already handled above)

        # ── WBB donations ─────────────────────────────────────────
        for i in range(N):
            has_demand = t2_patients[i] > 0 or len(self.surgery_queue) > 0
            if (has_demand
                    and t > self.config["t_setup"][i]
                    and self.NR_state[i] > 0
                    and self.B_state["WBB"][i] < self.config["B_max"]):
                max_draw = min(
                    self.wbb_rate * dt,
                    self.NR_state[i],
                    self.config["B_max"] - self.B_state["WBB"][i])
                if max_draw > 0:
                    # [D] Processing delay on WBB donations too
                    self.wbb_queues[i].append([self.fwb_processing_delay, max_draw])
                    self.NR_state[i]               -= max_draw
                    self.NU_state[i]               += max_draw
                    self.cumulative_wbb_generated[i] += max_draw
                    self.nu_queues[i].append([0.0, max_draw])

        # ── OR / PACU / PCC ───────────────────────────────────────
        # (Kit reprocessing handled inside _process_or per node)
        self._process_or(t)
        self._process_postop(t)
        self._process_pcc(t)

        # ── Resupply arrivals ─────────────────────────────────────
        for i in range(N):
            for arrival_time in self.resupply_schedule[i]:
                if due_this_step(t, t_prev, arrival_time):
                    delivery = 120.0  # fixed 120u per flight — intentional planning constraint

                    # [E] Incremental kit resupply per node — do NOT reset reprocessing
                    kits_delivered = self.config.get("KITS_PER_RESUPPLY", 10) // self.config["N"]
                    kits_max_per_node = self.config.get("DCS_KITS", 30) // self.config["N"]
                    self.dcs_kits_available[i] = min(
                        self.dcs_kits_available[i] + kits_delivered,
                        kits_max_per_node)

                    # Proportional blood product delivery by depletion
                    baseline = self.config["INIT_STOCK"]
                    live_plt = sum(q for _, q in self.plt_queues[i])
                    depletion = {
                        k: max(0.0, 1.0 - (
                            (self.B_state[k][i] + (live_plt if k == "PLT" else 0.0))
                            / baseline[k]))
                        for k in ["RBC", "FFP", "PLT", "CRYO"]
                    }
                    total_dep = sum(depletion.values()) + 1e-6
                    for k in ["RBC", "FFP", "PLT", "CRYO"]:
                        share  = depletion[k] / total_dep
                        units  = delivery * share
                        current = (self.B_state["RBC"][i] + self.B_state["FFP"][i]
                                   + self.B_state["CRYO"][i]
                                   + sum(q for _, q in self.wbb_queues[i])
                                   + sum(q for _, q in self.plt_queues[i]))
                        space  = max(0.0, self.config["MAX_STORAGE"] - current)
                        add    = min(units, space)
                        if add <= 0:
                            continue
                        if k == "PLT":
                            self.plt_queues[i].append([0.0, add])
                        else:
                            self.B_state[k][i] += add
                    # Evacuate post-op patients on the return flight
                    self._process_evac(t, i)
                    # do NOT break — all nodes must be checked every step

        # ── Donor recovery ────────────────────────────────────────
        for i in range(N):
            recovered = 0.0
            new_q = []
            for age, qty in self.nu_queues[i]:
                age_new = age + dt
                if age_new >= self.config["tau"]:
                    recovered += qty
                else:
                    new_q.append([age_new, qty])
            if recovered > 0:
                self.NR_state[i] += recovered
                self.NU_state[i]  = max(0.0, self.NU_state[i] - recovered)
            self.nu_queues[i] = new_q

        # ── Clamp negatives ───────────────────────────────────────
        for k in self.B_state:
            self.B_state[k] = np.maximum(self.B_state[k], 0.0)

    # ----------------------------------------------------------------
    def run(self):
        for t in self.config["time"]:
            self._step(t)

        total_unmet   = np.sum(self.unmet_demand_log)
        cumulative_deaths = (
            self.surgery_wait_deaths      # T1: OR never started
          + self.postop_overflow_deaths   # T1: died on table
          + self.pacu_deaths              # T1: post-op, no evac
          + self.pcc_deaths               # T1: PACU overflow
          + self.deaths_no_blood_t2       # T2: zero transfusion
          + round(self.deaths_partial_tx) # T2: partial transfusion, 50% rate
          + self.t2_pcc_deaths            # T2: sepsis/MOF in PCC (Bellamy Fig. 1)
        )
        total_failures = np.sum(self.partial_tx_log + self.unmet_demand_log)
        fill_rate     = (self.units_delivered / self.units_demanded * 100) if self.units_demanded > 0 else 100.0

        def first_nonzero_time(log):
            idxs = np.where(log.sum(axis=0) > 0)[0]
            return idxs[0] * self.dt if len(idxs) > 0 else self.config["T_max"]

        return {
            # Blood outcomes
            "unmet_total":       total_unmet,
            "cumulative_deaths": cumulative_deaths,
            "fill_rate":         fill_rate,   # units delivered / units demanded
            "units_delivered":   self.units_delivered,
            "units_demanded":    self.units_demanded,
            "full_mtp_total":    np.sum(self.full_mtp_log),
            "partial_tx_total":  np.sum(self.partial_tx_log),
            "first_failure_MTP": first_nonzero_time(self.partial_tx_log + self.unmet_demand_log),
            "first_failure_time": first_nonzero_time(self.unmet_demand_log),
            "surgeon_shortage_hours": self.surgeon_shortage_hours,
            "FWB_from_DD":       np.sum(self.FWB_from_DD),
            "FWB_from_WIA":      np.sum(self.FWB_from_WIA),
            "FWB_from_WBB":      np.sum(self.cumulative_wbb_generated),
            # Surgical outcomes
            "surgeries_started":    self.surgeries_started,
            "surgeries_completed":  self.surgeries_completed,
            "surgery_wait_deaths":  self.surgery_wait_deaths,
            "intraop_deaths":       self.postop_overflow_deaths,   # died on table: surgery finished after deadline
            "pcc_deaths":           self.pcc_deaths,
            "pcc_entries":          self.pcc_entries,
            "pacu_deaths":          self.pacu_deaths,
            "evac_count":           self.evac_count,
            # ── Cumulative death totals ───────────────────────────────────
            # Sources:
            #   surgery_wait_deaths   — T1 in queue, blood unavailable or deadline passed
            #   intraop_deaths        — T1 on table, deadline passed mid-surgery
            #   pacu_deaths           — post-op, no CASEVAC before survival window expires
            #   pcc_deaths            — PACU overflow into PCC, same survival clock
            #   deaths_no_blood_t2    — T2 received zero transfusion
            #   deaths_partial_tx     — T2 partial transfusion, 50% mortality rate
            "t2_pcc_entries":       self.t2_pcc_entries,
            "t2_pcc_deaths":        self.t2_pcc_deaths,
            "deaths_no_blood_t2":   self.deaths_no_blood_t2,
            "deaths_partial_tx":    round(self.deaths_partial_tx),
            "kit_stockouts":        self.kit_stockouts,
            "dcs_queue_final":      sum(len(q) for q in self.surgery_queue),
            "postop_entries":       self.postop_entries,
            "postop_exits":         self.postop_exits,
            "or_blocked_time":      self.or_blocked_time,
            "or_operating_time":    self.or_operating_time,
            "pacu_full_time":       self.pacu_full_time,
            # Config echo
            "mttd":             self.mean_time_to_death,
            "PACU_EXIT_TIMES":  self.postop_exit_duration,
        }


# ------------------------------------------------------------------
def run_single_simulation_oo(config):
    sim     = Simulation(config)
    results = sim.run()
    results.update({
        "casualties_per_day": config["casualty_rate"] * 24,
        "resupply_interval":  config["interval_hours"],
        "wbb_rate":           config["wbb_rate"],
        "pct_kia_suitable":   config["pct_kia_suitable"],
        "pct_wia_stable":     config["pct_wia_stable"],
    })
    return results
