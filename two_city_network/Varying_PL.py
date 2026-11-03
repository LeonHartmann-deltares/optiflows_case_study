"""Two-city network: Pipe Length Sweep (PL) - Volledig vrije topologie.

Voor elke lengte PL wordt het model opgelost waarbij de solver 
volledige vrijheid heeft om de goedkoopste netwerk-layout te kiezen.
Inclusief referentie-iteratie voor de pompen.
"""

from __future__ import annotations
import copy
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from sklearn.cluster import KMeans

from optiflows import goal
from optiflows.components.hydraulics import Demand, Pipe, Source, Pump
from optiflows.components.hydraulics.catalog import DEFAULT_PIPE_CATALOG
from optiflows.components.hydraulics.junction import Junction
from optiflows.core.financial import FinancialParameters
from optiflows.network import Network

# --------------------------------------------------------------------------
# 1. Input data & configuration
# --------------------------------------------------------------------------

HOURS_PER_YEAR = 8760
SECONDS_PER_HOUR = 3600.0

CITY_COLUMNS = {"DL": "Large City", "DM": "Medium City"}
DEMANDS = tuple(CITY_COLUMNS)
CITY_JUNCTIONS = {"DL": "JL", "DM": "JM"}

CSV_NAME = "water_demand_profiles_2025.csv"
CSV_PATH = next(
    (p for p in (Path(__file__).with_name(CSV_NAME), Path.cwd() / CSV_NAME) if p.exists()),
    Path(__file__).with_name(CSV_NAME),
)

V_MAX = 3.0
N_HYPERPLANES = 20
RESERVOIR_HEAD = 0.0
MIN_CITY_HEAD = 25.0
PUMP_HEAD_MAX = 50.0
PUMP_EFFICIENCY = 0.75
PUMP_FIXED_COST = 50000
ENERGY_COST_PER_WH = 0.12 / 1000

FINANCIAL = FinancialParameters(
    discount_rate=0.05,
    lifetime_y=30.0,
    operating_hours_y=float(HOURS_PER_YEAR),
)

GOALS = (
    {"name": "min_cost", "target": goal.total_cost, "direction": "min", "priority": 1},
)

# Sweep-instellingen
PL_LIST = sorted(set([3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0]))
K_CLUSTERS = 8
MAX_ITERS = 10          
REF_TOL = 0.01          
ALL_PIPES = ("PL", "PM", "PLM", "PML")
OUT_CSV = Path("pl_sweep_results_free.csv")

# --------------------------------------------------------------------------
# 2. Data processing
# --------------------------------------------------------------------------

def load_profiles() -> dict[str, list[float]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset {CSV_NAME} niet gevonden.")
    csv = pd.read_csv(CSV_PATH, index_col="tijd")
    return {node: (csv[col] / SECONDS_PER_HOUR).tolist() for node, col in CITY_COLUMNS.items()}

PROFILES = load_profiles()
N_HOURS = len(PROFILES[DEMANDS[0]])

TOTAL_PROFILE = [sum(PROFILES[d][h] for d in DEMANDS) for h in range(N_HOURS)]
SYSTEM_PEAK_HOUR = max(range(N_HOURS), key=lambda h: TOTAL_PROFILE[h])
SYSTEM_PEAK_FLOWS = {d: PROFILES[d][SYSTEM_PEAK_HOUR] for d in DEMANDS}
TOTAL_PEAK_FLOW = sum(SYSTEM_PEAK_FLOWS.values())
CITY_PEAK = {d: max(PROFILES[d]) for d in DEMANDS}

FLOW_MAX = {
    "PL": TOTAL_PEAK_FLOW,
    "PM": TOTAL_PEAK_FLOW,
    "PLM": CITY_PEAK["DM"],
    "PML": CITY_PEAK["DL"],
}

INITIAL_REFS = {
    "PL": {"head": 30.27, "flow": 0.4756},
    "PM": {"head": 28.49, "flow": 0.1585},
    "PLM": {"head": MIN_CITY_HEAD, "flow": FLOW_MAX["PLM"] / 2.0},
    "PML": {"head": MIN_CITY_HEAD, "flow": FLOW_MAX["PML"] / 2.0},
}

# --------------------------------------------------------------------------
# 3. Demand scenarios (k-means + separate peak hour)
# --------------------------------------------------------------------------

def get_demand_states_kmeans(k: int = 8) -> list[tuple]:
    states = [("peak", 1.0, SYSTEM_PEAK_FLOWS["DL"], SYSTEM_PEAK_FLOWS["DM"])]
    dl_rem = [PROFILES["DL"][i] for i in range(N_HOURS) if i != SYSTEM_PEAK_HOUR]
    dm_rem = [PROFILES["DM"][i] for i in range(N_HOURS) if i != SYSTEM_PEAK_HOUR]
    X = np.column_stack((dl_rem, dm_rem))
    labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
    for i in range(k):
        pts = X[labels == i]
        if len(pts) > 0:
            states.append((f"km_{i + 1:02d}", float(len(pts)),
                           float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))))
    return states

# --------------------------------------------------------------------------
# 4. Network specification and model building
# --------------------------------------------------------------------------

def make_topology(pl_length: float) -> tuple:
    return (
        ("PL", "JS", "JL", pl_length, None),
        ("PM", "JS", "JM", 3000.0, None),
        ("PLM", "JL", "JM", 2500.0, None),
        ("PML", "JM", "JL", 2500.0, None),
    )

def build_network(demand_states: list[tuple], best_refs: dict, topology: tuple) -> Network:
    network = Network()

    network.add("S1", Source(head=RESERVOIR_HEAD))
    network.add("JS", Junction(n_ports=3))
    network.add("JL", Junction(n_ports=4))
    network.add("JM", Junction(n_ports=4))

    for node in DEMANDS:
        network.add(node, Demand(flow=0.0))

    network.connect("S1.outlet", "JS.port_0")

    for pipe_id, _a, _b, length, fixed_diam in topology:
        pump_name = f"PU_{pipe_id}"
        pump = Pump(
            head_max=PUMP_HEAD_MAX,
            flow_max=FLOW_MAX[pipe_id],
            efficiency=PUMP_EFFICIENCY,
            energy_cost=ENERGY_COST_PER_WH,
            fixed_cost=PUMP_FIXED_COST,
            power_model="max",
            reference_head=best_refs[pipe_id]["head"],
            reference_flow=best_refs[pipe_id]["flow"],
        )
        network.add(pump_name, pump, optional=True)

        pipe = Pipe(
            length=length,
            catalog=DEFAULT_PIPE_CATALOG,
            flow_max=FLOW_MAX[pipe_id],
            v_min=0.0,
            v_max=V_MAX,
            n_hyperplanes=N_HYPERPLANES,
        )
        network.add(pipe_id, pipe, optional=(fixed_diam is None))
        network.connect(f"{pump_name}.outlet", f"{pipe_id}.inlet")

    network.connect("JS.port_1", "PU_PL.inlet")
    network.connect("PL.outlet", "JL.port_0")
    network.connect("PML.outlet", "JL.port_1")
    network.connect("JL.port_2", "PU_PLM.inlet")
    network.connect("JL.port_3", "DL.inlet")

    network.connect("JS.port_2", "PU_PM.inlet")
    network.connect("PM.outlet", "JM.port_0")
    network.connect("PLM.outlet", "JM.port_1")
    network.connect("JM.port_2", "PU_PML.inlet")
    network.connect("JM.port_3", "DM.inlet")

    for state_name, duration, flow_dl, flow_dm in demand_states:
        network.add_scenario(state_name, {"DL": {"flow": flow_dl}, "DM": {"flow": flow_dm}}, weight=duration)
        for junc in CITY_JUNCTIONS.values():
            network.add_constraint(f"min_head:{junc}", network[junc].head >= MIN_CITY_HEAD, scenario=state_name)

    return network

def add_design_constraints(model: pyo.ConcreteModel, pipes: tuple) -> None:
    model.pump_follows_pipe = pyo.ConstraintList()
    for pipe_id in pipes:
        model.pump_follows_pipe.add(model.comp[f"PU_{pipe_id}"].is_installed == model.comp[pipe_id].is_installed)

# --------------------------------------------------------------------------
# 5. Reading out the solution
# --------------------------------------------------------------------------

def _val(v):
    try:
        return pyo.value(v, exception=False)
    except Exception:
        return None

def chosen_diameter_mm(model, pipe_id: str):
    comp = model.comp[pipe_id]
    for k in comp.b:
        val = _val(comp.b[k])
        if val is not None and val > 0.5:
            return int(DEFAULT_PIPE_CATALOG[k].diameter * 1000)
    return 0

PATTERN = re.compile(r"(PU_[A-Za-z]+):\s*reference_head~([\d\.]+),\s*reference_flow~([\d\.]+)")

def parse_suggestions(caught) -> dict[str, tuple[float, float]]:
    out = {}
    for w in caught:
        for pump, h, q in PATTERN.findall(str(w.message)):
            out[pump.replace("PU_", "")] = (float(h.rstrip(".")), float(q.rstrip(".")))
    return out

# --------------------------------------------------------------------------
# 6. Eén PL oplossen met referentie-iteratie
# --------------------------------------------------------------------------

def solve_topology(pl_length: float, refs: dict, states: list[tuple]) -> dict:
    """Itereer de pompreferenties tot ze stabiel zijn. `refs` wordt in-place bijgewerkt (warm start)."""
    res = {"pl_m": pl_length, "cost": np.nan, "iterations": 0,
           "converged": False, "rel_change": np.nan, "note": "",
           **{f"d_{p}": np.nan for p in ALL_PIPES}}
    topology = make_topology(pl_length)

    for it in range(1, MAX_ITERS + 1):
        try:
            net = build_network(states, refs, topology)
            for g in GOALS:
                net.add_goal(**g)
            model = net.build_model(objective=None, financial=FINANCIAL,
                                    head_ceiling=RESERVOIR_HEAD + 2 * PUMP_HEAD_MAX)
            add_design_constraints(model, ALL_PIPES)
            
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                net.solve_goals(model, solver="appsi_highs", tee=False)
            cost = pyo.value(model.total_cost)
        except Exception as exc:                      
            res["note"] = f"{type(exc).__name__}: {str(exc)[:70]}"
            print(f"   it {it}: GEEN OPLOSSING ({res['note']})")
            return res

        diams = {p: chosen_diameter_mm(model, p) for p in ALL_PIPES}
        installed = {p for p in ALL_PIPES if (_val(model.comp[f"PU_{p}"].is_installed) or 0) > 0.5}
        suggestions = parse_suggestions(caught)

        rel = 0.0
        relevant = {p: v for p, v in suggestions.items() if p in installed}
        for p, (h, q) in relevant.items():
            rel = max(rel, abs(h - refs[p]["head"]) / refs[p]["head"],
                      abs(q - refs[p]["flow"]) / refs[p]["flow"])
            refs[p] = {"head": h, "flow": q}

        res.update(cost=cost, iterations=it, rel_change=rel, **{f"d_{p}": diams[p] for p in ALL_PIPES})
        negeer = sorted(set(suggestions) - installed)
        print(f"   it {it}: kosten = € {cost:>10,.0f} | diam = {list(diams.values())} "
              f"| max rel. ref-verandering = {rel:6.2%}"
              + (f" | genegeerd (niet gebouwd): {negeer}" if negeer else ""))

        if not relevant or rel < REF_TOL:
            res["converged"] = True
            break
    return res

# --------------------------------------------------------------------------
# 7. Samenvatting
# --------------------------------------------------------------------------

def summarize(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 80)
    print(f"{'PL (m)':>8} | {'Kosten (€)':>12} | {'Iters':>5} | {'Conv.':>5} | Topologie (PL, PM, PLM, PML)")
    print("-" * 80)
    
    for _, row in df.iterrows():
        pl = row["pl_m"]
        cost = row["cost"]
        fmt_cost = f"{cost:,.0f}" if not pd.isna(cost) else "-"
        iters = row["iterations"]
        conv = "ja" if row["converged"] else "nee"
        topo = f"{row['d_PL']:.0f}, {row['d_PM']:.0f}, {row['d_PLM']:.0f}, {row['d_PML']:.0f}"
        
        print(f"{pl:>8.0f} | {fmt_cost:>12} | {iters:>5} | {conv:>5} | {topo}")
        
    print("=" * 80)
    print(f"\nResultaten opgeslagen in {OUT_CSV.resolve()}")

def main() -> None:
    print("\nStarting free topology optimization for pipe length PL...")
    states = get_demand_states_kmeans(k=K_CLUSTERS)
    
    # Eén gedeelde warm-start dict die evolueert tijdens de sweep
    refs = copy.deepcopy(INITIAL_REFS)   
    
    rows = []
    for pl in PL_LIST:
        print(f"\n--- PL = {pl:.0f} m ---")
        t0 = time.perf_counter()
        res = solve_topology(pl, refs, states)
        status = "geconvergeerd" if res["converged"] else "NIET geconvergeerd"
        print(f"   -> Resultaat: {status} na {res['iterations']} iteraties ({time.perf_counter() - t0:.1f} s)")
        rows.append(res)
        
    summarize(rows)

if __name__ == "__main__":
    main()