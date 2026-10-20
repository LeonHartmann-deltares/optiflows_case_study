"""Two-city network: K-means cluster sweep using the Convex approach
with the max-approximation pump model to generate data for Figure 10.
"""

from __future__ import annotations

import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pyomo.environ as pyo
from pathlib import Path
import matplotlib.pyplot as plt

from optiflows import goal
from optiflows.components.hydraulics import Demand, Pipe, Source, Pump
from optiflows.components.hydraulics.catalog import DEFAULT_PIPE_CATALOG
from optiflows.components.hydraulics.junction import Junction
from optiflows.core.financial import FinancialParameters
from optiflows.network import Network

# --------------------------------------------------------------------------
# 1. Input data & Configuration 
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

PIPE_LENGTHS = {"PL": 3000.0, "PM": 3000.0, "PLM": 2500.0, "PML": 2500.0}
V_MAX = 3.0  

# Using the Convex approach (n_hyperplanes = 20) with max pump approximation
N_HYPERPLANES = 20

RESERVOIR_HEAD = 0.0  
MIN_CITY_HEAD = 25.0  
PUMP_HEAD_MAX = 50.0  
PUMP_EFFICIENCY = 0.75
PUMP_FIXED_COST = 50_000.0 
ENERGY_COST_PER_WH = 0.12 / 1000  

FINANCIAL = FinancialParameters(
    discount_rate=0.05,
    lifetime_y=30.0,
    operating_hours_y=float(HOURS_PER_YEAR),
)

TOPOLOGY = (
    ("PL", "JS", "JL", PIPE_LENGTHS["PL"], None),
    ("PM", "JS", "JM", PIPE_LENGTHS["PM"], None),
    ("PLM", "JL", "JM", PIPE_LENGTHS["PLM"], None),
    ("PML", "JM", "JL", PIPE_LENGTHS["PML"], None),
)
PIPES = tuple(t[0] for t in TOPOLOGY)

GOALS = (
    {
        "name": "min_cost",
        "target": goal.total_cost,
        "direction": "min",
        "priority": 1,
    },
)

# --------------------------------------------------------------------------
# 2. Data Processing
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

# --------------------------------------------------------------------------
# 3. Test scenarios for the network (including separate peak hour)
# --------------------------------------------------------------------------

def get_demand_states_kmeans(k: int = 12) -> list[tuple]:
    states = []
    # Dedicated separate peak hour scenario
    states.append(("peak", 1.0, SYSTEM_PEAK_FLOWS["DL"], SYSTEM_PEAK_FLOWS["DM"]))
    
    dl_rem = [PROFILES["DL"][i] for i in range(N_HOURS) if i != SYSTEM_PEAK_HOUR]
    dm_rem = [PROFILES["DM"][i] for i in range(N_HOURS) if i != SYSTEM_PEAK_HOUR]
    
    X = np.column_stack((dl_rem, dm_rem))
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    
    for i in range(k):
        cluster_points = X[labels == i]
        duration = float(len(cluster_points))
        if duration > 0:
            flow_dl = float(np.mean(cluster_points[:, 0]))
            flow_dm = float(np.mean(cluster_points[:, 1]))
            states.append((f"km_{i+1:02d}", duration, flow_dl, flow_dm))
            
    return states

# --------------------------------------------------------------------------
# 4. Network specification and model building
# --------------------------------------------------------------------------

def build_network(demand_states: list[tuple], best_refs: dict) -> Network:
    network = Network()
    
    network.add("S1", Source(head=RESERVOIR_HEAD))
    network.add("JS", Junction(n_ports=3))
    network.add("JL", Junction(n_ports=4))
    network.add("JM", Junction(n_ports=4))
    
    for node in DEMANDS:
        network.add(node, Demand(flow=0.0))

    network.connect("S1.outlet", "JS.port_0")

    for pipe_id, _a, _b, length, fixed_diam in TOPOLOGY:
        pump_name = f"PU_{pipe_id}"
        
        # Configure pump using the max approximation model
        pump = Pump(
            head_max=PUMP_HEAD_MAX,
            flow_max=FLOW_MAX[pipe_id],
            efficiency=PUMP_EFFICIENCY,
            energy_cost=ENERGY_COST_PER_WH,
            fixed_cost=PUMP_FIXED_COST,
            power_model="max", 
            reference_head=best_refs[pipe_id]["head"],
            reference_flow=best_refs[pipe_id]["flow"]
        )
        network.add(pump_name, pump, optional=True)
        
        # Convex approach using n_hyperplanes
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


def add_design_constraints(model: pyo.ConcreteModel) -> None:
    model.pump_follows_pipe = pyo.ConstraintList()
    for pipe_id in PIPES:
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

# --------------------------------------------------------------------------
# 6. Cluster Sweep Loop & Plotting for Figure 10
# --------------------------------------------------------------------------

def main() -> None:
    # Cluster range matching your plot (from k=2 to k=50 with step 2, plus k=3)
    clusters_to_test = [1,2,3,4,6,8,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50]
    
    best_refs = {
        "PL": {"head": 30.27, "flow": 0.4756},
        "PM": {"head": 28.49, "flow": 0.1585},
        "PLM": {"head": MIN_CITY_HEAD, "flow": FLOW_MAX["PLM"] / 2.0},
        "PML": {"head": MIN_CITY_HEAD, "flow": FLOW_MAX["PML"] / 2.0},
    }

    results_data = []

    print("\nStarting K-means analysis (Convex Extended McCormick approach)...\n")
    print(f"{'k':<4} | {'Cost (EUR)':<12} | {'Time (s)':<10} | {'Topology (PL, PM, PLM, PML)'}")
    print("-" * 65)

    for k in clusters_to_test:
        states_kmeans = get_demand_states_kmeans(k=k)
        net = build_network(states_kmeans, best_refs)
        for g in GOALS: 
            net.add_goal(**g)
        
        model = net.build_model(objective=None, financial=FINANCIAL, head_ceiling=RESERVOIR_HEAD + 2 * PUMP_HEAD_MAX)
        add_design_constraints(model)
        
        t0 = time.perf_counter()
        results = net.solve_goals(model, solver="appsi_highs", tee=False) 
        t_solve = time.perf_counter() - t0
        
        cost = pyo.value(model.total_cost)
        diams = {p: chosen_diameter_mm(model, p) for p in PIPES}
        topo_str = f"{diams['PL']}, {diams['PM']}, {diams['PLM']}, {diams['PML']}"
        
        print(f"{k:<4} | {cost:<12,.0f} | {t_solve:<10.2f} | {topo_str}")
        print("-" * 65)
        
        results_data.append({
            "k": k,
            "cost": cost,
            "time": t_solve,
            "topology": topo_str
        })

if __name__ == "__main__":
    main()

