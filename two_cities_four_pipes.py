"""Two-city networjk with four pipes and optional pumps.
This example is a simplified version of the fully resolved 2-city network example.
The network consists of two cities (DL and DM) connected to a reservoir (S1)
via a junction (JS) and two optional pumps (PU_PL and PU_PM). The two cities are connected to each other via two optional pipes (PLM and PML) and two optional
"""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pyomo.environ as pyo

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


#Reading the csv file that contains the demand profiles for the two cities. 
CSV_NAME = "water_demand_profiles_2025.csv"
CSV_PATH = next(
    (p for p in (Path(__file__).with_name(CSV_NAME), Path.cwd() / CSV_NAME) if p.exists()),
    Path(__file__).with_name(CSV_NAME),
)

#Configuration parameters for the network 
PIPE_LENGTHS = {"PL": 3000.0, "PM": 3000.0, "PLM": 2500.0, "PML": 2500.0}
V_MAX = 3.0  
N_FLOW_BREAKPOINTS = 20
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
# 2. Data and Scenario Processing (here we can add the different approaches that we want to test, like LDC and K-means etc...)
# --------------------------------------------------------------------------

def load_profiles() -> dict[str, list[float]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset {CSV_NAME} niet gevonden.")
    csv = pd.read_csv(CSV_PATH, index_col="tijd")
    return {node: (csv[col] / SECONDS_PER_HOUR).tolist() for node, col in CITY_COLUMNS.items()}

PROFILES = load_profiles()
N_HOURS = len(PROFILES[DEMANDS[0]])

# Determine the total peak flow and the peak hour for the whole system, as well as the peak flows for each city.
TOTAL_PROFILE = [sum(PROFILES[d][h] for d in DEMANDS) for h in range(N_HOURS)]
SYSTEM_PEAK_HOUR = max(range(N_HOURS), key=lambda h: TOTAL_PROFILE[h])
SYSTEM_PEAK_FLOWS = {d: PROFILES[d][SYSTEM_PEAK_HOUR] for d in DEMANDS} #Demand per city in that specific hour
TOTAL_PEAK_FLOW = sum(SYSTEM_PEAK_FLOWS.values()) #Total peak demand in that specific hour for the whole system 
CITY_PEAK = {d: max(PROFILES[d]) for d in DEMANDS} #Maximum demand for each city

# Define the maximum flow for each pipe based on the peak flows of the cities and the total peak flow of the system.
FLOW_MAX = {
    "PL": TOTAL_PEAK_FLOW,
    "PM": TOTAL_PEAK_FLOW,
    "PLM": CITY_PEAK["DM"],
    "PML": CITY_PEAK["DL"],
}

#----------------------------------------------------------------------------
# 3. Test scenarios for the network, we can define different approaches here. For now, we have defined a peak and mean scenario as a placeholder.
#----------------------------------------------------------------------------

#Here we can define the different approaches that we want to test
def get_demand_states() -> tuple:
    """Definieer hier je scenario's. Huidig: 1 piek + 1 mean als placeholder."""
    mean_flows = {d: sum(PROFILES[d]) / N_HOURS for d in DEMANDS}
    return (
        ("peak", 1.0, SYSTEM_PEAK_FLOWS["DL"], SYSTEM_PEAK_FLOWS["DM"]),
        ("mean", float(N_HOURS - 1.0), mean_flows["DL"], mean_flows["DM"]),
    )

DEMAND_STATES = get_demand_states()

# --------------------------------------------------------------------------
# 4. Network specification and model building
# --------------------------------------------------------------------------

def build_network() -> Network:
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
        pump = Pump(
            head_max=PUMP_HEAD_MAX,
            flow_max=FLOW_MAX[pipe_id],
            efficiency=PUMP_EFFICIENCY,
            energy_cost=ENERGY_COST_PER_WH,
            fixed_cost=PUMP_FIXED_COST,
        )
        network.add(pump_name, pump, optional=True)
        
        pipe = Pipe(
            length=length,
            catalog=DEFAULT_PIPE_CATALOG,
            flow_max=FLOW_MAX[pipe_id],
            v_min=0.0,
            v_max=V_MAX,
            n_flow_breakpoints=N_FLOW_BREAKPOINTS,
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

    for state_name, duration, flow_dl, flow_dm in DEMAND_STATES:
        network.add_scenario(state_name, {"DL": {"flow": flow_dl}, "DM": {"flow": flow_dm}}, weight=duration)
        for junc in CITY_JUNCTIONS.values():
            network.add_constraint(f"min_head:{junc}", network[junc].head >= MIN_CITY_HEAD, scenario=state_name)

    return network


# Here we can add additional constraints to the model. For now I have added a constraint that ensures that if a pipe is installed, the corresponding pump is also installed.
def add_design_constraints(model: pyo.ConcreteModel) -> None:
    model.pump_follows_pipe = pyo.ConstraintList()
    for pipe_id in PIPES:
        model.pump_follows_pipe.add(model.comp[f"PU_{pipe_id}"].is_installed == model.comp[pipe_id].is_installed)



# --------------------------------------------------------------------------
# 5. Reading out the solution, the results now present the optimal design, as well as the flow, velocity and head in each pipe for each scenario.
# --------------------------------------------------------------------------

def _val(v):
    """Safe value of a Pyomo Var/Param/Expression (None if not set)."""
    try:
        return pyo.value(v, exception=False)
    except Exception:
        return None


def chosen_diameter_mm(model, pipe_id: str):
    """Diameter (mm) from the catalog selection binaries comp[pipe].b[k].

    The catalog sizes are parsed from the variable names (e.g. 'w:PML:250mm:@peak'),
    sorted ascending; b[k] = 1 means the k-th size is chosen.
    """
    comp = model.comp[pipe_id]
    sizes = sorted({
        int(m.group(1))
        for v in comp.component_data_objects(pyo.Var, descend_into=True)
        for m in [re.search(r":(\d+)mm:", v.name)] if m
    })
    for k in comp.b:
        val = _val(comp.b[k])
        if val is not None and val > 0.5:
            return sizes[k] if k < len(sizes) else f"index {k}"
    return None


def print_design(model) -> None:
    print("\n=== Best Design ===")
    for pipe_id in PIPES:
        comp = model.comp[pipe_id]
        installed = _val(comp.is_installed)
        
        if installed is None or installed < 0.5:
            print(f"{pipe_id:>4}: Not implemented")
        else:
            d_mm = chosen_diameter_mm(model, pipe_id)
            print(f"{pipe_id:>4}: Implemented | diameter: {d_mm} mm")

            for state_name, *_ in DEMAND_STATES:
                q = _val(comp.flow[state_name])
                v = _val(comp.velocity[state_name])
                h_in = _val(comp.head_in[state_name])
                h_out = _val(comp.head_out[state_name])
                print(f"        [{state_name:>4}] Q = {q:.4f} m3/s | v = {v:.2f} m/s | "
                      f"head in = {h_in:.1f} m | head out = {h_out:.1f} m")
        
    
        print()


# --------------------------------------------------------------------------
# 6. Solving the model, and printing the results including the building and solving time.
# --------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()

    network = build_network()
    for g in GOALS:
        network.add_goal(**g)

    t0 = time.perf_counter()
    model = network.build_model(
            objective=None,
            financial=FINANCIAL,
            head_ceiling=RESERVOIR_HEAD + 2 * PUMP_HEAD_MAX,
        )
    add_design_constraints(model)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = network.solve_goals(model, solver="appsi_highs")
    t_solve = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start

    print(f"\nStatus: {results.status} | Total cost: {pyo.value(model.total_cost):,.0f} EUR")
    print(f"Time: building model {t_build:.2f} s | solving {t_solve:.2f} s | total {t_total:.2f} s")
    print_design(model)
if __name__ == "__main__":
    main()