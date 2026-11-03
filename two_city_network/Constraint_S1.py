"""
Onzekerheidsanalyse Scenario 2: Afnemende jaarvergunning voor de hoofdbron (S1).
S1 mag per jaar maximaal een fractie van het totale jaarvolume (grote + middelgrote stad)
leveren. Die fractie daalt in stappen van 10% van 100% naar 0% (dus eerst 20 Mm3/jaar,
dan 18, 16, ... 0).
Een kandidaatbron S2 kan worden aangesloten op de middelgrote stad (3 km) of op de
grote stad (5,5 km). Per stap wordt het goedkoopste net bepaald.
Inclusief iteratie op de pomp-referentiepunten (power_model="max").
"""

from __future__ import annotations

import re
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pyomo.environ as pyo
from pathlib import Path

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

PIPE_LENGTHS = {"PL": 3000.0, "PM": 3000.0, "PLM": 2500.0, "PML": 2500.0,
                "PS2M": 3000.0, "PS2L": 5500.0}
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

TOPOLOGY = (
    ("PL", "JS", "JL", PIPE_LENGTHS["PL"], None),
    ("PM", "JS", "JM", PIPE_LENGTHS["PM"], None),
    ("PLM", "JL", "JM", PIPE_LENGTHS["PLM"], None),
    ("PML", "JM", "JL", PIPE_LENGTHS["PML"], None),
    ("PS2M", "JS2", "JM", PIPE_LENGTHS["PS2M"], None),   # S2 -> middelgrote stad (3 km)
    ("PS2L", "JS2", "JL", PIPE_LENGTHS["PS2L"], None),   # S2 -> grote stad (5,5 km)
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

# --- Sweep-instellingen ---
S1_START = 1.0              # begin bij 100% van het totale jaarvolume (= 20 Mm3)
S1_STEP = 0.1               # elke stap 10 procentpunt (= 2 Mm3) omlaag
S1_FRACTIES = [round(S1_START - S1_STEP * i, 2) for i in range(int(round(S1_START / S1_STEP)) + 1)]
# -> 100%, 90%, ..., 0%  (11 cases)
K_CLUSTERS = 8
MAX_ITERS = 25
ONE_S2_CONNECTION = False   # True = S2 mag aan maximaal EEN stad worden gekoppeld
VERBOSE = False             # True = print elke druk-correctie van de iteratie

# --- Stopcriterium van de iteratie ---
# De iteratie stopt als OptiFlows geen referentiepunt meer voorstelt, OF als de grootste
# aanpassing in head kleiner is dan TOL_H (m) en in flow kleiner dan TOL_Q (relatief).
# Zet TOL_H = 0 om alleen op "geen waarschuwingen meer" te stoppen (strengste variant).
TOL_H = 0.3
TOL_Q = 0.05

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

# Jaarvolume van beide steden samen (m3/jaar): dit is 100% van de S1-limiet
ANNUAL_DEMAND_M3 = sum(TOTAL_PROFILE) * SECONDS_PER_HOUR

FLOW_MAX = {
    "PL": TOTAL_PEAK_FLOW,
    "PM": TOTAL_PEAK_FLOW,
    "PLM": CITY_PEAK["DM"],
    "PML": CITY_PEAK["DL"],
    "PS2M": TOTAL_PEAK_FLOW,
    "PS2L": TOTAL_PEAK_FLOW,
}

# --------------------------------------------------------------------------
# 3. K-means Clustering
# --------------------------------------------------------------------------

def get_demand_states_kmeans(k: int = 12) -> list[tuple]:
    states = []
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

def state_weights(states: list[tuple]) -> tuple[dict[str, float], float]:
    """Gewicht per toestand (aandeel van het jaar, som = 1) en gemiddelde totale vraag (m3/s).

    Clustergemiddelde x aantal uren = som van de uurwaarden, dus het gewogen gemiddelde
    over de toestanden is exact het jaargemiddelde: k-means benadert het jaarvolume niet.
    """
    total_hours = sum(duur for _, duur, _, _ in states)
    weights = {name: duur / total_hours for name, duur, _, _ in states}
    mean_demand = sum(weights[name] * (dl + dm) for name, _, dl, dm in states)
    return weights, mean_demand

# --------------------------------------------------------------------------
# 4. Network specification and model building
# --------------------------------------------------------------------------

def build_network(demand_states: list[tuple], best_refs: dict) -> Network:
    network = Network()

    network.add("S1", Source(head=RESERVOIR_HEAD))
    network.add("JS", Junction(n_ports=3))
    network.connect("S1.outlet", "JS.port_0")

    network.add("S2", Source(head=RESERVOIR_HEAD))
    network.add("JS2", Junction(n_ports=3))      # 1 in (S2), 2 uit (naar DM en DL)
    network.connect("S2.outlet", "JS2.port_0")

    network.add("JL", Junction(n_ports=5))       # PL, PML, PS2L in; PLM en DL uit
    network.add("JM", Junction(n_ports=5))       # PM, PLM, PS2M in; PML en DM uit

    for node in DEMANDS:
        network.add(node, Demand(flow=0.0))

    for pipe_id, _a, _b, length, fixed_diam in TOPOLOGY:
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

    # Bron 1
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

    # Bron 2: naar de middelgrote stad (3 km) en/of de grote stad (5,5 km)
    network.connect("JS2.port_1", "PU_PS2M.inlet")
    network.connect("PS2M.outlet", "JM.port_4")
    network.connect("JS2.port_2", "PU_PS2L.inlet")
    network.connect("PS2L.outlet", "JL.port_4")

    for state_name, duration, flow_dl, flow_dm in demand_states:
        network.add_scenario(state_name, {"DL": {"flow": flow_dl}, "DM": {"flow": flow_dm}}, weight=duration)
        for junc in CITY_JUNCTIONS.values():
            network.add_constraint(f"min_head:{junc}", network[junc].head >= MIN_CITY_HEAD, scenario=state_name)

    return network

def add_design_constraints(model: pyo.ConcreteModel, states: list[tuple], s1_fraction: float) -> None:
    # Pomp en leiding horen bij elkaar
    model.pump_follows_pipe = pyo.ConstraintList()
    for pipe_id in PIPES:
        model.pump_follows_pipe.add(model.comp[f"PU_{pipe_id}"].is_installed == model.comp[pipe_id].is_installed)

    # Optioneel: S2 aan maximaal een stad koppelen
    if ONE_S2_CONNECTION:
        model.s2_one_connection = pyo.Constraint(
            expr=model.comp["PS2M"].is_installed + model.comp["PS2L"].is_installed <= 1
        )

    # Jaarlimiet op S1: een som over ALLE scenario's, daarom direct in Pyomo.
    # Gewogen gemiddeld S1-debiet <= fractie x gemiddelde totale vraag
    # (identiek aan: S1-jaarvolume <= fractie x jaarvolume van beide steden).
    weights, mean_demand = state_weights(states)
    try:
        s1_flow = model.comp["S1"].flow
        terms = [weights[s] * s1_flow[s] for s in weights]
    except Exception as exc:
        raise RuntimeError(
            "Kon model.comp['S1'].flow[<scenario>] niet benaderen. Bekijk met "
            "list(model.comp['S1'].component_map(pyo.Var).keys()) hoe de debietvariabele van "
            "de bron heet en hoe hij is geindexeerd."
        ) from exc
    model.s1_annual_limit = pyo.Constraint(expr=sum(terms) <= s1_fraction * mean_demand)

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
# 5. Een case oplossen (met automatische iteratie op de referentiepunten)
# --------------------------------------------------------------------------

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
REF_PATTERN = re.compile(rf"(PU_\w+):\s*reference_head~({NUM}),\s*reference_flow~({NUM})")


class SolveFailed(Exception):
    """De solver gaf een fout (los van infeasible)."""


def build_and_solve(states: list[tuple], best_refs: dict, s1_fraction: float):
    """Bouw het model, los het op en geef (model, warnings, t_build, t_solve) terug."""
    t0 = time.perf_counter()
    net = build_network(states, best_refs)
    for g in GOALS:
        net.add_goal(**g)
    model = net.build_model(objective=None, financial=FINANCIAL,
                            head_ceiling=RESERVOIR_HEAD + 2 * PUMP_HEAD_MAX)
    add_design_constraints(model, states, s1_fraction)
    t_build = time.perf_counter() - t0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        try:
            net.solve_goals(model, solver="appsi_highs", tee=False)
        except Exception as exc:
            raise SolveFailed(str(exc)) from exc
        t_solve = time.perf_counter() - t0

    return model, list(caught), t_build, t_solve


def update_references(caught: list, best_refs: dict) -> tuple[bool, float, float]:
    """Verwerk de OptiFlows-waarschuwingen en werk best_refs bij (met demping).

    Geeft (needs_update, max_dh in m, max_rel_dq) terug.
    """
    needs_update = False
    max_dh, max_rel_dq = 0.0, 0.0

    for warning in caught:
        msg = str(warning.message)
        if "Set PU_" not in msg or "reference_head~" not in msg:
            continue
        for pump_name, new_h, new_q in REF_PATTERN.findall(msg):
            pipe_id = pump_name.replace("PU_", "")

            oud_h = best_refs[pipe_id]["head"]
            oud_q = best_refs[pipe_id]["flow"]
            # Demping: 50% van de oude en 50% van de nieuwe waarde
            nieuw_h = (0.5 * oud_h) + (0.5 * float(new_h))
            nieuw_q = float(new_q)

            max_dh = max(max_dh, abs(nieuw_h - oud_h))
            max_rel_dq = max(max_rel_dq, abs(nieuw_q - oud_q) / max(oud_q, 1e-6))

            if VERBOSE and abs(nieuw_h - oud_h) > 0.001:
                print(f"      -> {pipe_id} druk-correctie: {abs(nieuw_h - oud_h):.4f} m "
                      f"(van {oud_h:.2f} naar {nieuw_h:.2f} m)")

            best_refs[pipe_id]["head"] = nieuw_h
            best_refs[pipe_id]["flow"] = nieuw_q
            needs_update = True

    return needs_update, max_dh, max_rel_dq


def source_share(model, source: str, states: list[tuple]) -> float:
    """Aandeel van de totale jaarvraag dat door een bron wordt geleverd."""
    weights, mean_demand = state_weights(states)
    try:
        flow = model.comp[source].flow
        return sum(weights[s] * pyo.value(flow[s]) for s in weights) / mean_demand
    except Exception:
        return float("nan")


def collect_result(model, states: list[tuple], status: str, stop_reason: str,
                   n_iter: int, t_build: float, t_solve: float) -> dict:
    """Haal kosten, diameters en bronaandelen uit de laatste oplossing."""
    try:
        cost = pyo.value(model.total_cost)
    except Exception:
        return {"status": "geen oplossing (infeasible?)", "iteraties": n_iter}

    result = {
        "status": status,
        "stopcriterium": stop_reason,
        "iteraties": n_iter,
        "kosten_eur": cost,
        "tijd_bouwen_s": round(t_build, 1),
        "tijd_oplossen_s": round(t_solve, 1),
    }
    for p in PIPES:
        result[f"{p}_mm"] = chosen_diameter_mm(model, p)

    # Hoeveel van de vraag komt uit S1 / S2? (S1 is begrensd door de jaarlimiet)
    result["S1_aandeel_vraag"] = source_share(model, "S1", states)
    result["S2_aandeel_vraag"] = source_share(model, "S2", states)
    return result


def solve_case(states: list[tuple], best_refs: dict, s1_fraction: float, case_tag: str) -> dict:
    """Los een case op. best_refs wordt bijgewerkt en dient als warme start voor de volgende case."""
    converged = False
    stop_reason = "max iteraties"
    t_build = t_solve = 0.0
    model = None
    iteration = 0

    for iteration in range(1, MAX_ITERS + 1):
        try:
            model, caught, tb, ts = build_and_solve(states, best_refs, s1_fraction)
        except SolveFailed as exc:
            print(f"{case_tag} iter {iteration:>2}/{MAX_ITERS}  SOLVE-FOUT: {exc}", flush=True)
            return {"status": f"solve-fout: {exc}", "iteraties": iteration}
        t_build += tb
        t_solve += ts

        needs_update, max_dh, max_rel_dq = update_references(caught, best_refs)

        if needs_update:
            print(f"{case_tag} iter {iteration:>2}/{MAX_ITERS}  "
                  f"max dH={max_dh:6.2f} m  max dQ={max_rel_dq:6.1%}  ({tb + ts:.1f}s)", flush=True)
        else:
            print(f"{case_tag} iter {iteration:>2}/{MAX_ITERS}  "
                  f"geen referentie-aanpassingen meer  ({tb + ts:.1f}s)", flush=True)

        if not needs_update:
            converged, stop_reason = True, "geen waarschuwingen"
            break
        if max_dh < TOL_H and max_rel_dq < TOL_Q:
            converged, stop_reason = True, "binnen tolerantie"
            break

    status = "ok" if converged else "niet geconvergeerd"
    return collect_result(model, states, status, stop_reason, iteration, t_build, t_solve)

# --------------------------------------------------------------------------
# 6. Sweep over de S1-jaarlimiet
# --------------------------------------------------------------------------

def initial_references() -> dict:
    """Startreferenties; de heads van PL, PML en PS2M zijn afgeleid van eerdere runs
    (dichter bij het werkpunt = minder iteraties nodig)."""
    return {
        "PL": {"head": 33.0, "flow": 0.4756},
        "PM": {"head": 28.49, "flow": 0.1585},
        "PLM": {"head": MIN_CITY_HEAD, "flow": FLOW_MAX["PLM"] / 2.0},
        "PML": {"head": 5.5, "flow": FLOW_MAX["PML"] / 2.0},
        "PS2M": {"head": 27.3, "flow": FLOW_MAX["PS2M"] / 2.0},
        "PS2L": {"head": MIN_CITY_HEAD, "flow": FLOW_MAX["PS2L"] / 2.0},
    }


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}u{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def print_case_summary(res: dict) -> None:
    cost = res.get("kosten_eur")
    cost_txt = f"EUR {cost:,.0f}" if cost is not None else "-"
    topo = ", ".join(f"{p}:{res.get(f'{p}_mm', '-')}" for p in PIPES)
    t_tot = res.get("tijd_bouwen_s", 0) + res.get("tijd_oplossen_s", 0)
    print(f"   => {res['status']} | iters={res['iteraties']} | {t_tot:.1f}s | "
          f"kosten={cost_txt}\n   => diameters (mm): {topo}\n", flush=True)


def main() -> None:
    n_cases = len(S1_FRACTIES)
    print(f"\nStarting S1 jaarlimiet-analyse: {n_cases} cases, "
          f"100% = {ANNUAL_DEMAND_M3 / 1e6:.2f} miljoen m3/jaar")
    print(f"Fracties: {', '.join(f'{f:.0%}' for f in S1_FRACTIES)}\n", flush=True)

    best_refs = initial_references()
    states_kmeans = get_demand_states_kmeans(k=K_CLUSTERS)

    rows = []
    t_start = time.perf_counter()

    for i, fraction in enumerate(S1_FRACTIES, start=1):
        limit_mm3 = fraction * ANNUAL_DEMAND_M3 / 1e6
        case_tag = f"[case {i}/{n_cases} | S1 max {fraction:>4.0%} = {limit_mm3:5.2f} Mm3]"
        print(f"{'-' * 78}\n{case_tag}", flush=True)

        res = solve_case(states_kmeans, best_refs, fraction, case_tag)   # best_refs = warme start
        rows.append({"S1_limiet": fraction, "S1_limiet_Mm3": limit_mm3, **res})
        print_case_summary(res)

        elapsed = time.perf_counter() - t_start
        eta = elapsed / i * (n_cases - i)
        print(f"   voortgang: {i}/{n_cases} cases klaar | verstreken {format_duration(elapsed)} | "
              f"nog ca. {format_duration(eta)}\n", flush=True)

    df = pd.DataFrame(rows)
    out_csv = Path(__file__).with_name("sweep_s1_jaarlimiet.csv")
    df.to_csv(out_csv, index=False)

    cols = ["S1_limiet_Mm3", "status", "iteraties", "kosten_eur", "S1_aandeel_vraag",
            "S2_aandeel_vraag", *[f"{p}_mm" for p in PIPES]]
    cols = [c for c in cols if c in df.columns]
    print("=" * 78)
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nKlaar in {format_duration(time.perf_counter() - t_start)}. "
          f"Resultaten opgeslagen in {out_csv}")

if __name__ == "__main__":
    main()
