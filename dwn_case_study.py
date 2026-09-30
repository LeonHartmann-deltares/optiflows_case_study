"""Expansion planning for a drinking water network: six cities, four sources.

When must new pipelines be built between 2025 and 2050?

Without the four candidate pipes the network is two disconnected islands:

    Area A   C1 C2 C3 C4   fed by S1 (20 Mm3/yr) and S2 (15)
    Area B   C5 C6         fed by S3 (10)
    S4       external (15), reachable only via candidate P12/P13

**Expansion is first required in 2035**, when S1 steps down to 17 Mm3/yr for
Natura 2000. P12 (S4 -> C3) is then built in every later year. The existing
pipes are priced at zero, so cost is the expansion decision alone:

    2025     0.0M   none
    2030     0.0M   none
    2035    69.1M   P12, P14, P15
    2040    99.4M   P12
    2045    99.4M   P12
    2050   138.4M   P12, and one of P14/P15

Caveats:

  - Holding S1 at 20 Mm3/yr still builds in 2035, so growth and the licence
    cut compound; the step-down alone does not explain the date.
  - 2035 still has volumetric and peak-flow headroom, so ``report_balance()``
    puts the shortfall past 2050. What binds instead is not established here.
  - Years are solved independently, so this is not a phasing schedule. P14
    and P15 are interchangeable, making the 2050 choice between them
    arbitrary.

Where the large city sits decides whether the model solves at all; see
``CITY_SIZES``.

The figures above are what ``GOALS`` produces; ``main()`` takes any goal
sequence in its place.

Usage::

    uv run python examples/dwn_case_study.py

The run also writes ``dwn_case_study_2035.json`` beside this file: the
georeferenced network with its goals attached, for a GIS layer or for
``uv run optiflows solve``. Other years via ``export_network(year)``.

From Python, with a different goal ranking::

    from dwn_case_study import main, GOALS
    main(goals=[{**g, "priority": 3 - g["priority"]} for g in GOALS])
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from xml.parsers.expat import model

import pyomo.environ as pyo

from optiflows import goal
from optiflows.components.hydraulics import Demand, Pipe, Source
from optiflows.components.hydraulics.catalog import PipeCatalogEntry
from optiflows.components.hydraulics.junction import Junction
from optiflows.core.results import Results
from optiflows.gis_mapping import route_length_m
from optiflows.network import Network

# --------------------------------------------------------------------------
# Input data
# --------------------------------------------------------------------------


SECONDS_PER_YEAR = 8760.0 * 3600.0
V_MAX = 3.0  # m/s erosion limit
# Released: a meshed network cannot hold every pipe above the 1.1 m/s
# sedimentation default.
V_MIN = 0.0
N_FLOW_BREAKPOINTS = 15
PEAK_FACTOR = 2.0  # mean-to-peak, from an hourly profile
GROWTH_PER_YEAR = 0.00863  # Rijkswaterstaat, 8.63% over 10 years
HORIZON = (2025, 2030, 2035, 2040, 2045, 2050)

# One large city, three average, two small. Which node is large is an
# assumption, and it decides the answer: C1's two feeds cannot carry a large
# city, so that placement is infeasible. C2, C3 and C5 all solve.
CITY_SIZES = {"C1": 5e6, "C2": 5e6, "C3": 15e6, "C4": 1e6, "C5": 5e6, "C6": 1e6}

# S1 steps down in 2035 under Natura 2000 restrictions.
SOURCE_CAPACITY = {"S1": 20e6, "S2": 15e6, "S3": 10e6, "S4": 15e6}
S1_AFTER_2035 = 17e6

# Commercial PVC diameters [m].
CATALOG_DIAMETERS = (0.315, 0.400, 0.500, 0.630, 0.710)

# Synthetic, fitted to TOPOLOGY's lengths (17 DOF against 14 distinct lengths,
# so an exact embedding exists) and chosen for a layout with no crossing pipes.
# Inert for the optimization: ``length=`` stays authoritative.
# See ``report_geometry()``.
COORDINATES = {
    "C1": (52.091413, 5.310010),
    "C2": (52.072960, 5.212031),
    "C3": (52.093116, 5.114945),
    "C4": (51.977863, 5.197453),
    "C5": (52.107035, 4.881801),
    "C6": (52.138276, 4.810180),
    "S1": (52.132080, 5.278776),
    "S2": (52.038819, 5.259668),
    "S3": (52.094099, 4.796522),
    "S4": (52.254339, 5.138614),
}

# (from, to, length [m], fixed diameter [m] or None if candidate)
TOPOLOGY = (
    ("P1", "C1", "S1", 5_000, 0.315),
    ("P2", "C1", "C2", 7_000, 0.400),
    ("P3", "S1", "C2", 8_000, 0.500),
    ("P4", "S1", "C3", 12_000, 0.630),
    ("P5", "C2", "C3", 7_000, 0.710),
    ("P6", "C2", "S2", 5_000, 0.400),
    ("P7", "S2", "C4", 8_000, 0.400),
    ("P8", "C3", "C4", 14_000, 0.500),
    ("P9", "C6", "C5", 6_000, 0.400),
    ("P10", "C6", "S3", 5_000, 0.400),
    ("P11", "C5", "S3", 6_000, 0.500),
    ("P12", "S4", "C3", 18_000, None),
    ("P13", "S4", "C5", 24_000, None),
    ("P14", "C3", "C5", 16_000, None),
    ("P15", "C5", "C3", 16_000, None),
)

# A steel fit, applied to the PVC catalog as well. Candidates only: the
# existing pipes are a sunk cost, priced at zero.
CAPEX_A, CAPEX_B = 28340.0, 3.54

# PVC. Only affects head loss, which is not the binding constraint here.
ROUGHNESS = 1.5e-6  # m

# --------------------------------------------------------------------------
# Data manipulation: the case at a horizon year
# --------------------------------------------------------------------------


def annual_to_flow(volume_m3_per_year: float, peak: bool = True) -> float:
    """Annual volume [m3/yr] as a flow [m3/s], at peak or mean."""
    flow = volume_m3_per_year / SECONDS_PER_YEAR
    return flow * PEAK_FACTOR if peak else flow


def demand_in(year: int, base_volume: float) -> float:
    """Linearly grown annual demand [m3/yr] for a horizon year."""
    return base_volume * (1.0 + GROWTH_PER_YEAR * (year - 2025))


def source_capacity_in(year: int, name: str) -> float:
    """Annual abstraction limit [m3/yr], honouring the 2035 S1 step."""
    if name == "S1" and year >= 2035:
        return S1_AFTER_2035
    return SOURCE_CAPACITY[name]


def pipe_capacity(diameter: float) -> float:
    """Largest flow a pipe passes within the erosion limit [m3/s]."""
    return V_MAX * math.pi / 4.0 * diameter**2


# --------------------------------------------------------------------------
# Model specification: components, goals, solver
# --------------------------------------------------------------------------


def city_port_count(city: str) -> int:
    """Junction ports a city needs: one per incident pipe, plus its own draw."""
    incident = sum(1 for _, a, b, *_rest in TOPOLOGY if city in (a, b))
    return incident + 1


def catalog() -> list[PipeCatalogEntry]:
    """The PVC catalog, priced by the power-law cost relation."""
    return [
        PipeCatalogEntry(diameter=d, roughness=ROUGHNESS, cost_per_m=CAPEX_A * d**CAPEX_B)
        for d in CATALOG_DIAMETERS
    ]


def _geo(name: str, georeference: bool) -> dict[str, float] | None:
    """``geo`` kwarg for a node, or None when georeferencing is off."""
    if not georeference:
        return None
    lat, lon = COORDINATES[name]
    return {"lat": lat, "lon": lon}


# Solved lexicographically, priority 1 first and pinned. The ranking matters:
# the source caps are soft slacks no plain objective prices, so minimizing cost
# alone would leave overdraw free and answer "build nothing".
#
# ``goal`` exposes total_cost, total_energy, total_demand_shortfall and
# total_source_excess as targets, and ``goal.var(comp, var)`` builds one from a
# component variable. All of them serialize through ``to_dict()``; a lambda does not.
GOALS = (
    {
        "name": "no_overdraw",
        "target": goal.total_source_excess,
        "direction": "min",
        "priority": 1,
        "tolerance": 1e-6,  # default tolerance (0.0) does not account for solver precision
    },
    {
        "name": "min_cost",
        "target": goal.total_cost,
        "direction": "min",
        "priority": 2,
    },
    {
        "name": "min_shortfall",
        "target": goal.total_demand_shortfall,
        "direction": "min",
        "priority": 1,
    },
)


def build_network(
    year: int, *, include_candidates: bool = True, georeference: bool = False
) -> Network:
    """The iteration-1 network at a horizon year.

    Sources are soft-capped, so the caller must price the slack as ``GOALS``
    does; otherwise overdraw is free. See ``solve_year``.

    ``georeference=True`` attaches ``geo`` to each node and a ``route`` to each
    connection, for GIS export. Inert for the optimization.
    """
    network = Network()

    for name in ("S1", "S2", "S3", "S4"):
        if name == "S4" and not include_candidates:
            continue

        network.add(
            name,
            Source(
                head=50.0,  # m, arbitrary
                flow_max=annual_to_flow(source_capacity_in(year, name)),
                soft=True,
            ),
            geo=_geo(name, georeference),
        )

    # A Junction plus a Demand, not a bare Demand: pipes route *through* a
    # city, and a Demand has only an inlet.
    for name, base in CITY_SIZES.items():
        network.add(name, Junction(n_ports=city_port_count(name)), geo=_geo(name, georeference))
        network.add(
            f"{name}_draw",
            Demand(flow=annual_to_flow(demand_in(year, base)), soft = True),
            geo=_geo(name, georeference),
        )

    ports: dict[str, int] = dict.fromkeys(CITY_SIZES, 0)

    def next_port(city: str) -> str:
        """Claim the next free Junction port for ``city``."""
        index = ports[city]
        ports[city] = index + 1
        return f"{city}.port_{index}"

    for pipe_id, node_a, node_b, length, diameter in TOPOLOGY:
        # City-to-city links carry flow either way; source feeds do not.
        two_cities = node_a in CITY_SIZES and node_b in CITY_SIZES
        if diameter is None:
            if not include_candidates:
                continue
            pipe = Pipe(
                length=length,
                catalog=catalog(),
                v_min=V_MIN,
                v_max=V_MAX,
                bidirectional=two_cities,
                n_flow_breakpoints=N_FLOW_BREAKPOINTS,
            )
        else:
            # Already built, so its CAPEX is sunk and must not weigh on the
            # expansion decision. Pricing it at zero makes ``total_cost`` the
            # cost of what the model chooses to build.
            entry = [PipeCatalogEntry(diameter=diameter, roughness=ROUGHNESS, cost_per_m=0.0)]
            pipe = Pipe(
                length=length,
                catalog=entry,
                v_min=V_MIN,
                v_max=V_MAX,
                bidirectional=two_cities,
                n_flow_breakpoints=N_FLOW_BREAKPOINTS,
            )
        network.add(pipe_id, pipe, optional=diameter is None)
        # Some pipes are listed city -> source. Flow direction is an outcome,
        # not an input, and a Source declares only an outlet, so orient every
        # connection to leave the Source.
        head_node, tail_node = node_a, node_b
        if head_node not in CITY_SIZES:
            pass
        elif tail_node not in CITY_SIZES:
            head_node, tail_node = tail_node, head_node
        from_port = f"{head_node}.outlet" if head_node not in CITY_SIZES else next_port(head_node)
        to_port = f"{tail_node}.inlet" if tail_node not in CITY_SIZES else next_port(tail_node)
        # Both connections carry the same polyline. Safe because every Pipe
        # has an explicit ``length``, so gis_mapping never derives one from a
        # route and its ambiguity guard never applies.
        head_geo = _geo(head_node, georeference)
        tail_geo = _geo(tail_node, georeference)
        leg = (
            [[head_geo["lat"], head_geo["lon"]], [tail_geo["lat"], tail_geo["lon"]]]
            if georeference
            else None
        )
        network.connect(from_port, f"{pipe_id}.inlet", route=leg)
        network.connect(f"{pipe_id}.outlet", to_port, route=leg)

    for name in CITY_SIZES:
        network.connect(next_port(name), f"{name}_draw.inlet")

    return network


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report_balance() -> None:
    """Per-area volumetric balance across the horizon."""
    print("Volumetric balance [Mm3/yr]: source capacity against demand\n")
    print(
        f"  {'year':<6}{'A dem':>8}{'A sup':>8}{'A slack':>9}{'B dem':>8}{'B sup':>8}{'B slack':>9}"
    )
    area_a = ("C1", "C2", "C3", "C4")
    area_b = ("C5", "C6")
    for year in (2025, 2030, 2035, 2040, 2045, 2050):
        dem_a = sum(demand_in(year, CITY_SIZES[c]) for c in area_a)
        dem_b = sum(demand_in(year, CITY_SIZES[c]) for c in area_b)
        sup_a = source_capacity_in(year, "S1") + source_capacity_in(year, "S2")
        sup_b = source_capacity_in(year, "S3")
        print(
            f"  {year:<6}{dem_a / 1e6:>8.2f}{sup_a / 1e6:>8.1f}"
            f"{(sup_a - dem_a) / 1e6:>+9.2f}"
            f"{dem_b / 1e6:>8.2f}{sup_b / 1e6:>8.1f}"
            f"{(sup_b - dem_b) / 1e6:>+9.2f}"
        )


def report_feeds() -> None:
    """Whether each city's incoming pipes can carry its peak demand."""
    print("\n\nFeed-capacity screen at 2025 peak\n")
    feeds: dict[str, list[tuple[str, float]]] = {c: [] for c in CITY_SIZES}
    for pipe_id, node_a, node_b, _, diameter in TOPOLOGY:
        if diameter is None:
            continue
        for node in (node_a, node_b):
            if node in feeds:
                feeds[node].append((pipe_id, pipe_capacity(diameter)))

    print(f"  {'city':<6}{'peak need':>11}{'feed cap':>10}{'margin':>9}   feeds")
    for city, base in CITY_SIZES.items():
        need = annual_to_flow(demand_in(2025, base))
        total = sum(cap for _, cap in feeds[city])
        flag = "" if total >= need else "   <-- UNSERVABLE"
        names = ",".join(p for p, _ in feeds[city])
        print(f"  {city:<6}{need:>11.4f}{total:>10.4f}{total - need:>+9.4f}   {names}{flag}")


def report_geometry() -> None:
    """Whether COORDINATES reproduces TOPOLOGY's lengths.

    Calls ``gis_mapping.route_length_m``, so the figures are exactly what a
    GIS export would compute from the emitted routes.
    """
    print("\n\nSynthetic coordinates against TOPOLOGY lengths\n")
    print(f"  {'pipe':<6}{'from->to':<12}{'declared':>9}{'geodesic':>11}{'error':>9}")
    worst = 0.0
    seen: set[tuple[str, str]] = set()
    for pipe_id, node_a, node_b, length, _diameter in TOPOLOGY:
        key = (min(node_a, node_b), max(node_a, node_b))
        distance = route_length_m([list(COORDINATES[node_a]), list(COORDINATES[node_b])])
        error = distance - length
        worst = max(worst, abs(error))
        duplicate = "  (coincident with the line above)" if key in seen else ""
        seen.add(key)
        print(
            f"  {pipe_id:<6}{node_a + '->' + node_b:<12}{length:>9,.0f}"
            f"{distance:>11,.1f}{error:>+9.1f}{duplicate}"
        )
    print(f"\n  worst deviation {worst:.1f} m; pipe lengths still come from TOPOLOGY.")


def report_expansion(rows: Sequence[dict]) -> None:
    """Tabulate the expansion decision from solved rows."""
    print("\n\nExpansion decision by horizon year\n")
    print("  Each year is solved independently; this is not a phasing schedule.\n")
    print(f"  {'year':<7}{'status':<11}{'expansion cost':>16}{'overdraw':>22}{'shortfall':>22}   candidates built")
    for row in rows:
        if not row["feasible"]:
            print(f"  {row['year']:<7}{'NO SOLUTION':<11}{'-':>16}{'-':>22}{'-':>22}   {row['reason']}")
            continue
        
        over_detail = ", ".join(f"{n} +{v:.4f}" for n, v in row["overdraw"]) or "none"
        # Zet de shortfalls in een leesbare string
        short_detail = ", ".join(f"{n} -{v:.4f}" for n, v in row.get("shortfall", [])) or "none"
        
        print(
            f"  {row['year']:<7}{row['status']:<11}{row['cost']:>16,.0f}"
            f"{over_detail:>22}{short_detail:>22}   {', '.join(row['built']) or 'none'}"
        )
    print("\n  overdraw names each source drawing above its licensed cap [m3/s].")
    print("  shortfall names each city receiving less than its demand [m3/s].")


# --------------------------------------------------------------------------
# Post-processing: reading a solved model
# --------------------------------------------------------------------------


def source_overdraw(
    network: Network, model: pyo.ConcreteModel, tolerance: float = 1e-6
) -> list[tuple[str, float]]:
    """Each source drawing above its cap, as ``(name, excess [m3/s])``.

    ``total_source_excess`` is weighted and so is not a flow; these are the raw
    slacks, which say *which* licence is broken.
    """
    over = []
    for name in ("S1", "S2", "S3", "S4"):
        variables = network.get_component_variables(model, name)
        slack = variables.get("excess")
        if slack is None:
            continue
        value = pyo.value(next(iter(slack.values())))
        if value > tolerance:
            over.append((name, value))
    return over

def demand_shortfall(
    network: Network, model: pyo.ConcreteModel, tolerance: float = 1e-6
) -> list[tuple[str, float]]:
    """Each demand falling short of its target, as ``(name, shortfall [m3/s])``."""
    shortfalls = []
    for name in CITY_SIZES:
        demand_name = f"{name}_draw"
        variables = network.get_component_variables(model, demand_name)
        slack = variables.get("shortfall")
        if slack is None:
            continue
        # Pak de waarde van de eerste (en enige) tijdsstap uit de variabele
        value = pyo.value(next(iter(slack.values())))
        if value > tolerance:
            shortfalls.append((name, value))
    return shortfalls


def candidates_built(results: Results) -> list[str]:
    """Names and chosen diameters of the optional pipes the solver chose to install."""
    built = []
    for pid, *_rest, base_diameter in TOPOLOGY:
        # We are only interested in the optional pipes, which have no base diameter.
        if base_diameter is None:
            design = results.components.get(pid, {}).get("design", {})
            # Check whether the solver installed it  
            if design.get("is_installed", 0) > 0.5:
                # Extract the chosen diameter  
                gekozen_diameter = design.get("diameter", 0.0)
                built.append(f"{pid} (D={gekozen_diameter:.3f}m)")
    return built


# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------

SOLVER = "appsi_highs"


def solve_year(
    year: int, goals: Sequence[dict] = GOALS
) -> tuple[Results, pyo.ConcreteModel, Network]:
    """Solve one horizon year under *goals*.

    The model is returned because cost must be read from ``model.total_cost``;
    the objective value is only the last priority group's weighted sum.
    """
    network = build_network(year)
    for g in goals:
        network.add_goal(**g)
    model = network.build_model(objective=None)
    return network.solve_goals(model, solver=SOLVER), model, network


def solve_horizon(goals: Sequence[dict] = GOALS, years: Sequence[int] = HORIZON) -> list[dict]:
    """Solve each year and collect what the table needs.

    Returns one row per year; an unsolved year carries ``feasible: False``
    and the solver's ``reason``. Overdrawing a source is not such a year: the
    sources are soft, so it stays feasible and shows in ``overdraw``.
    """
    check_setup(goals)
    rows = []
    for year in years:
        try:
            results, model, network = solve_year(year, goals)
        except RuntimeError as exc:
            # Keep the message: it carries the solver's diagnostic.
            rows.append({"year": year, "feasible": False, "reason": str(exc).splitlines()[0]})
            continue
        rows.append(
            {
                "year": year,
                "feasible": True,
                "status": results.status,
                "cost": pyo.value(model.total_cost),
                "overdraw": source_overdraw(network, model),
                "shortfall": demand_shortfall(network, model),
                "built": candidates_built(results),
            }
        )
    return rows


def check_setup(goals: Sequence[dict]) -> None:
    """Fail fast on a bad goal list or missing solver.

    Called by ``solve_horizon`` before any year is attempted, so that a
    RuntimeError from the loop can only mean an infeasible network.
    """
    if not goals:
        raise SystemExit("No goals given: nothing to optimize.")
    if not pyo.SolverFactory(SOLVER).available():
        raise SystemExit(f"Solver {SOLVER!r} is not available.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def export_network(year: int = 2035, goals: Sequence[dict] = GOALS) -> Path:
    """Write the georeferenced network for *year* as declarative JSON.

    Attaching *goals* is the point of this helper. A network exported without
    them solves on the default objective, where the soft source caps cost
    nothing, and answers a different question without saying so.

    Written beside this file, so the script runs from any directory. Reports
    where it landed and how to solve it, and returns the path.
    """
    network = build_network(year, georeference=True)
    for g in goals:
        network.add_goal(**g)
    path = Path(__file__).with_name(f"dwn_case_study_{year}.json")
    path.write_text(json.dumps(network.to_dict(), indent=2), encoding="utf-8")
    print(f"\n\nWrote {path.name} ({len(goals)} goals), solvable with:")
    print(f"  uv run optiflows solve examples/{path.name}")
    return path


def main(goals: Sequence[dict] = GOALS, years: Sequence[int] = HORIZON) -> None:
    report_balance()
    report_feeds()
    report_geometry()

    rows = solve_horizon(goals, years)
    report_expansion(rows)

    export_network(goals=goals)


if __name__ == "__main__":
    main()
