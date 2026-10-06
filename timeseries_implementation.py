"""Handling a demand time series: a reference series and four approximations.

Adapted from ``timeseries_approximation.py``. Two changes:

1. The demand series is no longer synthetic. It is one city's hourly series
   from the CSV written by ``demand_simpel.py`` (``water_demand_profiles_2025.csv``,
   8760 rows, m3/h). Pick the city with ``CITY``.
2. Every approximation that fits the series (``ldc`` and ``kmeans``) always
   carries the single highest hour of the year as its own 1 h scenario. The
   fit then runs on the other 8759 h. A bin mean understates the peak, and the
   peak is what sizes the pipe, so this keeps the peak exact at any ``k``.

The network is one source, one pump and one candidate pipe feeding one demand.
It is small on purpose: the pipe diameter trades against pump energy, so both
the peak (which sizes the pipe) and the duration (which prices the energy)
change the answer.

Approaches
----------

================= ==================================== ====================
Approach          API                                  Basis
================= ==================================== ====================
``reference``     ``add_scenario(time_steps=)``        ``dt`` in hours
                  + ``add_time()``
``peak_all_year`` one flat scenario at the peak        ``weight`` = 8760 h
``ldc``           ``scenarios_from_timeseries``        ``weight`` = hours
                  ``(method="quantile")`` + 1 h peak
``kmeans``        ``scenarios_from_timeseries``        ``weight`` = hours
                  ``(method="kmeans")`` + 1 h peak
``scenarios``     hand-picked peak/average/low         ``weight`` = hours
================= ==================================== ====================

``method="quantile"`` sorts the series and cuts it into equal-duration bins,
which is a discretized load duration curve, so it serves as the LDC here.
``ldc_6`` therefore means 6 bins plus the 1 h peak, 7 scenarios in total.

Two properties of the library decide how these are written
----------------------------------------------------------

**``weight`` and ``dt`` both multiply the operating cost.** In time mode the
objective forms ``weight[s] * dt[t] * operational_cost`` (``network.py:1185``),
so a time-series scenario carrying both would count its hours twice. The
reference series therefore uses ``dt`` with the default ``weight=1.0``, and
every other approach uses flat overrides with ``weight`` in hours. All five
total 8760 h, without which the operating costs are not comparable.

**Nothing couples consecutive time steps here.** Independent tank inflow and
outflow are not expressible in time mode, so this network has no storage, and
the reference series is equivalent to solving one scenario per step. Time mode
earns its cost only once storage or another inter-step coupling is present.

The reference series is the yardstick the solver can handle, not ground truth.
It condenses 8760 hours into ``REFERENCE_STEPS`` equal-duration steps, carrying
the peak exactly in its first step but holding it for that step's whole
duration, so it slightly overstates the annual mean. A clustered fit's bins
understate the highest values instead, since a bin's representative value is
the mean of its members; the 1 h peak scenario removes that for the top.

Why the whole series is rarely practical
----------------------------------------

The per-(diameter, flow-interval) binaries in the pipe's piecewise head-loss
linearization are replicated per time step. Measured on this network with the
default catalog and ``n_flow_breakpoints=10``:

======  =========  ========  ========
steps   variables  binaries  build
======  =========  ========  ========
24      6,421      2,629     0.2 s
168     44,869     18,325    1.5 s
======  =========  ========  ========

That is 109 binaries per step plus 13 fixed, so a full year would reach about
955,000. Solve time is the harder limit: it grows faster than build time and
varies with the instance, so these figures bound only the model size.

Usage::

    uv run --frozen python examples/timeseries_implementation.py
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyomo.environ as pyo

from optiflows.components.hydraulics import Demand, Pipe, Source
from optiflows.components.hydraulics.catalog import DEFAULT_PIPE_CATALOG
from optiflows.components.hydraulics.pump import Pump
from optiflows.constants import GRAVITY, WATER_DENSITY
from optiflows.core.financial import FinancialParameters
from optiflows.network import Network
from optiflows.preprocessing import scenarios_from_timeseries

# --------------------------------------------------------------------------
# Input data
# --------------------------------------------------------------------------

HOURS_PER_YEAR = 8760
SECONDS_PER_HOUR = 3600.0

# Which column of the CSV feeds the single demand: "Small City" (1 Mm3/yr),
# "Medium City" (5) or "Large City" (15). The network has one pipe and one
# pump, so a bigger city pushes the flow towards the limits of the default
# catalog and the pump; start small.
CITY = "Small City"
DEMAND_YEAR = 2025  # the CSV is the 2025 series; other years are scaled
GROWTH_PER_YEAR = 0.00863  # same linear growth as dwn_case_study.py

# demand_simpel.py saves the CSV in the folder it is run from, so look beside
# this script first and then in the current working directory.
CSV_NAME = "water_demand_profiles_2025.csv"
CSV_PATH = next(
    (p for p in (Path(__file__).with_name(CSV_NAME), Path.cwd() / CSV_NAME) if p.exists()),
    Path(__file__).with_name(CSV_NAME),
)

# Steps in the reference series, which stands in for all 8760; see the
# binary-count table in the module docstring.
REFERENCE_STEPS = 24

PIPE_LENGTH = 3000.0  # m
SOURCE_HEAD = 10.0  # m
PUMP_HEAD_MAX = 40.0  # m
PUMP_EFFICIENCY = 0.75
PUMP_FIXED_COST = 50_000.0  # currency
ENERGY_COST_PER_WH = 0.12 / 1000  # 0.12 currency/kWh

FINANCIAL = FinancialParameters(
    discount_rate=0.05,
    lifetime_y=30.0,
    operating_hours_y=float(HOURS_PER_YEAR),
)

# Hand-picked representative operating points, the classic alternative to
# fitting the series: a short peak block, a long average block, a low block.
# This one keeps its own 500 h peak block; it is the engineer's choice.
SCENARIO_HOURS = {"peak": 500.0, "average": 6000.0, "low": 2260.0}


# --------------------------------------------------------------------------
# The series
# --------------------------------------------------------------------------


def load_profile() -> list[float]:
    """One year of hourly demand [m3/s] for ``CITY``, from the CSV."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Run demand_simpel.py first and copy {CSV_NAME} "
            "next to this script."
        )
    csv = pd.read_csv(CSV_PATH, index_col="tijd")  # m3/h
    growth = 1.0 + GROWTH_PER_YEAR * (DEMAND_YEAR - 2025)
    return (csv[CITY] * growth / SECONDS_PER_HOUR).tolist()


PROFILE = load_profile()
assert len(PROFILE) == HOURS_PER_YEAR, f"expected {HOURS_PER_YEAR} hourly values"
MEAN_FLOW = sum(PROFILE) / len(PROFILE)  # m3/s, annual mean demand
PEAK_FLOW = max(PROFILE)
PEAK_INDEX = PROFILE.index(PEAK_FLOW)  # the hour that gets its own scenario
PROFILE_WITHOUT_PEAK = PROFILE[:PEAK_INDEX] + PROFILE[PEAK_INDEX + 1 :]  # 8759 h


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------


def build_network() -> Network:
    """Source -> pump -> candidate pipe -> demand.

    ``flow_max`` is the series peak on both the pump and the pipe, which keeps
    the pump's power linearization as tight as its bounds allow.
    ``pin_pump_power`` then closes the error those bounds still leave.
    """
    network = Network()
    network.add("S1", Source(head=SOURCE_HEAD))
    network.add(
        "PU",
        Pump(
            head_max=PUMP_HEAD_MAX,
            flow_max=PEAK_FLOW,
            efficiency=PUMP_EFFICIENCY,
            energy_cost=ENERGY_COST_PER_WH,
            fixed_cost=PUMP_FIXED_COST,
        ),
    )
    network.add(
        "P1",
        # v_min released: the whole point is that off-peak steps run slow.
        Pipe(length=PIPE_LENGTH, catalog=DEFAULT_PIPE_CATALOG, flow_max=PEAK_FLOW, v_min=0.0),
    )
    network.add("D1", Demand(flow=MEAN_FLOW))
    network.connect("S1.outlet", "PU.inlet")
    network.connect("PU.outlet", "P1.inlet")
    network.connect("P1.outlet", "D1.inlet")
    return network


# --------------------------------------------------------------------------
# The five approaches
# --------------------------------------------------------------------------


def sampled_series(steps: int = REFERENCE_STEPS) -> list[float]:
    """*steps* values standing in for the year, each representing equal time.

    Sorted by magnitude and cut into equal-duration bins, taking each bin's
    mean, except the first, which takes its maximum so the annual peak is
    always carried exactly. The mean converges from above as *steps* rises.

    Ordering by magnitude rather than by clock matters. A stride through the
    chronological series aliases against the diurnal cycle whenever the stride
    is a whole number of days: at ``steps=365`` every sample would land on the
    same hour. Nothing here couples consecutive steps, so the order carries no
    information and this ordering is the safe one.

    Built from the same ``quantile`` binning the ``ldc`` approaches use, so the
    only difference between this reference and an ``ldc`` fit is the top bin's
    treatment.
    """
    bins = scenarios_from_timeseries(
        {"D1.flow": PROFILE}, n_clusters=steps, method="quantile", dt=1.0
    )
    series = [overrides["D1.flow"] for overrides, _ in reversed(bins)]
    series[0] = PEAK_FLOW  # the peak sizes the pipe, so carry it exactly
    return series


def apply_full(network: Network, steps: int = REFERENCE_STEPS) -> None:
    """The series itself, as one time-series scenario."""
    series = sampled_series(steps)
    network.add_scenario(
        "year",
        time_steps=[{"D1": {"flow": flow}} for flow in series],
        dt=HOURS_PER_YEAR / steps,
    )
    network.add_time()


def apply_peak_all_year(network: Network) -> None:
    """One scenario at the peak, held for the whole year."""
    network.add_scenario("peak", {"D1": {"flow": PEAK_FLOW}}, weight=float(HOURS_PER_YEAR))


def apply_clustered(network: Network, n_clusters: int, method: str) -> None:
    """The 1 h peak scenario plus *n_clusters* duration-weighted scenarios.

    The fit runs on the 8759 other hours, so the weights are the 1 h of the
    peak plus 8759 h of clusters: 8760 h in all. ``scenarios_from_timeseries``
    returns ``(overrides, hours)``, so the weights are hours by construction.
    """
    network.add_scenario("peak", {"D1": {"flow": PEAK_FLOW}}, weight=1.0)
    for i, (overrides, hours) in enumerate(
        scenarios_from_timeseries(
            {"D1.flow": PROFILE_WITHOUT_PEAK}, n_clusters=n_clusters, method=method, dt=1.0
        )
    ):
        network.add_scenario(f"s{i}", {"D1": {"flow": overrides["D1.flow"]}}, weight=hours)


def apply_scenarios(network: Network) -> None:
    """Hand-picked peak/average/low blocks.

    The durations are chosen by the engineer; the flows are read off the series
    at the quantile each duration implies, so the two are consistent.
    """
    ordered = sorted(PROFILE)
    cursor = 0
    # Low first: the blocks are laid against the sorted series from the bottom.
    for name in ("low", "average", "peak"):
        hours = SCENARIO_HOURS[name]
        width = round(hours / HOURS_PER_YEAR * len(ordered))
        block = ordered[cursor : cursor + width]
        cursor += width
        flow = max(block) if name == "peak" else sum(block) / len(block)
        network.add_scenario(name, {"D1": {"flow": flow}}, weight=hours)


def apply_opex_formula(network: Network) -> None:
    # 1. Determine the mean flow
    q_avg = sum(PROFILE) / len(PROFILE)
    
    # 2. Determine the penalty factor F
    F = sum((q / q_avg)**3 for q in PROFILE) / len(PROFILE)
    
    # 3. Determine the effective OPEX flow
    q_eff = q_avg * (F ** (1/3))

    print('')
    print('The factor is: ', F)
    
    # 4. Add the two scenarios to the network
    network.add_scenario("peak", {"D1": {"flow": PEAK_FLOW}}, weight=1.0)
    network.add_scenario("hand_derived", {"D1": {"flow": q_eff}}, weight=HOURS_PER_YEAR - 1.0)

# Configures a freshly built network for one approach.
ApplyApproach = Callable[[Network], None]

APPROACHES: dict[str, ApplyApproach] = {
    "peak_all_year": apply_peak_all_year,
    "scenarios": apply_scenarios,
    "ldc_2": lambda network: apply_clustered(network, 2, "quantile"),
    "ldc_6": lambda network: apply_clustered(network, 6, "quantile"),
    "ldc_12": lambda network: apply_clustered(network, 12, "quantile"),
    "kmeans_2": lambda network: apply_clustered(network, 2, "kmeans"),
    "kmeans_6": lambda network: apply_clustered(network, 6, "kmeans"),
    "kmeans_12": lambda network: apply_clustered(network, 12, "kmeans"),
    "hand_derived": apply_opex_formula,
    "reference": apply_full,
}


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------


def catalog_index(diameter: float) -> int:
    """Catalog position of *diameter*, for fixing the selection binaries."""
    for i, entry in enumerate(DEFAULT_PIPE_CATALOG):
        if math.isclose(entry.diameter, diameter, rel_tol=1e-9):
            return i
    raise ValueError(f"No catalog entry with diameter {diameter}.")


def pin_pump_power(model: pyo.ConcreteModel) -> None:
    """Constrain pump power to the physical requirement at the known flow.

    ``Pump`` linearizes ``power = flow x head_gain`` with a McCormick envelope
    whose lower bound is exact only at ``flow_max``. Every index running below
    it can report less power than it delivers, by roughly half on this network,
    and by different amounts at different diameters, which is what makes the
    designs incomparable without this.

    Demand is fixed here and there is one path, so pump flow is a constant per
    index and ``rho g q h / eta`` is linear in ``head_gain``. Adding it closes
    the envelope exactly. This relies on the open model ``build_model()``
    returns.
    """
    pump, demand = model.comp["PU"], model.comp["D1"]
    model.exact_pump_power = pyo.ConstraintList()
    for idx in pump.power:
        flow = pyo.value(demand.flow[idx])
        model.exact_pump_power.add(
            pump.power[idx]
            >= WATER_DENSITY * GRAVITY * flow * pump.head_gain[idx] / PUMP_EFFICIENCY
        )


def build_pinned(apply: ApplyApproach) -> tuple[Network, pyo.ConcreteModel]:
    """A built, power-pinned model under one approach.

    ``Pump`` warns whenever the network's peak sits below ``flow_max``, which
    it does for every approach whose representative flows are bin means. The
    warning is suppressed because ``pin_pump_power`` answers it, and only that
    warning is suppressed.
    """
    network = build_network()
    apply(network)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*McCormick.*")
        model = network.build_model(financial=FINANCIAL)
    pin_pump_power(model)
    return network, model


def solve(apply: ApplyApproach, solver: str = "appsi_highs") -> dict[str, Any]:
    """Build and solve under one approach."""
    network, model = build_pinned(apply)
    results = network.solve(model, solver=solver)
    return {
        "status": results.status,
        "cost": pyo.value(model.total_cost),
        "diameter": results.components["P1"]["design"]["diameter"],
    }


def audit(diameter: float, solver: str = "appsi_highs") -> dict[str, Any]:
    """Cost of *diameter* when re-solved against the reference series.

    Each approach picks a diameter from its own summary of the series, and so
    reports a cost computed on that summary. This re-solves each choice on one
    common basis, which is what makes the designs comparable.
    """
    network, model = build_pinned(apply_full)
    target = catalog_index(diameter)
    for i in model.comp["P1"].b:
        model.comp["P1"].b[i].fix(1 if i == target else 0)
    try:
        results = network.solve(model, solver=solver)
    except RuntimeError as exc:
        return {"status": "infeasible", "cost": None, "reason": str(exc).splitlines()[0]}
    return {"status": results.status, "cost": pyo.value(model.total_cost)}


def compare(solver: str = "appsi_highs") -> list[dict[str, Any]]:
    """Solve every approach, then re-cost each diameter on the reference series."""
    rows = []
    # Keyed by catalog index, not by the diameter itself: the solver returns it
    # as a sum over the selection binaries, so one catalog entry comes back as
    # several floats a hair apart and a float-keyed memo would miss every time.
    audited: dict[int, dict[str, Any]] = {}
    for name, apply in APPROACHES.items():
        row: dict[str, Any] = {"approach": name}
        row.update(solve(apply, solver=solver))
        diameter = float(row["diameter"])
        entry = catalog_index(diameter)
        if entry not in audited:
            audited[entry] = audit(diameter, solver=solver)
        row["audited"] = audited[entry]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report_profile() -> None:
    print(f"Demand profile ({CITY}, from {CSV_NAME})")
    print(f"  steps          {len(PROFILE)}")
    print(f"  mean           {MEAN_FLOW:.4f} m3/s")
    print(f"  peak           {PEAK_FLOW:.4f} m3/s  (hour {PEAK_INDEX} of the year)")
    print(f"  peak / mean    {PEAK_FLOW / MEAN_FLOW:.3f}")
    sampled = sampled_series()
    print(
        f"  reference      {REFERENCE_STEPS} steps sampled across the year, "
        f"mean {sum(sampled) / len(sampled):.4f}, peak {max(sampled):.4f}"
    )


def report_peak_recovery() -> None:
    """How much of the true peak a fit's top bin keeps WITHOUT the 1 h scenario.

    This is why the 1 h peak scenario exists: without it, the top bin of a fit
    on the full series understates the peak.
    """
    print("\nPeak retained by the top bin of a fit on all 8760 h")
    print(f"  {'k':>4} {'quantile':>10} {'kmeans':>10}   (true peak {PEAK_FLOW:.4f} m3/s)")
    for k in (6, 12, 24):
        tops = []
        for method in ("quantile", "kmeans"):
            fitted = scenarios_from_timeseries(
                {"D1.flow": PROFILE}, n_clusters=k, method=method, dt=1.0
            )
            tops.append(max(o["D1.flow"] for o, _ in fitted))
        print(f"  {k:>4} {tops[0]:>10.4f} {tops[1]:>10.4f}")


from typing import Any

def report(rows: list[dict[str, Any]]) -> None:
    """Design table, then each design re-costed on one basis."""
    
    # 1. Zoek dynamisch de referentiekosten op in de data (voorkomt hardcoding)
    ref_row = next((r for r in rows if r["approach"] == "reference"), None)
    ref_cost = ref_row["cost"] if ref_row else None
    ref_audited_cost = ref_row["audited"]["cost"] if ref_row and "audited" in ref_row else None

    # --- TABEL 1: Originele ontwerpkosten ---
    print("\nDesign under each approach")
    print(f"  {'approach':<14} {'status':<9} {'D [m]':>7} {'cost':>12}  {'vs ref [%]':>10}")
    for row in rows:
        # Bereken percentage vs reference
        if ref_cost and row["cost"] is not None:
            diff_pct = ((row["cost"] - ref_cost) / ref_cost) * 100
            diff_str = f"{diff_pct:>+9.2f}%"
        else:
            diff_str = f"{'—':>10}"
            
        print(
            f"  {row['approach']:<14} {row['status']:<9} {row['diameter']:>7.3f} "
            f"{row['cost']:>12,.0f}  {diff_str}"
        )
    print("  cost is each approach's own estimate, on its own summary of the series")


    # --- TABEL 2: Ge-audite kosten (re-costed) ---
    print("\nEach design re-costed against the reference series")
    print(f"  {'approach':<14} {'D [m]':>7} {'status':<11} {'ref cost':>12}  {'vs best':>9}  {'vs ref [%]':>10}")
    
    costed = [r for r in rows if r["audited"]["cost"] is not None]
    best = min((r["audited"]["cost"] for r in costed), default=None)
    
    for row in rows:
        audited = row["audited"]
        # A design chosen on a coarse fit can be infeasible at the reference
        # series' peak, which is one of the failures this example exists to show.
        if audited["cost"] is None:
            cost, penalty, pct_vs_ref = f"{'—':>12}", f"{'—':>9}", f"{'—':>10}"
        else:
            cost = f"{audited['cost']:>12,.0f}"
            penalty = f"{audited['cost'] - best if best else 0.0:>+9,.0f}"
            
            # Bereken percentage vs reference in de ge-audite resultaten
            if ref_audited_cost:
                pct_val = ((audited['cost'] - ref_audited_cost) / ref_audited_cost) * 100
                pct_vs_ref = f"{pct_val:>+9.2f}%"
            else:
                pct_vs_ref = f"{'—':>10}"
                
        print(
            f"  {row['approach']:<14} {row['diameter']:>7.3f} "
            f"{audited['status']:<11} {cost}  {penalty}  {pct_vs_ref}"
        )


def main(solver: str = "appsi_highs") -> list[dict[str, Any]]:
    report_profile()
    report_peak_recovery()
    rows = compare(solver=solver)
    report(rows)
    return rows


if __name__ == "__main__":
    main()