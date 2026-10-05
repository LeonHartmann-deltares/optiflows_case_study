"""Hanoi three-loop water distribution network - pipe sizing benchmark.

Reference
---------
Fujiwara, O. and Khang, D.B. (1990). A two-phase decomposition method for optimal
design of looped water distribution networks. Water Resources Research, 26(4),
539-549. The commonly reported optimum with the six-diameter catalog below is about
USD 6.08 million (6.081e6 in most of the literature).

Network data (reservoir head 100 m, 31 demand nodes, 34 pipes, Hazen-Williams C=130,
all node elevations 0 m) is taken from the OpenWaterAnalytics epanet-example-networks
file ``epanet-tests/exeter/hanoi-3.inp``. The EPANET file carries a 0.75 demand
multiplier and a placeholder 800 mm diameter; neither is used here. The catalog
diameters and unit costs (1.1 * D[in]**1.5 USD/m) are the standard Hanoi values and
are not in that file.

Model
-----
Pipe flow directions are fixed to the EPANET orientation by default;
``bidirectional=True`` lets the solver choose them for every pipe except the reservoir
pipe. The minimum head is 30 m at every demand node. Cost is the pipe capital cost only.
A gap to the published value has three sources that can only raise the cost: the
fixed directions (default), the velocity cap below, and the chord overestimate of head
loss between breakpoints. A fourth, the package's Hazen-Williams coefficient 10.67
versus the one behind the published optimum, has unknown direction. The published
problem has no velocity limit; every pipe is capped at 7 m/s, the smallest round value
that lets the reservoir pipe carry the total demand in the 40 inch entry (6.83 m/s). The
cap bounds each catalog entry's flow range to v_max * area. Piecewise uses 10
breakpoints per flow direction (9 intervals) and convex uses 9 hyperplanes, the
like-for-like pair.

Fixed-diameter test
-------------------
``main()`` fixes every pipe to the diameter of the published literature design
(``LITERATURE_INCHES``) before solving, so the model only has to decide the flows and
heads. If the model is infeasible, the published design violates the 30 m minimum head
under that head-loss model.

Units: head [m], flow [m3/s], diameter [m], cost [USD].
"""

from __future__ import annotations

import pyomo.environ as pyo

from optiflows.components.hydraulics.catalog import PipeCatalogEntry
from optiflows.components.hydraulics.junction import Junction
from optiflows.components.hydraulics.pipe import Pipe
from optiflows.components.hydraulics.source_demand import Demand, Source
from optiflows.network import Network

PUBLISHED_OPTIMUM = 6.081e6
RESERVOIR_HEAD = 100.0
MIN_HEAD = 30.0
_HW_C = 131.1

# (diameter [m], cost [USD/m]): 12, 16, 20, 24, 30 and 40 inch pipes.
HANOI_CATALOG = [
    PipeCatalogEntry(diameter=d, roughness=0.0, cost_per_m=c, hw_c=_HW_C)
    for d, c in [
        (0.3048, 45.726),
        (0.4064, 70.40),
        (0.508, 98.387),
        (0.6096, 129.333),
        (0.762, 180.748),
        (1.016, 278.28),
    ]
]

# Node demands [m3/h] by node id; node 1 is the reservoir.
_DEMAND_M3H = {
    2: 890,
    3: 850,
    4: 130,
    5: 725,
    6: 1005,
    7: 1350,
    8: 550,
    9: 525,
    10: 525,
    11: 500,
    12: 560,
    13: 940,
    14: 615,
    15: 280,
    16: 310,
    17: 865,
    18: 1345,
    19: 60,
    20: 1275,
    21: 930,
    22: 485,
    23: 1045,
    24: 820,
    25: 170,
    26: 900,
    27: 370,
    28: 290,
    29: 360,
    30: 360,
    31: 105,
    32: 805,
}

_PIPES = [
    (1, 1, 2, 100),
    (2, 2, 3, 1350),
    (3, 3, 4, 900),
    (4, 4, 5, 1150),
    (5, 5, 6, 1450),
    (6, 6, 7, 450),
    (7, 7, 8, 850),
    (8, 8, 9, 850),
    (9, 9, 10, 800),
    (10, 10, 11, 950),
    (11, 11, 12, 1200),
    (12, 12, 13, 3500),
    (13, 10, 14, 800),
    (14, 14, 15, 500),
    (15, 15, 16, 550),
    (16, 17, 16, 2730),
    (17, 18, 17, 1750),
    (18, 19, 18, 800),
    (19, 3, 19, 400),
    (20, 3, 20, 2200),
    (21, 20, 21, 1500),
    (22, 21, 22, 500),
    (23, 20, 23, 2650),
    (24, 23, 24, 1230),
    (25, 24, 25, 1300),
    (26, 26, 25, 850),
    (27, 27, 26, 300),
    (28, 16, 27, 750),
    (29, 23, 28, 1500),
    (30, 28, 29, 2000),
    (31, 29, 30, 1600),
    (32, 30, 31, 150),
    (33, 32, 31, 860),
    (34, 25, 32, 950),
]

# Diameters [inch] of the published literature design, in the order of _PIPES.
LITERATURE_INCHES = [
    40, 40, 40, 40, 40, 40, 40, 40, 40, 30,
    24, 24, 20, 16, 12, 12, 16, 20, 20, 40,
    20, 12, 40, 30, 30, 20, 12, 12, 16, 16,
    12, 12, 16, 20,
]
_INCH_TO_M = {12: 0.3048, 16: 0.4064, 20: 0.508, 24: 0.6096, 30: 0.762, 40: 1.016}


_V_MAX = 7.0  # m/s; the smallest round value above total demand / 40 inch area (6.83)


_MODELS = {
    "piecewise": {"head_loss_model": "piecewise", "n_flow_breakpoints": 10},
    "convex": {"head_loss_model": "convex", "n_hyperplanes": 9},
}


def build_network(
    model: str = "convex", bidirectional: bool = True
) -> tuple[Network, pyo.ConcreteModel]:
    """Build the Hanoi network and its Pyomo model.

    ``model`` is ``"piecewise"`` or ``"convex"``, the pipe head-loss model.
    """
    demand = {n: q / 3600.0 for n, q in _DEMAND_M3H.items()}
    pipe_kw = dict(
        catalog=HANOI_CATALOG,
        flow_max=sum(demand.values()),
        v_min=0.0,
        v_max=_V_MAX,
        **_MODELS[model],
    )
    net = Network()
    net.add("R", Source(head=RESERVOIR_HEAD))
    ports = dict.fromkeys(demand, 1)  # next free junction port; port_0 is the demand
    for node, q in demand.items():
        n_pipes = sum(node in (a, b) for _, a, b, _ in _PIPES)
        net.add(f"J{node}", Junction(n_ports=n_pipes + 1))
        net.add(f"D{node}", Demand(flow=q))
        net.connect(f"J{node}.port_0", f"D{node}.inlet")
    for pid, a, b, length in _PIPES:
        net.add(
            f"P{pid}",
            Pipe(
                length=length,
                bidirectional=bidirectional and a != 1,
                **pipe_kw,
            ),
        )
        if a == 1:
            net.connect("R.outlet", f"P{pid}.inlet")
        else:
            net.connect(f"J{a}.port_{ports[a]}", f"P{pid}.inlet")
            ports[a] += 1
        net.connect(f"P{pid}.outlet", f"J{b}.port_{ports[b]}")
        ports[b] += 1
    net.add_scenario("design", {})
    m = net.build_model(objective="min_total_cost")
    for node in demand:
        m.add_component(
            f"min_head_{node}",
            pyo.Constraint(expr=m.comp[f"J{node}"].head["design"] >= MIN_HEAD),
        )
    return net, m


def fix_literature_diameters(m: pyo.ConcreteModel) -> None:
    """Force every pipe to the diameter of the published literature design."""
    m.fixed_diameters = pyo.ConstraintList()
    for (pid, *_rest), inches in zip(_PIPES, LITERATURE_INCHES):
        m.fixed_diameters.add(m.comp[f"P{pid}"].diameter == _INCH_TO_M[inches])


def main() -> None:
    for model in _MODELS:
        net, m = build_network(model)
        fix_literature_diameters(m)
        try:
            net.solve(m, tee=False)
        except RuntimeError:
            print(f"{model:10s} infeasible with the literature diameters fixed")
            continue
        cost = pyo.value(m.total_cost)
        print(
            f"{model:10s} cost {cost:12,.0f} USD   "
            f"vs published {PUBLISHED_OPTIMUM:,.0f}: {cost / PUBLISHED_OPTIMUM - 1:+.2%}"
        )


if __name__ == "__main__":
    main()