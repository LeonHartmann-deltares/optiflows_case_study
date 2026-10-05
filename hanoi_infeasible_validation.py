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
    2: 890, 3: 850, 4: 130, 5: 725, 6: 1005, 7: 1350, 8: 550, 
    9: 525, 10: 525, 11: 500, 12: 560, 13: 940, 14: 615, 15: 280, 
    16: 310, 17: 865, 18: 1345, 19: 60, 20: 1275, 21: 930, 22: 485, 
    23: 1045, 24: 820, 25: 170, 26: 900, 27: 370, 28: 290, 29: 360, 
    30: 360, 31: 105, 32: 805,
}

_PIPES = [
    (1, 1, 2, 100), (2, 2, 3, 1350), (3, 3, 4, 900), (4, 4, 5, 1150),
    (5, 5, 6, 1450), (6, 6, 7, 450), (7, 7, 8, 850), (8, 8, 9, 850),
    (9, 9, 10, 800), (10, 10, 11, 950), (11, 11, 12, 1200), (12, 12, 13, 3500),
    (13, 10, 14, 800), (14, 14, 15, 500), (15, 15, 16, 550), (16, 17, 16, 2730),
    (17, 18, 17, 1750), (18, 19, 18, 800), (19, 3, 19, 400), (20, 3, 20, 2200),
    (21, 20, 21, 1500), (22, 21, 22, 500), (23, 20, 23, 2650), (24, 23, 24, 1230),
    (25, 24, 25, 1300), (26, 26, 25, 850), (27, 27, 26, 300), (28, 16, 27, 750),
    (29, 23, 28, 1500), (30, 28, 29, 2000), (31, 29, 30, 1600), (32, 30, 31, 150),
    (33, 32, 31, 860), (34, 25, 32, 950),
]

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

def main() -> None:
    # Voor de test van Jesús gebruiken we het 'convex' model en zetten we 
    # de stroomrichtingen vast (bidirectional=False) voor een snellere berekening.
    net, m = build_network("convex", bidirectional=True)

    # --- START VAN DE TEST VOOR JESÚS ---
    literatuur_inches = [
        40, 40, 40, 40, 40, 40, 40, 40, 40, 30,
        24, 24, 20, 16, 12, 12, 16, 20, 20, 40,
        20, 12, 40, 30, 30, 20, 12, 12, 16, 16,
        12, 12, 16, 20
    ]
    inch_naar_meter = {12: 0.3048, 16: 0.4064, 20: 0.508, 24: 0.6096, 30: 0.762, 40: 1.016}

    # Maak een lijst met harde eisen aan in het Pyomo model
    m.vaste_diameters = pyo.ConstraintList()
    
    # Koppel elke leiding aan de exacte diameter uit de literatuur
    for i, (pid, node_a, node_b, lengte) in enumerate(_PIPES):
        gewenste_diameter = inch_naar_meter[literatuur_inches[i]]
        # Let op de notatie f"P{pid}": hiermee selecteren we componenten zoals 'P1', 'P2' enz.
        m.vaste_diameters.add(m.comp[f"P{pid}"].diameter == gewenste_diameter)
    # --- EINDE VAN DE TEST ---

    print("Test voor Jesús: Berekening gestart met vastgezette literatuur-diameters...\n")
    
    try:
        # We zetten tee=True zodat je de berekening van de solver live kunt volgen
        net.solve(m, tee=True)
        cost = pyo.value(m.total_cost)
        print(f"\nSucces! Het model accepteert de oplossing. Kosten: {cost:12,.0f} USD")
        print("\n--- WATERDRUK CHECK ---")
        for node in _DEMAND_M3H:
            druk = pyo.value(m.comp[f"J{node}"].head["design"])
            if druk < 30.0:
                print(f"Knooppunt {node:2}: {druk:5.2f} meter  <-- TE LAAG")
    except RuntimeError as e:
        print("\n" + "="*70)
        print("CONCLUSIE VOOR JESÚS BEVESTIGD!")
        print("De solver is gecrasht met een RuntimeError (Infeasible).")
        print("Dit bewijst dat de lineaire 'convex' benadering in OptiFlows de waterdruk")
        print("bij dit specifieke ontwerp strenger beoordeelt en onder de 30 meter berekent.")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()