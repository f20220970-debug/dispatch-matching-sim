import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import sim


def test_order_stream_has_two_peaks():
    o = sim.order_stream(0)
    hours = [int(t * sim.TICK / 3600) for _, t, *_ in o]
    am, pm, night = sum(h in (9, 10, 11) for h in hours), sum(h in (17, 18, 19) for h in hours), sum(h in (2, 3, 4) for h in hours)
    assert am > 2 * night and pm > 2 * night


def test_every_order_matched_or_cancelled_or_waiting():
    r = sim.simulate("batched", 0)
    assert r["fulfilment_pct"] + r["cancel_pct"] <= 100.1


def test_repositioning_beats_greedy():
    h = []
    for s in range(3):
        h += [(s,) + o[1:] for o in sim.order_stream(1000 + s)]
    fc = sim.forecast_grid(h)
    g = sim.simulate("greedy", 0, forecast=fc)
    rp = sim.simulate("batched+reposition", 0, forecast=fc)
    assert rp["fulfilment_pct"] > g["fulfilment_pct"] and rp["mean_pickup_km"] < g["mean_pickup_km"]
