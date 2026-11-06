"""Order-to-driver dispatch simulator for an intra-city logistics marketplace.

A 20 x 20 km city, 30-second ticks, a full day. Orders arrive from demand hotspots that shift between morning
and evening. Drivers accept a request with a probability that falls as pickup distance grows; an order cancels if
nobody is matched within 5 minutes.

Dispatch policies compared on the same order stream:
  1. Greedy           - each new order instantly gets the nearest idle driver
  2. Batched optimal  - every 60 s, solve an assignment problem (Hungarian algorithm) minimising total pickup
                        distance across all waiting orders and idle drivers, with a 4 km pickup cap
  3. Batched + reposition - as 2, and idle drivers drift toward cells where the demand forecast (trailing
                        same-hour average from earlier days) exceeds nearby supply
"""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

SIZE, TICK = 20.0, 30          # km, seconds
DAY = 24 * 3600 // TICK
SPEED_KMPH = 18
MAX_WAIT = 5 * 60 // TICK      # ticks before an unmatched order cancels
PICKUP_CAP = 4.0               # km
HOTSPOTS = {"am": [(5, 5), (14, 6), (8, 15)], "pm": [(15, 14), (6, 12), (12, 4)]}


def demand_rate(h):
    """Orders per tick for the whole city by hour (two peaks)."""
    return 0.35 + 1.1 * np.exp(-((h - 10) / 2.2) ** 2) + 1.3 * np.exp(-((h - 18.5) / 2.0) ** 2)


def order_stream(seed, days=1):
    r = np.random.default_rng(seed)
    out = []
    for d in range(days):
        for t in range(DAY):
            h = t * TICK / 3600
            for _ in range(r.poisson(demand_rate(h))):
                spots = HOTSPOTS["am" if h < 14 else "pm"]
                if r.random() < .7:
                    cx, cy = spots[r.integers(len(spots))]
                    x, y = np.clip(r.normal([cx, cy], 1.6), 0, SIZE - 1e-6)
                else:
                    x, y = r.uniform(0, SIZE, 2)
                trip = r.gamma(3, 2.0)          # delivery distance, km
                out.append((d, t, x, y, trip))
    return out


def forecast_grid(history):
    """Same-hour average demand per 2 km cell from earlier days (the 'forecast' used for repositioning)."""
    g = np.zeros((24, 10, 10))
    days = max(1, len({o[0] for o in history}))
    for d, t, x, y, _ in history:
        g[int(t * TICK / 3600), int(y // 2), int(x // 2)] += 1
    return g / days


def simulate(policy, seed=0, n_drivers=80, forecast=None):
    r = np.random.default_rng(seed + 1)
    orders = order_stream(seed + 50)
    by_tick = {}
    for i, (_, t, x, y, trip) in enumerate(orders):
        by_tick.setdefault(t, []).append(i)
    pos = r.uniform(0, SIZE, (n_drivers, 2))
    busy_until = np.zeros(n_drivers, dtype=int)
    waiting = {}                       # order id -> arrival tick
    matched, cancelled, pickup_km, rejects = [], 0, [], 0
    busy_ticks, reposition_km = 0, 0.0
    per_tick_km = SPEED_KMPH * TICK / 3600

    def try_assign(oid, d):
        nonlocal rejects
        o = orders[oid]
        dist = np.abs(pos[d] - (o[2], o[3])).sum()                 # Manhattan distance
        if r.random() > np.exp(-dist / 5.0):                        # driver declines far pickups
            rejects += 1
            return False
        trip_ticks = int((dist + o[4]) / per_tick_km) + 1
        busy_until[d] = t + trip_ticks
        pos[d] = (np.clip(o[2] + r.normal(0, o[4] / 2), 0, SIZE - 1e-6), np.clip(o[3] + r.normal(0, o[4] / 2), 0, SIZE - 1e-6))
        matched.append(oid); pickup_km.append(dist)
        return True

    for t in range(DAY):
        for oid in by_tick.get(t, []):
            waiting[oid] = t
        idle = np.where(busy_until <= t)[0]
        busy_ticks += n_drivers - len(idle)

        if policy == "greedy":
            for oid in sorted(waiting, key=waiting.get):
                if not len(idle):
                    break
                o = orders[oid]
                dist = np.abs(pos[idle] - (o[2], o[3])).sum(1)
                k = int(np.argmin(dist))
                if dist[k] <= PICKUP_CAP and try_assign(oid, idle[k]):
                    del waiting[oid]; idle = np.delete(idle, k)
        elif t % 2 == 0 and waiting and len(idle):                   # batch every 60 s
            oids = list(waiting)
            O = np.array([(orders[i][2], orders[i][3]) for i in oids])
            D = np.abs(pos[idle][:, None, :] - O[None, :, :]).sum(2)  # drivers x orders
            # older orders get a small priority bonus so nobody starves
            age = np.array([t - waiting[i] for i in oids])
            cost = D - 0.15 * age[None, :]
            cost[D > PICKUP_CAP] = 1e6
            ri, ci = linear_sum_assignment(cost)
            for a, b in zip(ri, ci):
                if cost[a, b] < 1e5 and try_assign(oids[b], idle[a]):
                    del waiting[oids[b]]

        for oid in [i for i, a in waiting.items() if t - a >= MAX_WAIT]:
            del waiting[oid]; cancelled += 1

        if policy == "batched+reposition" and forecast is not None and t % 20 == 0:
            idle = np.where(busy_until <= t)[0]
            h = int(t * TICK / 3600)
            nxt = forecast[min(h + 1, 23)]                          # next hour's expected demand per cell
            supply = np.zeros((10, 10))
            for d in idle:
                supply[int(pos[d, 1] // 2), int(pos[d, 0] // 2)] += 1
            gap = nxt - supply * nxt.sum() / max(len(idle), 1)
            targets = np.argwhere(gap > np.quantile(gap, .85))
            if len(targets):
                for d in idle:
                    cy, cx = int(pos[d, 1] // 2), int(pos[d, 0] // 2)
                    if gap[cy, cx] > 0:
                        continue                                    # already in a short cell
                    ty, tx = targets[np.argmin(np.abs(targets - (cy, cx)).sum(1))]
                    goal = np.array([tx * 2 + 1, ty * 2 + 1], float)
                    step = goal - pos[d]
                    n = np.abs(step).sum()
                    if n > 0:
                        move = min(n, per_tick_km * 20 * .5)        # drift for half of the next 10 minutes
                        pos[d] = np.clip(pos[d] + step / n * move, 0, SIZE - 1e-6)
                        reposition_km += move

    total = len(orders)
    pk = np.array(pickup_km)
    return dict(policy=policy, orders=total, fulfilment_pct=round(100 * len(matched) / total, 1),
                cancel_pct=round(100 * cancelled / total, 1),
                mean_pickup_km=round(float(pk.mean()), 2), p90_pickup_km=round(float(np.quantile(pk, .9)), 2),
                mean_pickup_min=round(float(pk.mean() / SPEED_KMPH * 60), 1),
                driver_utilisation_pct=round(100 * busy_ticks / (n_drivers * DAY), 1),
                driver_declines=int(rejects), reposition_km_per_driver=round(float(reposition_km) / n_drivers, 1))


def main(seeds=(0, 1, 2)):
    history = []
    for s in range(3):
        history += [(s,) + o[1:] for o in order_stream(1000 + s)]          # earlier days for the forecast
    fc = forecast_grid(history)
    out = {}
    for pol in ["greedy", "batched", "batched+reposition"]:
        runs = [simulate(pol, s, forecast=fc) for s in seeds]
        out[pol] = {k: (round(float(np.mean([r[k] for r in runs])), 2) if k != "policy" else pol) for k in runs[0]}
    g, b, rp = out["greedy"], out["batched"], out["batched+reposition"]
    out["headline"] = dict(
        fulfilment_greedy_batched_reposition=[g["fulfilment_pct"], b["fulfilment_pct"], rp["fulfilment_pct"]],
        pickup_km_greedy_vs_reposition=[g["mean_pickup_km"], rp["mean_pickup_km"]],
        pickup_cut_pct=round(100 * (1 - rp["mean_pickup_km"] / g["mean_pickup_km"]), 1))
    # fleet sizing: drivers needed for 80% fulfilment under each policy (seed 0)
    sweep = {}
    for pol in ["greedy", "batched+reposition"]:
        sweep[pol] = {n: simulate(pol, 0, n_drivers=n, forecast=fc)["fulfilment_pct"] for n in range(60, 181, 20)}
    need = {p: next((n for n, f in v.items() if f >= 80), None) for p, v in sweep.items()}
    out["fleet_sweep_fulfilment_pct"] = sweep
    out["drivers_needed_for_80pct"] = need
    with open("results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
