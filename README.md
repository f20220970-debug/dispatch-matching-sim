# Dispatch & Matching Simulator

[![tests](https://github.com/f20220970-debug/dispatch-matching-sim/actions/workflows/tests.yml/badge.svg)](https://github.com/f20220970-debug/dispatch-matching-sim/actions/workflows/tests.yml)

How should a logistics marketplace match orders to driver-partners, and where should idle drivers wait? This is a full-day simulation of a 20 × 20 km city that compares three dispatch policies on the **same** order stream.

> **Simulated city:** about 2,050 orders a day with morning and evening peaks, demand hotspots that shift between them, and drivers who are more likely to decline far pickups. Orders cancel after 5 minutes unmatched. Results are the average of 3 seeds. No company data is used.

Run it: `pip install -r requirements.txt && python sim.py && pytest -q` (about 3 seconds).

## Policies

1. **Greedy:** each new order immediately gets the nearest idle driver.
2. **Batched optimal:** every 60 seconds, solve an assignment problem across all waiting orders and idle drivers (Hungarian algorithm, `scipy.optimize.linear_sum_assignment`) to minimise total pickup distance, with a 4 km pickup cap and a priority bonus for older orders.
3. **Batched + repositioning:** as in 2, plus idle drivers drift toward cells where **next hour's forecast demand** exceeds nearby supply. The forecast is the same-hour average from 3 earlier days.

## Results (80 drivers, mean of 3 seeds)

| Policy | Fulfilment | Cancelled | Mean pickup | Driver utilisation | Empty km to reposition per driver |
| --- | --- | --- | --- | --- | --- |
| Greedy | 63.6% | 36.3% | 2.37 km (7.9 min) | 31.2% | 0 |
| Batched optimal | 65.9% | 34.0% | 2.27 km | 32.1% | 0 |
| **Batched + repositioning** | **83.5%** | **16.4%** | **1.91 km (6.4 min)** | **38.8%** | 114 km |

**Fleet sizing (seed 0):** reaching **80% fulfilment needs 160 drivers with greedy dispatch but only 80 with batching + repositioning.**

## What this shows

- **Batching on its own helps only a little (+2.3 pts).** The real problem is that drivers sit where the last trip dropped them, not where the next order will come from.
- **Forecast-led repositioning does most of the work.** It adds 17.6 pts of fulfilment, cuts pickup distance by 19%, and halves the fleet needed for the same service level.
- **Repositioning has a cost.** It adds about 114 empty km per driver per day. In practice, pay a repositioning incentive only where the forecast gap is large, and test it zone by zone (switchback), because drivers will move whether or not demand shows up.

## Files

```
sim.py        city, order stream, forecast, three policies, fleet sweep
results.json  per-policy metrics + fleet sweep
tests/        demand has two peaks; outcomes add up; repositioning beats greedy
```

Companion repos: [porter-category-growth-lab](https://github.com/f20220970-debug/porter-category-growth-lab) and [dynamic-pricing-bandits](https://github.com/f20220970-debug/dynamic-pricing-bandits).
