"""Validation G1: which (prior, allocation) combination keeps coverage >= 88% everywhere without losing the gain?

Paired simulations (the same worlds for every method).

Usage
-----
    python experiments/g1_coverage_grid.py [m] [reps]   # writes results/g1_coverage_grid_m{m}.csv"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt.simulate import SCENARIOS, evaluate  # noqa: E402
from cvt.estimators import gain_law  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 20
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
N_V = 500
configs = [
    ("reset", "jeffreys", "proportional"),
    ("reset", "hybrid", "proportional"),
    ("ds_bayes", "jeffreys", "proportional"),
    ("ds_bayes", "hybrid", "proportional"),
    ("ds_bayes", "hybrid", "floor"),
    ("ds_freq", "-", "proportional"),
    ("full", "-", "proportional"),
]
rows = []
for sc in SCENARIOS:
    base = evaluate(sc, "reset", m=M, n_v=N_V, reps=REPS, seed=1, prior="jeffreys")
    for method, prior, alloc in configs:
        r = evaluate(sc, method, m=M, n_v=N_V, reps=REPS, seed=1,
                     prior=prior if prior != "-" else "jeffreys", allocation=alloc)
        rows.append(dict(scenario=sc.name, phi2=round(sc.phi2, 3), method=method, prior=prior, allocation=alloc,
                         m=M, gain_vs_reset=round(base["mse"] / r["mse"], 3),
                         gain_law=round(gain_law(sc.phi2, M, N_V), 3), coverage=round(r["coverage"], 3),
                         width=round(r["width"], 4), guard_rate=r["guard_rate"]))
out = Path(__file__).resolve().parents[1] / "results" / f"g1_coverage_grid_m{M}.csv"
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

labels = [f"{m[:6]}-{p[:6]}-{a[:4]}" for m, p, a in configs]
print(f"m = {M}, n_v = {N_V}, {REPS} mondes appariés par scénario")
print(f"{'scenario':22s} phi2  loi   | " + " | ".join(f"{l:>17s}" for l in labels))
for sc in SCENARIOS:
    rs = [r for r in rows if r["scenario"] == sc.name]
    print(f"{sc.name:22s} {sc.phi2:.2f} {gain_law(sc.phi2, M, N_V):5.2f} | "
          + " | ".join(f"x{r['gain_vs_reset']:5.2f} {r['coverage']*100:4.1f}%   " for r in rs))
print("\nrésumé par configuration")
for (m_, p_, a_), l in zip(configs, labels):
    rs = [r for r in rows if (r["method"], r["prior"], r["allocation"]) == (m_, p_, a_)]
    cov = [r["coverage"] for r in rs]
    g = [r["gain_vs_reset"] for r in rs]
    print(f"{l:20s} couverture min {min(cov):.3f} moy {sum(cov)/len(cov):.3f} | gain min {min(g):.2f}")
print("écrit :", out)
