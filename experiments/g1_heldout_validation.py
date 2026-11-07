"""Validation hors échantillon de la règle v0.3.1 (prior hybride + allocation adaptative) sur des
scénarios tirés au hasard, jamais vus pendant la mise au point.
Usage : python experiments/g1_heldout_validation.py [n_scenarios] [reps] [m]"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt.simulate import Scenario, evaluate  # noqa: E402
from cvt.estimators import gain_law  # noqa: E402

N_SC = int(sys.argv[1]) if len(sys.argv) > 1 else 200
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
M = int(sys.argv[3]) if len(sys.argv) > 3 else 20
METHOD = sys.argv[4] if len(sys.argv) > 4 else "ds_bayes"
ALLOC = sys.argv[5] if len(sys.argv) > 5 else "adaptive"
PRIOR = "hybrid"
N_V = 500
rng = np.random.default_rng(20261107)  # graine fixée avant de voir les résultats
rows = []
for i in range(N_SC):
    sc = Scenario(f"tire_{i:03d}", float(rng.uniform(0.5, 0.98)), float(rng.uniform(0.05, 0.95)),
                  float(rng.uniform(0.05, 0.95)))
    seed = 100 + i
    b = evaluate(sc, "reset", m=M, n_v=N_V, reps=REPS, seed=seed)
    bh = evaluate(sc, "reset", m=M, n_v=N_V, reps=REPS, seed=seed, prior="hybrid")
    r = evaluate(sc, METHOD, m=M, n_v=N_V, reps=REPS, seed=seed, prior=PRIOR, allocation=ALLOC)
    rows.append(dict(scenario=sc.name, pi=sc.pi, q1=sc.q1, q0=sc.q0, phi2=sc.phi2, m=M,
                     gain=b["mse"] / r["mse"], gain_law=gain_law(sc.phi2, M, N_V),
                     coverage=r["coverage"], coverage_reset_hybrid=bh["coverage"], coverage_reset_jeffreys=b["coverage"]))
out = Path(__file__).resolve().parents[1] / "results" / f"g1_heldout_{METHOD}_{ALLOC}_m{M}.csv"
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

g = np.array([r["gain"] for r in rows]); gl = np.array([r["gain_law"] for r in rows])
c = np.array([r["coverage"] for r in rows]); ch = np.array([r["coverage_reset_hybrid"] for r in rows])
cj = np.array([r["coverage_reset_jeffreys"] for r in rows]); ph = np.array([r["phi2"] for r in rows])
se_cov = np.sqrt(0.9 * 0.1 / REPS)
print(f"[{METHOD}, {ALLOC}] {N_SC} scénarios tirés au hasard, m = {M}, {REPS} mondes chacun (erreur type d'une couverture ≈ {se_cov:.3f})")
print(f"gain vs reset        : médiane x{np.median(g):.2f}, min x{g.min():.2f}, part < 0,95 : {np.mean(g < 0.95):.1%}")
print(f"gain / loi finie     : médiane {np.median(g / gl):.2f}, 5e-95e centiles {np.quantile(g / gl, .05):.2f}-{np.quantile(g / gl, .95):.2f}")
print(f"  si phi2 >= 0,3     : médiane {np.median((g / gl)[ph >= .3]):.2f} (n={int((ph >= .3).sum())})")
print(f"couverture estimateur: moyenne {c.mean():.3f}, min {c.min():.3f}, part < 0,88 : {np.mean(c < .88):.1%}, part < 0,85 : {np.mean(c < .85):.1%}")
print(f"couverture reset hyb.: moyenne {ch.mean():.3f}, min {ch.min():.3f}, part < 0,88 : {np.mean(ch < .88):.1%}")
print(f"couverture reset Jef.: moyenne {cj.mean():.3f}, min {cj.min():.3f}, part < 0,88 : {np.mean(cj < .88):.1%}")
print("écrit :", out)
