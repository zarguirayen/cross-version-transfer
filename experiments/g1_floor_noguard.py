"""Validation G1: allocation with a per-stratum floor and a stratified estimator without fallback (always unbiased)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt.simulate import SCENARIOS, evaluate
from cvt.estimators import gain_law
for M in (20, 40):
    print(f"--- m = {M}")
    worst_cov, worst_gain = 1, 9
    for sc in SCENARIOS:
        base = evaluate(sc, "reset", m=M, reps=2000, seed=1)
        bh = evaluate(sc, "reset", m=M, reps=2000, seed=1, prior="hybrid")
        r = evaluate(sc, "ds_bayes", m=M, reps=2000, seed=1, prior="hybrid", allocation="floor")
        g = base["mse"] / r["mse"]; worst_cov = min(worst_cov, r["coverage"]); worst_gain = min(worst_gain, g)
        print(f"{sc.name:22s} phi2 {sc.phi2:.2f} loi x{gain_law(sc.phi2, M, 500):.2f} | gain x{g:.2f} couv {r['coverage']*100:.1f}% (reset hybride {bh['coverage']*100:.1f}%)")
    print(f"pire couverture {worst_cov:.3f}, pire gain {worst_gain:.2f}")
