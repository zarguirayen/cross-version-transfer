"""Comparer la moyenne de modèles (BMA) aux variantes précédentes, sur les 9 scénarios de référence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt.simulate import SCENARIOS, evaluate
from cvt.estimators import gain_law
M = int(sys.argv[1]) if len(sys.argv) > 1 else 20
for alloc in ("proportional", "adaptive"):
    print(f"--- BMA, allocation {alloc}, m = {M}")
    wc, wg = 1, 9
    for sc in SCENARIOS:
        b = evaluate(sc, "reset", m=M, reps=2000, seed=1)
        r = evaluate(sc, "ds_bma", m=M, reps=2000, seed=1, allocation=alloc)
        g = b["mse"] / r["mse"]; wc = min(wc, r["coverage"]); wg = min(wg, g)
        print(f"{sc.name:22s} phi2 {sc.phi2:.2f} loi x{gain_law(sc.phi2, M, 500):.2f} | gain x{g:.2f} couv {r['coverage']*100:.1f}%  P(lien) moy {r['guard_rate']:.2f}")
    print(f"pire couverture {wc:.3f}, pire gain {wg:.2f}")
