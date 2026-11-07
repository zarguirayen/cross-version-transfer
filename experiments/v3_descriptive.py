"""Analyses descriptives de la v3, prévues dans PREREGISTRATION_V3.md (« rapportés sans critère »).
Lit results/v3/*.csv ; n'ajoute aucun test confirmatoire."""
import csv
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "v3"
rep = list(csv.DictReader((D / "replays.csv").open()))
atl = {(a["old"], a["new"]): a for a in csv.DictReader((D / "atlas.csv").open())}
h4 = list(csv.DictReader((D / "h4_real.csv").open()))
f = lambda r, k: float(r[k])
KEYS = ["bma", "eco_only", "graph", "graph_v3", "graph_v3_strict"]


def line(S, lab):
    out = [f"{lab:32s} n={len(S):3d}"]
    for k in KEYS:
        g = np.array([f(r, f"gain_{k}") for r in S]); c = np.array([f(r, f"cov_{k}") for r in S])
        out.append(f"{k} x{np.median(g):.2f} (min {g.min():.2f}; >=0,95 {np.mean(g >= 0.95):.0%}; couv {c.mean():.3f}/{c.min():.3f})")
    return "\n    ".join(out)


for m in ("20", "40"):
    S = [r for r in rep if r["m"] == m]
    print(f"\n=== m = {m} ===")
    print(line(S, "toutes"))
    for kind in ("produit", "mini"):
        print(line([r for r in S if r["kind"] == kind], kind))
    for lo, hi in ((0, 0.10), (0.10, 0.90), (0.90, 1.01)):
        T = [r for r in S if lo <= f(r, "p_new") < hi]
        if T:
            print(line(T, f"p' dans [{lo:.2f}, {hi:.2f})"))
    eco = np.array([int(atl[(r["old"], r["new"])]["n_eco"]) for r in S])
    for lo, hi in ((0, 20), (20, 60), (60, 200)):
        T = [r for r, e in zip(S, eco) if lo <= e < hi]
        if T:
            print(line(T, f"écosystème de {lo} à {hi - 1} agents"))
    fb = np.array([f(r, "fallback_rate") for r in S])
    print(f"taux de repli de la garde : moyen {fb.mean():.3f}, max {fb.max():.3f}")
    g3 = np.array([f(r, "gain_graph_v3") for r in S])
    print(f"probes équivalentes (gain x m) : médiane {np.median(g3) * int(m):.1f} pour {m} probes réelles")

A = list(atl.values())
print("\n=== atlas ===")
for kind in ("produit", "mini"):
    v = np.array([f(a, "phi2") for a in A if a["kind"] == kind])
    print(f"{kind}: phi² médian {np.median(v):.3f} (q1 {np.quantile(v, .25):.3f}, q3 {np.quantile(v, .75):.3f}), n={len(v)}")
print("\n=== régressions réelles (delta <= -0,10) ===")
for h in sorted(h4, key=lambda h: float(h["delta"])):
    if float(h["delta"]) <= -0.10:
        print(f"  {h['old']} -> {h['new']} : {float(h['p_old']):.3f} -> {float(h['p_new']):.3f} ; alerte {float(h['alert_rate']):.2f} ; probes {float(h['mean_probes']):.1f}")
mid = [h for h in h4 if -0.10 < float(h["delta"]) < 0]
print(f"baisses de moins de 10 points : {len(mid)} ; taux d'alerte moyen {np.mean([float(h['alert_rate']) for h in mid]):.3f}")
