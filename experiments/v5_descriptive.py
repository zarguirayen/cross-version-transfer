"""Analyses descriptives v5 prévues (« rapportées sans critère ») : par jeu de tâches, par famille, m = 40,
versions sous 10 %, largeurs. N'ajoute aucun test confirmatoire."""
import csv
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "v5"
rows = list(csv.DictReader((D / "replays.csv").open()))
K = ["bma_u", "v3_u", "reset_s", "v3_s", "greg_cf_s"]
f = lambda S, k: np.array([float(r[k]) for r in S])


def show(S, lab):
    out = f"{lab:30s} n={len(S):2d} |"
    for q in K:
        g = f(S, "gain_" + q); c = f(S, "cov_" + q)
        out += f" {q} x{np.median(g):.2f} (min {g.min():.2f}; couv {c.mean():.2f}/{c.min():.2f}) |"
    print(out)


for m in ("20", "40"):
    S = [r for r in rows if r["m"] == m]
    print(f"\n=== m = {m} ===")
    show(S, "toutes")
    for fam in ("doubao_mswe", "autre"):
        show([r for r in S if r["family"] == fam], fam)
    for ts in sorted({r["taskset"] for r in S}):
        show([r for r in S if r["taskset"] == ts], ts)
    show([r for r in S if float(r["p_new"]) < 0.10], "p' < 0,10")
    show([r for r in S if float(r["p_new"]) >= 0.10], "p' >= 0,10")
    de = {q: f(S, "decerr_" + q).mean() for q in ["reset_u"] + K}
    print("erreur de décision moyenne (toutes) : " + ", ".join(f"{q} {v:.3f}" for q, v in de.items()))
S = [r for r in rows if r["m"] == "20"]
print("\nmédianes par jeu de tâches (unité = jeu), m = 20 :")
ts = sorted({r["taskset"] for r in S})
med = {t: {q: np.median(f([r for r in S if r["taskset"] == t], "gain_" + q)) for q in K} for t in ts}
for t in ts:
    print(f"  {t:18s} " + " ".join(f"{q} x{med[t][q]:.2f}" for q in K))
better = sum(med[t]["v3_s"] > med[t]["v3_u"] for t in ts)
print(f"v3_s > v3_u dans {better}/{len(ts)} jeux de tâches ; greg_cf_s >= 0,95 dans {sum(med[t]['greg_cf_s'] >= 0.95 for t in ts)}/{len(ts)}")
