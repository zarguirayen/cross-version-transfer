"""DÉVELOPPEMENT : règles de repli évaluées a posteriori sur les rejeux stockés (données déjà vues).
Usage : python experiments/dev_v3_rules.py results/dev_v3_appworld_replays.csv [autres fichiers]"""
import csv
import sys
from collections import defaultdict

import numpy as np

M = 20


def rule(r, name):
    s = int(r["s_probe"]); f = M - s
    if name not in ("bma", "graph", "graph_fp", "graph_v3") and f"{name.split('|')[0].replace('fp','graph_fp').replace('v3','graph_v3')}_mean" not in r:
        return None
    pick = {
        "bma": "bma", "graph": "graph", "graph_fp": "graph_fp", "graph_v3": "graph_v3",
        "fp|s<2->reset": "graph_fp" if s >= 2 else "reset",
        "fp|s<2->bma": "graph_fp" if s >= 2 else "bma",
        "fp|min(s,f)<2->bma": "graph_fp" if min(s, f) >= 2 else "bma",
        "v3|min(s,f)<2->bma": "graph_v3" if min(s, f) >= 2 else "bma",
        "fp|min(s,f)<1->reset": "graph_fp" if min(s, f) >= 1 else "reset",
        "fp|min(s,f)<1->bma": "graph_fp" if min(s, f) >= 1 else "bma",
        "v3|min(s,f)<1->reset": "graph_v3" if min(s, f) >= 1 else "reset",
        "v3|min(s,f)<1->bma": "graph_v3" if min(s, f) >= 1 else "bma",
        "fp|min(s,f)<2->reset": "graph_fp" if min(s, f) >= 2 else "reset",
    }[name]
    return float(r[f"{pick}_mean"]), float(r[f"{pick}_lo"]), float(r[f"{pick}_hi"])


RULES = ["bma", "graph", "graph_fp", "graph_v3", "fp|s<2->reset", "fp|s<2->bma", "fp|min(s,f)<2->bma",
         "fp|min(s,f)<2->reset", "v3|min(s,f)<2->bma",
         "fp|min(s,f)<1->reset", "fp|min(s,f)<1->bma", "v3|min(s,f)<1->reset", "v3|min(s,f)<1->bma"]

for path in sys.argv[1:]:
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[r["case"]].append(r)
    res = {k: [] for k in RULES}
    low = []
    for case, rs in by.items():
        p = float(rs[0]["p_new"]); low.append(p < 0.10 or p > 0.90)
        mr = np.mean([(float(r["reset_mean"]) - p) ** 2 for r in rs])
        for k in RULES:
            est = [rule(r, k) for r in rs]
            mse = np.mean([(e[0] - p) ** 2 for e in est])
            cov = np.mean([e[1] <= p <= e[2] for e in est])
            res[k].append((mr / mse, cov))
    low = np.array(low)
    print(f"\n{path} : {len(by)} cas ({low.sum()} avec p' < 0,10 ou > 0,90)")
    for k in RULES:
        g = np.array([x[0] for x in res[k]]); c = np.array([x[1] for x in res[k]])
        msg = f"  {k:22s} médiane x{np.median(g):.3f} min x{g.min():.2f} >=0,95 {np.mean(g >= 0.95):.0%} couv {c.mean():.3f} (min {c.min():.3f})"
        if low.any():
            msg += f" | extrêmes : méd x{np.median(g[low]):.2f} min x{g[low].min():.2f} couv min {c[low].min():.3f}"
        print(msg)
