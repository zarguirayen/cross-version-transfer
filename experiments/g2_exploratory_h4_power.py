"""EXPLORATORY (after the freeze, not pre-registered): power of the H4 detection by number of probes.

Answers "how many probes are needed to detect a 10-point degradation?". Same protocol as g2_preregistered.py (H4),
budgets m = 20, 40, 60, 80, R = 300."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvt import estimators as E  # noqa: E402
import g2_preregistered as G  # noqa: E402  (réutilise load, flips, PAIRS, BENCHES)

G.DATA = Path(sys.argv[1]) / "results"
rng = np.random.default_rng(20261108)
rows = []
for pair, (label, change, in_scope) in G.PAIRS.items():
    if not in_scope:
        continue
    for bench in G.BENCHES:
        y1, y2 = G.load(pair, bench)
        n, p2 = len(y2), y2.mean()
        k = int(round(G.DELTA_H4 * n))
        succ = np.flatnonzero(y2)
        for m in (20, 40, 60, 80):
            if m > n // 2:
                continue
            det = {"bma": 0, "reset": 0}
            for _ in range(300):
                y3 = y2.copy()
                y3[rng.choice(succ, k, replace=False)] = False
                idx = rng.choice(n, m, replace=False)
                f = G.flips(y2[idx], y3[idx])
                det["bma"] += E.double_sampling_bma(int(y2.sum()), int((~y2).sum()), *f, rng=rng, draws=2000).hi < p2
                det["reset"] += E.reset(*f, rng=rng, draws=2000, prior="hybrid").hi < p2
            rows.append(dict(pair=pair, bench=bench, m=m, power_bma=det["bma"] / 300, power_reset=det["reset"] / 300))
out = Path(__file__).resolve().parents[1] / "results" / "g2" / "exploratory_h4_power.csv"
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
for m in (20, 40, 60, 80):
    S = [r for r in rows if r["m"] == m]
    b = np.array([r["power_bma"] for r in S]); r_ = np.array([r["power_reset"] for r in S])
    print(f"m={m:3d} ({len(S)} transitions) puissance BMA moy {b.mean():.3f} (part >= 0,8 : {np.mean(b >= .8):.0%}) | reset {r_.mean():.3f} (part >= 0,8 : {np.mean(r_ >= .8):.0%})")
