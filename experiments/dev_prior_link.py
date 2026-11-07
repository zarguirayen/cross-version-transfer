"""DÉVELOPPEMENT (pas une preuve) : régler le prior P(lien) du BMA.
- Entraînement : données Sheng & Lu (déjà vues), validation croisée en laissant une famille de côté.
- Coût dans le pire cas : scénarios synthétiques où les versions sont indépendantes.
La valeur retenue sera figée dans prereg-v2 puis confirmée sur AppWorld (jamais vu)."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvt import estimators as E  # noqa: E402
from cvt.simulate import Scenario  # noqa: E402
import g2_preregistered as G  # noqa: E402

G.DATA = Path(sys.argv[1]) / "results"
GRID = (0.5, 0.7, 0.8, 0.9, 0.95)
M, R = 20, 400
rng = np.random.default_rng(20261109)


def replay(y1, y2, prior_link, m, R, rng):
    n, p2 = len(y1), y2.mean()
    er, eb, cov = [], [], []
    for _ in range(R):
        idx = rng.choice(n, m, replace=False)
        a, b = y1[idx], y2[idx]
        f = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
        r = E.reset(*f, rng=rng, draws=1500, prior="hybrid")
        x = E.double_sampling_bma(int(y1.sum()), int((~y1).sum()), *f, rng=rng, draws=1500, prior_link=prior_link)
        er.append((r.mean - p2) ** 2); eb.append((x.mean - p2) ** 2); cov.append(x.lo <= p2 <= x.hi)
    return np.mean(er) / np.mean(eb), np.mean(cov)


rows = []
for pair, (label, change, in_scope) in G.PAIRS.items():
    for bench in G.BENCHES:
        y1, y2 = G.load(pair, bench)
        for pl in GRID:
            g, c = replay(y1, y2, pl, M, R, rng)
            rows.append(dict(source="sheng_lu", pair=pair, bench=bench, in_scope=in_scope, prior_link=pl, gain=g, coverage=c))
    print("fait", pair, flush=True)

# pire cas synthétique : versions indépendantes, plusieurs niveaux de réussite
for pi, p in ((0.7, 0.8), (0.7, 0.5), (0.5, 0.3), (0.9, 0.9), (0.3, 0.1)):
    sc = Scenario(f"indep_{pi}_{p}", pi, p, p)
    for pl in GRID:
        gs, cs = [], []
        for rep in range(3):
            w = np.random.default_rng([7, rep])
            y1 = w.random(500) < sc.pi
            y2 = w.random(500) < sc.q1
            g, c = replay(y1, y2, pl, M, 300, rng)
            gs.append(g); cs.append(c)
        rows.append(dict(source="synth_indep", pair=sc.name, bench="-", in_scope=True, prior_link=pl,
                         gain=float(np.mean(gs)), coverage=float(np.mean(cs))))

out = Path(__file__).resolve().parents[1] / "results" / "dev_prior_link.csv"
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

fams = sorted({r["pair"] for r in rows if r["source"] == "sheng_lu" and r["in_scope"]})
print("\nprior_link | Sheng (périmètre) gain médian / min / couverture moy | hors périmètre gain min | synthétique indépendant gain min / couverture min")
for pl in GRID:
    S = [r for r in rows if r["source"] == "sheng_lu" and r["in_scope"] and r["prior_link"] == pl]
    O = [r for r in rows if r["source"] == "sheng_lu" and not r["in_scope"] and r["prior_link"] == pl]
    Y = [r for r in rows if r["source"] == "synth_indep" and r["prior_link"] == pl]
    print(f"  {pl:4.2f}    | x{np.median([r['gain'] for r in S]):.3f} / x{min(r['gain'] for r in S):.3f} / {np.mean([r['coverage'] for r in S]):.3f}"
          f" | x{min(r['gain'] for r in O):.3f} | x{min(r['gain'] for r in Y):.3f} / {min(r['coverage'] for r in Y):.3f}")
print("\nValidation croisée (famille laissée de côté) : prior choisi sur les 4 autres familles -> gain médian sur la famille laissée")
for f in fams:
    best = max(GRID, key=lambda pl: np.median([r["gain"] for r in rows if r["source"] == "sheng_lu" and r["in_scope"]
                                              and r["pair"] != f and r["prior_link"] == pl]))
    held = [r["gain"] for r in rows if r["source"] == "sheng_lu" and r["pair"] == f and r["prior_link"] == best]
    base = [r["gain"] for r in rows if r["source"] == "sheng_lu" and r["pair"] == f and r["prior_link"] == 0.5]
    print(f"  {f:8s} prior choisi {best} -> gain médian x{np.median(held):.3f} (prior 0,5 : x{np.median(base):.3f})")
