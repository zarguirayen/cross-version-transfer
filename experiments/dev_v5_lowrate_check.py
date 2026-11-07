"""DÉVELOPPEMENT : vérifie la convention de Jeffreys de graph_greg_cf sur les transitions à faible taux de
Verified (déjà vu), y compris la soumission publiée vide."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import estimators as E, graph as Gr
import v3_swebench_data as D
ids, Y, dates, lin, _ = D.load("/home/claude/data/swe/experiments")
rng = np.random.default_rng(7)
for L, old, new in D.transitions(dates, lin):
    if Y[new].mean() >= 0.10:
        continue
    eco = D.ecosystem(dates, lin, old, new)
    X = Gr.ecosystem_features(np.array([Y[e] for e in eco]) if eco else None, Y[old]); y = Y[new]; p = y.mean()
    blocks = np.array_split(np.argsort(X[:, 0], kind="stable"), 20); w = np.array([len(b) for b in blocks], float)
    er, eg, cv = [], [], []
    for _ in range(300):
        iu = rng.choice(len(y), 20, replace=False); a, b = Y[old][iu], y[iu]
        r = E.reset(int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum()), rng=rng, draws=800, prior="hybrid")
        idx = np.array([rng.choice(bb) for bb in blocks])
        g = Gr.graph_greg_cf(X, idx, y[idx], w, rng=rng, draws=800)
        er.append((r.mean - p) ** 2); eg.append((g.mean - p) ** 2); cv.append(g.lo <= p <= g.hi)
    print(f"{new[:50]:50s} p'={p:.3f} gain greg_cf x{np.mean(er)/np.mean(eg):.2f} couverture {np.mean(cv):.3f}")
