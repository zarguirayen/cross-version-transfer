"""DEVELOPMENT (AppWorld, already seen): does the lineage edge add anything beyond the ecosystem alone?"""
import sys, importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/claude/cross-version-transfer/src")
from cvt import estimators as E, graph as Gr
sys.argv = ["x", "/home/claude/data/aw_root"]
spec = importlib.util.spec_from_file_location("v2", "/home/claude/cross-version-transfer/experiments/v2_appworld_preregistered.py")
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
rng = np.random.default_rng(5); out = []
for split in v2.SPLITS:
    av = v2.agents_available(split); Y = {a: v2.load(a, split)[1] for a in av}
    for kind, old, new in v2.transitions():
        y1, y2 = Y[old], Y[new]
        aux = np.array([Y[a] for a in av if a not in (old, new)])
        X = Gr.ecosystem_features(aux, y1); n = len(y2); p = y2.mean()
        er, e3, ee = [], [], []
        for _ in range(100):
            idx = rng.choice(n, 20, replace=False); s = int(y2[idx].sum())
            a, b = y1[idx], y2[idx]
            f = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
            r = E.reset(*f, rng=rng, draws=1000, prior="hybrid")
            g3 = Gr.graph_v3(X, idx, b, y1, rng=rng, draws=1500)
            eo = Gr.graph_bma_logit(X[:, :1], idx, b, rng=rng, draws=1500, average=False) if 0 < s < 20 else r
            er.append((r.mean-p)**2); e3.append((g3.mean-p)**2); ee.append((eo.mean-p)**2)
        out.append((kind, np.mean(er)/np.mean(e3), np.mean(er)/np.mean(ee)))
        print(kind, old, new, f"v3 x{out[-1][1]:.2f} eco x{out[-1][2]:.2f}", flush=True)
g3 = np.array([o[1] for o in out]); ge = np.array([o[2] for o in out])
from scipy import stats
print(f"v3 médiane x{np.median(g3):.3f} | eco seul x{np.median(ge):.3f} | v3 > eco sur {np.mean(g3>ge):.0%} | Wilcoxon p {stats.wilcoxon(g3, ge, alternative='greater').pvalue:.3g}")
for t in ("A_modele","B_architecture","C_famille"):
    s=[o for o in out if o[0]==t]; print(t, f"v3 x{np.median([o[1] for o in s]):.2f} eco x{np.median([o[2] for o in s]):.2f}")
