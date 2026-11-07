"""DÉVELOPPEMENT : estimateur graphe logistique, mondes synthétiques variés (niveau de la nouvelle version
tiré au hasard pour éviter l'artefact « vrai taux proche de 50 % »)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvt import estimators as E
from cvt import graph as Gr
from dev_graph_synthetic import world

def run(cfg, m=20, reps=400, seed=0, slope_sd=1.0):
    rng = np.random.default_rng(seed)
    er, eb, el, cl, cr, r2 = [], [], [], [], [], []
    for rep in range(reps):
        wr = np.random.default_rng([seed, rep])
        c = dict(cfg); c.setdefault("theta_new", float(wr.uniform(-2.5, 2.5)))
        aux, y_prev, y_new = world(wr, **c)
        X = Gr.ecosystem_features(aux, y_prev)
        n, p = len(y_new), y_new.mean()
        idx = rng.choice(n, m, replace=False)
        a, b = y_prev[idx], y_new[idx]
        f = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
        r = E.reset(*f, rng=rng, draws=1500, prior="hybrid")
        x = E.double_sampling_bma(int(y_prev.sum()), int((~y_prev).sum()), *f, rng=rng, draws=1500)
        g = Gr.graph_logit(X, idx, b, rng=rng, draws=3000, slope_sd=slope_sd)
        er.append((r.mean-p)**2); eb.append((x.mean-p)**2); el.append((g.mean-p)**2)
        cl.append(g.lo <= p <= g.hi); cr.append(r.lo <= p <= r.hi)
        # R² « oracle » de la régression logistique sur la population (pour la loi)
        r2.append(np.corrcoef(1/(1+np.exp(-(X @ np.linalg.lstsq(X, y_new - y_new.mean(), rcond=None)[0]))), y_new)[0, 1]**2 if y_new.std() > 0 else 0)
    mr = np.mean(er)
    return mr/np.mean(eb), mr/np.mean(el), np.mean(cl), np.mean(cr)

cfgs = {
    "écosystème très informatif": dict(spread=2.0, link=0.5),
    "écosystème informatif": dict(spread=1.5, link=1.0),
    "lignée forte, écosystème faible": dict(spread=0.5, link=2.0),
    "rien d'informatif": dict(spread=0.0, link=0.0),
    "nouvelle version très forte": dict(spread=1.5, link=1.0, theta_new=2.5),
    "nouvelle version très faible": dict(spread=1.5, link=1.0, theta_new=-2.5),
}
for name, c in cfgs.items():
    gb, gl, cl, cr = run(c)
    print(f"{name:32s} | prédécesseur seul x{gb:.2f} | graphe logistique x{gl:.2f} | couverture graphe {cl:.3f} (reset {cr:.3f})")
