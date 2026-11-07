"""DÉVELOPPEMENT : estimateur « graphe » sur un monde synthétique de type réponse à l'item.
Chaque tâche a une difficulté d_j ; chaque agent a une capacité theta_a ; P(succès) = sigmoïde(theta_a - d_j + bruit).
On vérifie : gain conforme à la loi en R², couverture, absence de perte quand l'écosystème n'apporte rien."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import estimators as E  # noqa: E402
from cvt import graph as Gr  # noqa: E402


def world(rng, n=500, n_aux=8, spread=1.5, link=1.0, noise=0.5, theta_new=0.3):
    """spread : dispersion des difficultés (0 = tâches toutes pareilles : l'écosystème n'apporte rien).
    link : lien entre prédécesseur et nouvelle version (interaction spécifique partagée)."""
    d = rng.normal(0, spread, n)
    inter = rng.normal(0, 1, n)  # composante propre à la lignée v -> v'
    sig = lambda x: 1 / (1 + np.exp(-x))
    aux = np.array([rng.random(n) < sig(rng.normal(0, 1) - d + noise * rng.normal(0, 1, n)) for _ in range(n_aux)])
    y_prev = rng.random(n) < sig(0.2 - d + link * inter + noise * rng.normal(0, 1, n))
    y_new = rng.random(n) < sig(theta_new - d + link * inter + noise * rng.normal(0, 1, n))
    return aux, y_prev, y_new


def run(cfg, m=20, reps=400, seed=0):
    rng = np.random.default_rng(seed)
    e_reset, e_bma, e_graph, cov_g, r2s, r2p = [], [], [], [], [], []
    for rep in range(reps):
        aux, y_prev, y_new = world(np.random.default_rng([seed, rep]), **cfg)
        score = np.vstack([aux, y_prev[None]]).mean(axis=0)
        strata = {"M0": np.zeros(len(y_new), int), "M1_pred": y_prev.astype(int), "M2_graph": Gr.score_strata(score)}
        n, p = len(y_new), y_new.mean()
        idx = rng.choice(n, m, replace=False)
        a, b = y_prev[idx], y_new[idx]
        f = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
        r = E.reset(*f, rng=rng, draws=1500, prior="hybrid")
        x = E.double_sampling_bma(int(y_prev.sum()), int((~y_prev).sum()), *f, rng=rng, draws=1500)
        g = Gr.graph_bma(strata, idx, b, rng=rng, draws=1500, shrink=SHRINK)
        e_reset.append((r.mean - p) ** 2); e_bma.append((x.mean - p) ** 2); e_graph.append((g.mean - p) ** 2)
        cov_g.append(g.lo <= p <= g.hi)
        r2s.append(Gr.r2_strata(y_new, strata["M2_graph"])); r2p.append(Gr.r2_strata(y_new, strata["M1_pred"]))
    mr = np.mean(e_reset)
    return dict(gain_pred_only=mr / np.mean(e_bma), gain_graph=mr / np.mean(e_graph),
                law_graph=Gr.gain_law_r2(np.mean(r2s), m, 500), law_pred=Gr.gain_law_r2(np.mean(r2p), m, 500),
                coverage_graph=float(np.mean(cov_g)))


import os
SHRINK = os.environ.get("SHRINK", "pooled")
cfgs = {
    "écosystème très informatif": dict(spread=2.0, link=0.5),
    "écosystème informatif": dict(spread=1.5, link=1.0),
    "lignée forte, écosystème faible": dict(spread=0.5, link=2.0),
    "rien d'informatif": dict(spread=0.0, link=0.0),
    "nouvelle version très forte": dict(spread=1.5, link=1.0, theta_new=2.5),
    "nouvelle version très faible": dict(spread=1.5, link=1.0, theta_new=-2.5),
}
for name, c in cfgs.items():
    r = run(c)
    print(f"{name:32s} | prédécesseur seul x{r['gain_pred_only']:.2f} (loi x{r['law_pred']:.2f}) | "
          f"graphe x{r['gain_graph']:.2f} (loi x{r['law_graph']:.2f}) | couverture graphe {r['coverage_graph']:.3f}")
