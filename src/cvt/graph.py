"""Double échantillonnage « graphe » : l'historique de tout l'écosystème d'agents sert de variable auxiliaire.

Pour une nouvelle version v', on dispose, sur les N mêmes tâches :
  - y_prev : résultats du prédécesseur v (vecteur binaire de longueur N) ;
  - score  : réussite moyenne des autres agents du graphe sur chaque tâche (dans [0, 1]) ;
  - m probes : résultats de v' sur m de ces tâches.

Trois modèles concourent, pondérés par leur vraisemblance marginale (Beta-binomiale exacte, prior de Jeffreys) :
  M0 : une seule strate (aucun lien)              -> p' = q
  M1 : strates = résultat du prédécesseur (2)     -> double échantillonnage v0.3.2
  M2 : strates = K niveaux du score de l'écosystème -> double échantillonnage « graphe »
Poids des strates : Dirichlet(1/2 + effectifs) sur les N tâches. Moyenne lue sur des tirages de Jeffreys,
intervalle à 90 % sur des tirages de Clopper-Pearson. Aucun paramètre réglé sur les données (K fixé a priori).

Loi attendue : gain ≈ 1 / (1 - R² + R²·m/N), où R² est la part de variance de v' expliquée par les strates.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betaln

from .estimators import LEVEL, Estimate, _beta_draws

K_DEFAULT = 4


def score_strata(score: np.ndarray, k: int = K_DEFAULT) -> np.ndarray:
    """Découpe le score en k niveaux par quantiles (les ex aequo restent ensemble : parfois moins de k niveaux)."""
    edges = np.unique(np.quantile(score, np.linspace(0, 1, k + 1)[1:-1]))
    return np.digitize(score, edges)


def _relabel(g: np.ndarray) -> np.ndarray:
    _, inv = np.unique(g, return_inverse=True)
    return inv


def _log_ml(g_probe: np.ndarray, y_probe: np.ndarray, n_strata: int) -> float:
    lm = 0.0
    for k in range(n_strata):
        sel = g_probe == k
        s = int(y_probe[sel].sum()); f = int(sel.sum()) - s
        lm += betaln(s + 0.5, f + 0.5) - betaln(0.5, 0.5)
    return lm


def _shrunk_draws(s: int, f: int, mu: float, c: float, draws: int, rng, mode: str) -> np.ndarray:
    """Strate tirée vers le taux global mu des probes (concentration c), au lieu de 1/2.
    point : Beta(c·mu + s, c·(1-mu) + f) ; band : mélange à la Clopper-Pearson (une observation de marge)."""
    a0, b0 = c * mu, c * (1 - mu)
    if mode == "jeffreys":
        return rng.beta(a0 + s, b0 + f, draws)
    half = draws // 2
    return np.concatenate([rng.beta(a0 + s, b0 + f + 1, half), rng.beta(a0 + s + 1, b0 + f, draws - half)])


def graph_bma(strata_by_model: dict, probe_idx: np.ndarray, y_probe: np.ndarray, prior: dict | None = None,
              draws: int = 4000, rng=None, level: float = LEVEL, shrink: str = "pooled", c: float = 2.0) -> Estimate:
    """strata_by_model : {nom_du_modèle: vecteur de strates de longueur N}. probe_idx : indices des probes.
    Renvoie l'estimation du taux de réussite de v' sur la population des N tâches."""
    rng = rng or np.random.default_rng()
    names = list(strata_by_model)
    prior = prior or {n: 1 / len(names) for n in names}
    gs = {n: _relabel(np.asarray(strata_by_model[n])) for n in names}
    logpost = np.array([np.log(prior[n]) + _log_ml(gs[n][probe_idx], y_probe, gs[n].max() + 1) for n in names])
    post = np.exp(logpost - logpost.max()); post /= post.sum()
    counts = rng.multinomial(draws, post)
    c_shrink = c

    def mixture(pr):
        out = []
        for n, cnt in zip(names, counts):
            if cnt == 0:
                continue
            g = gs[n]; K = g.max() + 1
            nk = np.bincount(g, minlength=K)
            W = rng.dirichlet(nk + 0.5, cnt)                                 # poids des strates
            gp = g[probe_idx]
            if shrink == "pooled" and K > 1:
                mu = (y_probe.sum() + 0.5) / (len(y_probe) + 1)
                Q = np.column_stack([_shrunk_draws(int(y_probe[gp == k].sum()), int((gp == k).sum() - y_probe[gp == k].sum()),
                                                   mu, c_shrink, cnt, rng, pr) for k in range(K)])
            else:
                Q = np.column_stack([_beta_draws(int(y_probe[gp == k].sum()), int((gp == k).sum() - y_probe[gp == k].sum()), cnt, rng, pr)
                                     for k in range(K)])
            for col in range(K):
                rng.shuffle(Q[:, col])
            out.append((W * Q).sum(axis=1))
        return np.concatenate(out)

    point, band = mixture("jeffreys"), mixture("cp")
    a = (1 - level) / 2
    lo, hi = np.quantile(band, [a, 1 - a])
    return Estimate(float(point.mean()), float(lo), float(hi), "graph_bma",
                    dict(posterior={n: float(p) for n, p in zip(names, post)}))


def ecosystem_features(aux: np.ndarray | None, y_prev: np.ndarray | None) -> np.ndarray:
    """Covariables par tâche, centrées-réduites sur la population des N tâches :
    - difficulté vue par l'écosystème : logit de la réussite moyenne (lissée) des autres agents ;
    - résultat du prédécesseur (lien propre à la lignée)."""
    cols = []
    if aux is not None and len(aux):
        k = aux.shape[0]
        s = (aux.sum(axis=0) + 0.5) / (k + 1)
        cols.append(np.log(s / (1 - s)))
    if y_prev is not None:
        cols.append(y_prev.astype(float))
    X = np.column_stack(cols)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def graph_logit(X: np.ndarray, probe_idx: np.ndarray, y_probe: np.ndarray, slope_sd: float = 1.0,
                intercept_sd: float = 2.0, draws: int = 4000, rng=None, level: float = LEVEL) -> Estimate:
    """Estimateur « graphe » assisté par modèle : P(v' réussit la tâche j) = sigmoïde(a + X_j·b).
    Priors : a ~ N(0, intercept_sd²) ; b ~ N(0, slope_sd²) (centré sur « aucun lien »).
    Postérieur par échantillonnage d'importance autour du mode (proposition t de Student à 4 ddl).
    Cible : moyenne des probabilités prédites sur les N tâches (population)."""
    rng = rng or np.random.default_rng()
    Xp = X[probe_idx]
    y = y_probe.astype(float)
    n, d = X.shape
    Z = np.column_stack([np.ones(len(y)), Xp])
    prec = np.diag([1 / intercept_sd ** 2] + [1 / slope_sd ** 2] * d)

    def logpost(B):  # B : (k, d+1)
        eta = B @ Z.T
        ll = (y * eta - np.logaddexp(0, eta)).sum(axis=1)
        return ll - 0.5 * np.einsum("ij,jk,ik->i", B, prec, B)

    beta = np.zeros(d + 1)
    for _ in range(50):  # Newton pour le mode a posteriori
        p = 1 / (1 + np.exp(-(Z @ beta)))
        grad = Z.T @ (y - p) - prec @ beta
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + prec
        step = np.linalg.solve(H, grad)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    cov = np.linalg.inv(H)
    L = np.linalg.cholesky(cov * 1.5)
    dfree = 4
    zz = rng.standard_normal((draws, d + 1))
    chi = rng.chisquare(dfree, draws) / dfree
    B = beta + (zz / np.sqrt(chi)[:, None]) @ L.T
    # densité de la proposition (t multivariée), à une constante près
    q = -(dfree + d + 1) / 2 * np.log1p(((zz / np.sqrt(chi)[:, None]) ** 2).sum(axis=1) / dfree)
    logw = logpost(B) - q
    w = np.exp(logw - logw.max()); w /= w.sum()
    ess = 1 / (w ** 2).sum()
    B = B[rng.choice(draws, draws, p=w)]
    eta_pop = B[:, :1] + B[:, 1:] @ X.T                                    # (draws, N)
    ppop = (1 / (1 + np.exp(-eta_pop))).mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(ppop, [a, 1 - a])
    return Estimate(float(ppop.mean()), float(lo), float(hi), "graph_logit",
                    dict(ess=float(ess), slopes=beta[1:].tolist()))


def r2_strata(y_new: np.ndarray, g: np.ndarray) -> float:
    """Part de la variance de y_new expliquée par les strates g (R² inter-strates)."""
    y = y_new.astype(float)
    tot = y.var()
    if tot == 0:
        return 0.0
    means = np.array([y[g == k].mean() for k in np.unique(g)])
    w = np.array([np.mean(g == k) for k in np.unique(g)])
    return float((w * (means - y.mean()) ** 2).sum() / tot)


def gain_law_r2(r2: float, m: int, n: int) -> float:
    return 1.0 / (1.0 - r2 + r2 * m / n)
