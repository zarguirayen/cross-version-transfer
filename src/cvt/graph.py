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


def _logit_posterior(Z: np.ndarray, y: np.ndarray, prec: np.ndarray, draws: int, rng, mu: np.ndarray | None = None):
    """Postérieur de la régression logistique bayésienne (prior N(0, prec⁻¹)) par échantillonnage d'importance
    autour du mode (proposition t multivariée à 4 ddl, covariance de Laplace × 1,5).
    Renvoie (tirages rééchantillonnés, log vraisemblance marginale estimée, ESS)."""
    from scipy.special import gammaln
    k = Z.shape[1]
    mu = np.zeros(k) if mu is None else np.asarray(mu, float)
    beta = mu.copy()
    for _ in range(60):
        p = 1 / (1 + np.exp(-(Z @ beta)))
        grad = Z.T @ (y - p) - prec @ (beta - mu)
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + prec
        step = np.linalg.solve(H, grad)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    S = np.linalg.inv(H) * 1.5
    L = np.linalg.cholesky(S)
    nu = 4
    u = rng.standard_normal((draws, k)) / np.sqrt(rng.chisquare(nu, draws) / nu)[:, None]
    B = beta + u @ L.T
    logq = (gammaln((nu + k) / 2) - gammaln(nu / 2) - k / 2 * np.log(nu * np.pi)
            - np.log(np.diag(L)).sum() - (nu + k) / 2 * np.log1p((u ** 2).sum(axis=1) / nu))
    eta = B @ Z.T
    loglik = (y * eta - np.logaddexp(0, eta)).sum(axis=1)
    logprior = (-0.5 * np.einsum("ij,jk,ik->i", B - mu, prec, B - mu) - k / 2 * np.log(2 * np.pi)
                + 0.5 * np.linalg.slogdet(prec)[1])
    logw = loglik + logprior - logq
    mx = logw.max()
    log_ml = mx + np.log(np.mean(np.exp(logw - mx)))
    w = np.exp(logw - mx); w /= w.sum()
    return B[rng.choice(draws, draws, p=w)], float(log_ml), float(1 / (w ** 2).sum())


def graph_bma_logit(X: np.ndarray, probe_idx: np.ndarray, y_probe: np.ndarray, slope_sd: float = 1.0,
                    intercept_sd: float = 2.0, draws: int = 3000, rng=None, level: float = LEVEL,
                    finite_population: bool = True, average: bool = True,
                    slope_mean: np.ndarray | None = None, slope_sds: np.ndarray | None = None) -> Estimate:
    """Estimateur graphe v3 (deux corrections de principe, fixées avant toute donnée de confirmation) :
    1. Population finie : la cible est le taux réalisé sur les N tâches ; les probes y entrent par leur résultat
       observé, les N − m autres tâches par un tirage de Bernoulli(p_j) (incertitude de prédiction, pas seulement
       de paramètre).
    2. Moyenne de modèles : tous les sous-ensembles de covariables (2^d modèles, dont « aucun lien » = reset),
       même prior, pondérés par leur vraisemblance marginale estimée par échantillonnage d'importance ;
       prior uniforme sur les modèles. Le graphe n'est utilisé que si les probes le soutiennent."""
    rng = rng or np.random.default_rng()
    N, d = X.shape
    y = y_probe.astype(float)
    mask = np.zeros(N, bool); mask[probe_idx] = True
    subsets = [()] + [tuple(c for c in range(d) if (s >> c) & 1) for s in range(1, 2 ** d)] if average else [tuple(range(d))]
    fits = []
    for cols in subsets:
        Z = np.column_stack([np.ones(len(y))] + [X[probe_idx, c] for c in cols])
        sds = [slope_sd if slope_sds is None else slope_sds[c] for c in cols]
        mus = [0.0] + [0.0 if slope_mean is None else slope_mean[c] for c in cols]
        prec = np.diag([1 / intercept_sd ** 2] + [1 / s ** 2 for s in sds])
        B, lml, ess = _logit_posterior(Z, y, prec, draws, rng, mu=np.array(mus))
        fits.append((cols, B, lml, ess))
    lml = np.array([f[2] for f in fits])
    post = np.exp(lml - lml.max()); post /= post.sum()
    counts = rng.multinomial(draws, post)
    pts, tots = [], []
    for (cols, B, _, _), cnt in zip(fits, counts):
        if cnt == 0:
            continue
        Bc = B[:cnt]
        Xc = np.column_stack([np.ones(N)] + [X[:, c] for c in cols])
        P = 1 / (1 + np.exp(-(Bc @ Xc.T)))                                  # (cnt, N)
        if finite_population:
            rest = P[:, ~mask]
            pts.append((y.sum() + rest.sum(axis=1)) / N)
            tots.append((y.sum() + (rng.random(rest.shape) < rest).sum(axis=1)) / N)
        else:
            pts.append(P.mean(axis=1)); tots.append(P.mean(axis=1))
    pt, tot = np.concatenate(pts), np.concatenate(tots)
    a = (1 - level) / 2
    lo, hi = np.quantile(tot, [a, 1 - a])
    return Estimate(float(pt.mean()), float(lo), float(hi), "graph_bma_logit",
                    dict(posterior={str(f[0]): float(p) for f, p in zip(fits, post)}))


def _logit_map(z: np.ndarray, y: np.ndarray, sd: float = 10.0) -> np.ndarray:
    Z = np.column_stack([np.ones(len(y)), z]); y = y.astype(float)
    beta = np.zeros(2); prec = np.eye(2) / sd ** 2
    for _ in range(60):
        p = 1 / (1 + np.exp(-(Z @ beta)))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + prec
        step = np.linalg.solve(H, Z.T @ (y - p) - prec @ beta)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    return beta


def ecosystem_slope_prior(aux: np.ndarray | None, min_agents: int = 3, floor_sd: float = 0.5):
    """Prior empirique (graphe) pour la pente de difficulté : chaque agent a de l'écosystème est régressé sur la
    difficulté vue par les AUTRES agents (logit lissé, centré-réduit), sur toutes les tâches. Le prior de v' sur
    cette pente est N(moyenne, écart-type² + floor_sd²) des pentes observées : la façon dont les agents existants
    dépendent de la difficulté des tâches. Moins de min_agents agents -> None (prior par défaut N(0, 1))."""
    if aux is None or len(aux) < min_agents:
        return None
    k = aux.shape[0]
    slopes = []
    for a in range(k):
        others = np.delete(aux, a, axis=0)
        s = (others.sum(axis=0) + 0.5) / (k - 1 + 1)
        z = np.log(s / (1 - s)); sd = z.std()
        if sd == 0 or aux[a].all() or not aux[a].any():
            continue
        slopes.append(_logit_map((z - z.mean()) / sd, aux[a])[1])
    if len(slopes) < min_agents:
        return None
    slopes = np.array(slopes)
    return float(slopes.mean()), float(np.sqrt(slopes.var() + floor_sd ** 2))


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


def graph_v3(X: np.ndarray, probe_idx: np.ndarray, y_probe: np.ndarray, y_prev: np.ndarray, draws: int = 3000,
             rng=None, level: float = LEVEL) -> Estimate:
    """Estimateur graphe v3 = graph_bma_logit (population finie + moyenne de modèles), avec une garde
    d'identifiabilité : si toutes les probes ont le même résultat (0 succès ou 0 échec), aucune pente n'est
    identifiable à partir des probes et le prior seul déciderait ; on revient alors au double échantillonnage
    sur le seul prédécesseur (double_sampling_bma), qui n'a pas de pente à estimer."""
    from .estimators import double_sampling_bma
    rng = rng or np.random.default_rng()
    s = int(y_probe.sum()); f = len(y_probe) - s
    if min(s, f) == 0:
        a, b = y_prev[probe_idx], y_probe.astype(bool)
        e = double_sampling_bma(int(y_prev.sum()), int((~y_prev).sum()), int((a & b).sum()), int((a & ~b).sum()),
                                int((~a & b).sum()), int((~a & ~b).sum()), rng=rng, draws=1500, level=level)
        return Estimate(e.mean, e.lo, e.hi, "graph_v3", dict(fallback=True))
    e = graph_bma_logit(X, probe_idx, y_probe, draws=draws, rng=rng, level=level)
    return Estimate(e.mean, e.lo, e.hi, "graph_v3", dict(fallback=False, **e.extra))


def graph_greg(X: np.ndarray, probe_idx: np.ndarray, y_probe: np.ndarray, weights: np.ndarray | None = None,
               draws: int = 2000, rng=None, level: float = LEVEL) -> Estimate:
    """Estimateur assisté par modèle (GREG, Särndal et al. 1992) : prédictions du graphe (moyenne de modèles,
    graph_bma_logit) corrigées par les résidus observés sur les probes. Sans biais sous le plan quel que soit le
    modèle. weights[i] = nombre de tâches que la probe i représente (N_h pour 1 probe par strate ; N/m sous un
    tirage uniforme). Variance : strates regroupées par paires adjacentes (Cochran 1977, §5A.12) ; intervalle
    t de Student, avec un plancher de variance binomiale de Jeffreys pour les résidus tous nuls."""
    rng = rng or np.random.default_rng()
    N, d = X.shape
    m = len(probe_idx)
    w = np.full(m, N / m) if weights is None else np.asarray(weights, float)
    y = y_probe.astype(float)
    s = int(y.sum())
    if 0 < s < m:
        cols = tuple(range(d))
        subsets = [()] + [tuple(c for c in range(d) if (k >> c) & 1) for k in range(1, 2 ** d)]
        fits = []
        for cs in subsets:
            Z = np.column_stack([np.ones(m)] + [X[probe_idx, c] for c in cs])
            prec = np.diag([1 / 4.0] + [1.0] * len(cs))
            B, lml, _ = _logit_posterior(Z, y, prec, draws, rng)
            Xc = np.column_stack([np.ones(N)] + [X[:, c] for c in cs])
            fits.append((lml, (1 / (1 + np.exp(-(B @ Xc.T)))).mean(axis=0)))
        lml = np.array([f[0] for f in fits]); pw = np.exp(lml - lml.max()); pw /= pw.sum()
        P = sum(p * f[1] for p, f in zip(pw, fits))
    else:
        P = np.full(N, (s + 0.5) / (m + 1))          # aucun modèle identifiable : prédiction constante de Jeffreys
    e = y - P[probe_idx]
    est = P.mean() + (w * e).sum() / N
    if weights is None:                                  # tirage uniforme : variance du résidu moyen
        var, df = (1 - m / N) * e.var(ddof=1) / m, m - 1
    else:                                                # 1 probe par strate, probes dans l'ordre des strates
        pairs = [(i, i + 1) for i in range(0, m - 1, 2)]
        var = sum(((w[a] + w[b]) / N) ** 2 * (e[a] - e[b]) ** 2 / 4 for a, b in pairs)
        df = len(pairs)
    # plancher : le plan ne peut pas être plus de 4 fois plus précis qu'un tirage uniforme au taux de Jeffreys
    pj = (s + 0.5) / (m + 1)
    var = max(var, pj * (1 - pj) / (4 * m))
    from scipy.stats import t as student
    h = student.ppf(1 - (1 - level) / 2, df=max(df, 1)) * np.sqrt(var)
    return Estimate(float(np.clip(est, 0, 1)), float(max(est - h, 0)), float(min(est + h, 1)), "graph_greg", {})


def _bma_predict(X: np.ndarray, idx: np.ndarray, y: np.ndarray, draws: int, rng) -> np.ndarray:
    """Probabilité prédite sur les N tâches, moyenne sur les sous-ensembles de covariables (vraisemblance
    marginale), ajustée sur les probes idx ; constante de Jeffreys si les probes ont toutes le même résultat."""
    N, d = X.shape
    m, s = len(idx), int(y.sum())
    if not 0 < s < m:
        return np.full(N, (s + 0.5) / (m + 1))
    subsets = [()] + [tuple(c for c in range(d) if (k >> c) & 1) for k in range(1, 2 ** d)]
    fits = []
    for cs in subsets:
        Z = np.column_stack([np.ones(m)] + [X[idx, c] for c in cs])
        prec = np.diag([1 / 4.0] + [1.0] * len(cs))
        B, lml, _ = _logit_posterior(Z, y.astype(float), prec, draws, rng)
        Xc = np.column_stack([np.ones(N)] + [X[:, c] for c in cs])
        fits.append((lml, (1 / (1 + np.exp(-(B @ Xc.T)))).mean(axis=0)))
    lml = np.array([f[0] for f in fits]); pw = np.exp(lml - lml.max()); pw /= pw.sum()
    return sum(p * f[1] for p, f in zip(pw, fits))


def graph_greg_cf(X: np.ndarray, probe_idx: np.ndarray, y_probe: np.ndarray, weights: np.ndarray,
                  draws: int = 1500, rng=None, level: float = LEVEL) -> Estimate:
    """GREG à ajustement croisé (cross-fitting ; Zrnic & Candès 2024 pour l'inférence assistée par prédiction).
    Plan stratifié, probes données dans l'ordre des strates. Pli A = strates paires, pli B = strates impaires :
    chaque pli couvre toute l'échelle de difficulté. Le modèle ajusté sur un pli prédit les résidus de l'autre,
    donc les résidus sont hors échantillon. Estimation = moyenne des deux modèles sur les N tâches + correction
    pondérée par les résidus croisés. Variance : strates appariées (Cochran §5A.12), plancher Jeffreys/(4m).
    Intervalle : proportion à taille effective n* = p(1 − p)/v, corrigée des degrés de liberté, quantiles de
    Beta de Jeffreys (Korn & Graubard 1998)."""
    from scipy.stats import beta as Beta, t as student
    rng = rng or np.random.default_rng()
    N = X.shape[0]; m = len(probe_idx)
    y = y_probe.astype(float); w = np.asarray(weights, float)
    A = np.arange(0, m, 2); Bf = np.arange(1, m, 2)
    PA = _bma_predict(X, probe_idx[A], y[A], draws, rng)
    PB = _bma_predict(X, probe_idx[Bf], y[Bf], draws, rng)
    e = np.empty(m)
    e[A] = y[A] - PB[probe_idx[A]]
    e[Bf] = y[Bf] - PA[probe_idx[Bf]]
    est = float(np.clip(((PA + PB) / 2).mean() + (w * e).sum() / N, 0, 1))
    pairs = [(i, i + 1) for i in range(0, m - 1, 2)]
    var = sum(((w[a] + w[b]) / N) ** 2 * (e[a] - e[b]) ** 2 / 4 for a, b in pairs)
    s = int(y.sum()); pj = (s + 0.5) / (m + 1)
    var = max(var, pj * (1 - pj) / (4 * m))
    pc = min(max(est, 1e-6), 1 - 1e-6)
    alpha = 1 - level
    n_eff = pc * (1 - pc) / var
    n_eff *= (student.ppf(1 - alpha / 2, m - 1) / student.ppf(1 - alpha / 2, max(len(pairs), 1))) ** 2
    x = pc * n_eff
    lo = Beta.ppf(alpha / 2, x + 0.5, n_eff - x + 0.5)
    hi = Beta.ppf(1 - alpha / 2, x + 0.5, n_eff - x + 0.5)
    # convention de l'intervalle de Jeffreys (Brown, Cai & DasGupta 2001) : borne basse 0 si aucune probe n'a
    # réussi, borne haute 1 si toutes ont réussi
    if s == 0:
        lo = 0.0
    if s == m:
        hi = 1.0
    return Estimate(est, float(lo), float(hi), "graph_greg_cf", dict(n_eff=float(n_eff)))
