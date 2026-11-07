"""Estimateurs de la fiabilité d'une nouvelle version v' à partir de l'historique de v et de probes appariées.

Notation
--------
s_v, f_v : succès / échecs de v sur son historique (n_v = s_v + f_v tâches)
n11, n10, n01, n00 : matrice de bascules sur les m probes
    n11 = v réussit, v' réussit ; n10 = v réussit, v' échoue
    n01 = v échoue,  v' réussit ; n00 = v échoue,  v' échoue
s_own, f_own : preuves propres de v' sur des tâches nouvelles (résultat de v inconnu)

Toutes les fonctions renvoient un objet Estimate (moyenne, intervalle à 90 %, champs de diagnostic).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEVEL = 0.90


@dataclass
class Estimate:
    mean: float
    lo: float
    hi: float
    method: str
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- loi de gain
def gain_law(phi2: float, m: int, n_v: int) -> float:
    """Gain d'efficacité du double échantillonnage : m_eff / m (loi finie)."""
    return 1.0 / (1.0 - phi2 + phi2 * m / n_v)


def phi_signed(n11: int, n10: int, n01: int, n00: int) -> float:
    m = n11 + n10 + n01 + n00
    r1, c1 = n11 + n10, n11 + n01
    den = r1 * (m - r1) * c1 * (m - c1)
    return 0.0 if den == 0 else (n11 * n00 - n10 * n01) / np.sqrt(den)


def phi2_posterior(n11, n10, n01, n00, draws=2000, rng=None, a=0.5) -> np.ndarray:
    """Tirages a posteriori de phi² (Dirichlet(a) sur la matrice de bascules)."""
    rng = rng or np.random.default_rng()
    p11, p10, p01, p00 = rng.dirichlet([n11 + a, n10 + a, n01 + a, n00 + a], draws).T
    r1, c1 = p11 + p10, p11 + p01
    return (p11 * p00 - p10 * p01) ** 2 / (r1 * (1 - r1) * c1 * (1 - c1))


def borrowed_information(n11, n10, n01, n00, n_v, draws=2000, rng=None) -> dict:
    """Information empruntée à l'historique. Borne basse (quantile 10 % de phi²) pour décider,
    médiane pour l'atlas."""
    m = n11 + n10 + n01 + n00
    d = phi2_posterior(n11, n10, n01, n00, draws, rng)
    med, q10 = float(np.median(d)), float(np.quantile(d, 0.10))
    m_eff_med, m_eff_low = m * gain_law(med, m, n_v), m * gain_law(q10, m, n_v)
    i_hist = max(0.0, m_eff_low - m)
    return dict(phi=phi_signed(n11, n10, n01, n00), phi2_median=med, phi2_q10=q10,
                m_eff_median=m_eff_med, m_eff_lower=m_eff_low, i_hist=i_hist, w_hist=i_hist / n_v)


# ---------------------------------------------------------------- tirages d'une proportion
def _beta_draws(s: int, f: int, draws: int, rng, prior: str) -> np.ndarray:
    """Tirages pour une proportion binomiale.
    jeffreys : Beta(s+1/2, f+1/2)
    cp       : mélange des deux lois de Clopper-Pearson, Beta(s, f+1) et Beta(s+1, f),
               qui donne des intervalles plus prudents pour les petits effectifs."""
    if prior == "jeffreys":
        return rng.beta(s + 0.5, f + 0.5, draws)
    if prior == "cp":
        half = draws // 2
        lo = rng.beta(max(s, 1e-9), f + 1, half) if s > 0 else np.zeros(half)
        hi = rng.beta(s + 1, max(f, 1e-9), draws - half) if f > 0 else np.ones(draws - half)
        return np.concatenate([lo, hi])
    raise ValueError(prior)


def _summarise(p: np.ndarray, method: str, extra=None, level=LEVEL) -> Estimate:
    a = (1 - level) / 2
    lo, hi = np.quantile(p, [a, 1 - a])
    return Estimate(float(p.mean()), float(lo), float(hi), method, extra or {})


def _reweight(p: np.ndarray, s_own: int, f_own: int, rng) -> np.ndarray:
    if s_own == 0 and f_own == 0:
        return p
    logw = s_own * np.log(np.clip(p, 1e-12, 1)) + f_own * np.log(np.clip(1 - p, 1e-12, 1))
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return p[rng.choice(len(p), len(p), p=w)]


# ---------------------------------------------------------------- estimateurs
def reset(n11, n10, n01, n00, s_own=0, f_own=0, draws=4000, rng=None, prior="jeffreys") -> Estimate:
    """Probes seules : on ignore l'historique de v."""
    rng = rng or np.random.default_rng()
    s, f = n11 + n01 + s_own, n10 + n00 + f_own
    if prior == "hybrid":
        point, band = _beta_draws(s, f, draws, rng, "jeffreys"), _beta_draws(s, f, draws, rng, "cp")
        lo, hi = np.quantile(band, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
        return Estimate(float(point.mean()), float(lo), float(hi), "reset[hybrid]")
    return _summarise(_beta_draws(s, f, draws, rng, prior), f"reset[{prior}]")


def full_inheritance(s_v, f_v, n11, n10, n01, n00, s_own=0, f_own=0, draws=4000, rng=None) -> Estimate:
    """L'historique de v est compté comme celui de v' (pratique de fait)."""
    rng = rng or np.random.default_rng()
    s = s_v + n11 + n01 + s_own
    f = f_v + n10 + n00 + f_own
    return _summarise(rng.beta(s + 0.5, f + 0.5, draws), "full_inheritance")


def double_sampling_bayes(s_v, f_v, n11, n10, n01, n00, s_own=0, f_own=0, draws=4000, rng=None,
                          prior="jeffreys", min_probes=20, guard=True, level=LEVEL) -> Estimate:
    """Méthode proposée (v0.3.1) : p' = pi*q1 + (1-pi)*q0 par tirages Monte Carlo.
    - les intervalles sont lus sur les quantiles des tirages (pas de Beta approchée) ;
    - les preuves propres sont intégrées par repondération ;
    - garde-fou : en dessous de min_probes, ou si la borne basse du gain <= 1, on revient au reset."""
    rng = rng or np.random.default_rng()
    m = n11 + n10 + n01 + n00
    n_v = s_v + f_v
    if n_v == 0:
        est = reset(n11, n10, n01, n00, s_own, f_own, draws, rng, prior)
        est.method, est.extra = "cold_start", dict(w_hist=0.0)
        return est
    info = borrowed_information(n11, n10, n01, n00, n_v, draws, rng)
    if guard and (m < min_probes or info["i_hist"] < 1.0):  # moins d'une pseudo-observation : rien à emprunter
        est = reset(n11, n10, n01, n00, s_own, f_own, draws, rng, prior)
        est.method, est.extra = "double_sampling(guard->reset)", dict(info, guard=True)
        return est

    def combine(pr):
        pi = rng.beta(s_v + 0.5, f_v + 0.5, draws)
        q1 = _beta_draws(n11, n10, draws, rng, pr)
        q0 = _beta_draws(n01, n00, draws, rng, pr)
        rng.shuffle(q1), rng.shuffle(q0)
        return _reweight(pi * q1 + (1 - pi) * q0, s_own, f_own, rng)

    if prior == "hybrid":  # moyenne par Jeffreys (précise), intervalle par Clopper-Pearson (prudent)
        point, band = combine("jeffreys"), combine("cp")
        a = (1 - level) / 2
        lo, hi = np.quantile(band, [a, 1 - a])
        return Estimate(float(point.mean()), float(lo), float(hi), "double_sampling[hybrid]", dict(info, guard=False))
    return _summarise(combine(prior), f"double_sampling[{prior}]", dict(info, guard=False), level)


def _log_beta_marginal(s, f, a=0.5):
    """log de la vraisemblance marginale d'un compte binomial sous un prior Beta(a, a)."""
    from scipy.special import betaln
    return betaln(s + a, f + a) - betaln(a, a)


def double_sampling_bma(s_v, f_v, n11, n10, n01, n00, s_own=0, f_own=0, draws=4000, rng=None,
                        prior_link=0.5, level=LEVEL) -> Estimate:
    """Méthode proposée (v0.3.2) : moyenne de modèles bayésienne, sans garde-fou ad hoc.
    M0 : v' ne dépend pas de v (q1 = q0 = q)  ->  p' = q, estimé sur toutes les probes
    M1 : v' dépend de v                       ->  p' = pi*q1 + (1-pi)*q0 (double échantillonnage)
    Les deux sont pondérés par leur vraisemblance marginale : l'historique n'est utilisé que si
    les probes montrent un lien. Moyenne par tirages Jeffreys, intervalle par tirages Clopper-Pearson."""
    rng = rng or np.random.default_rng()
    n_v = s_v + f_v
    s_all, f_all = n11 + n01, n10 + n00
    if n_v == 0:
        est = reset(n11, n10, n01, n00, s_own, f_own, draws, rng, "hybrid")
        est.method, est.extra = "cold_start", dict(w_hist=0.0, p_link=0.0)
        return est
    lm0 = _log_beta_marginal(s_all, f_all)
    lm1 = _log_beta_marginal(n11, n10) + _log_beta_marginal(n01, n00)
    lo0, lo1 = np.log(1 - prior_link) + lm0, np.log(prior_link) + lm1
    p_link = float(np.exp(lo1 - np.logaddexp(lo0, lo1)))
    k1 = int(round(draws * p_link))

    def mixture(pr):
        pi = rng.beta(s_v + 0.5, f_v + 0.5, k1)
        q1 = _beta_draws(n11, n10, k1, rng, pr) if k1 else np.empty(0)
        q0 = _beta_draws(n01, n00, k1, rng, pr) if k1 else np.empty(0)
        rng.shuffle(q1), rng.shuffle(q0)
        linked = pi * q1 + (1 - pi) * q0
        unlinked = _beta_draws(s_all, f_all, draws - k1, rng, pr) if draws - k1 else np.empty(0)
        return _reweight(np.concatenate([linked, unlinked]), s_own, f_own, rng)

    point, band = mixture("jeffreys"), mixture("cp")
    a = (1 - level) / 2
    lo, hi = np.quantile(band, [a, 1 - a])
    info = borrowed_information(n11, n10, n01, n00, n_v, draws, rng)
    info["i_hist"] *= p_link  # l'information empruntée ne compte que dans la mesure où le lien est établi
    info["w_hist"] = info["i_hist"] / n_v
    return Estimate(float(point.mean()), float(lo), float(hi), "double_sampling_bma", dict(info, p_link=p_link))


def double_sampling_freq(s_v, f_v, n11, n10, n01, n00) -> Estimate:
    """Baseline fréquentiste : post-stratification, variance à deux phases (Cochran), IC normal."""
    n_v = s_v + f_v
    w1 = s_v / n_v
    n1, n0 = n11 + n10, n01 + n00
    if n1 == 0 or n0 == 0:
        m = n1 + n0
        p = (n11 + n01) / m
        v = p * (1 - p) / m
    else:
        q1, q0 = n11 / n1, n01 / n0
        s1 = q1 * (1 - q1) * n1 / max(n1 - 1, 1)
        s0 = q0 * (1 - q0) * n0 / max(n0 - 1, 1)
        p = w1 * q1 + (1 - w1) * q0
        v = w1 ** 2 * s1 / n1 + (1 - w1) ** 2 * s0 / n0 + w1 * (1 - w1) * (q1 - q0) ** 2 / n_v
    sd = np.sqrt(max(v, 1e-8))
    return Estimate(p, p - 1.645 * sd, p + 1.645 * sd, "double_sampling_freq")
