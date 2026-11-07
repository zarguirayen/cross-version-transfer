"""Simulateur : scénarios synthétiques de transition v -> v', tirage des probes, évaluation des estimateurs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import estimators as E


@dataclass(frozen=True)
class Scenario:
    name: str
    pi: float   # taux de réussite de v
    q1: float   # P(v' réussit | v a réussi)
    q0: float   # P(v' réussit | v a échoué)

    @property
    def p_true(self) -> float:
        return self.pi * self.q1 + (1 - self.pi) * self.q0

    @property
    def phi2(self) -> float:
        p = self.p_true
        return (self.pi * (1 - self.pi) * (self.q1 - self.q0)) ** 2 / (self.pi * (1 - self.pi) * p * (1 - p))


SCENARIOS = [
    Scenario("identiques", 0.70, 0.95, 0.05),
    Scenario("proches", 0.70, 0.90, 0.20),
    Scenario("regression", 0.70, 0.60, 0.10),
    Scenario("amelioree", 0.70, 0.97, 0.70),
    Scenario("inversees", 0.70, 0.10, 0.95),
    Scenario("independantes_p08", 0.70, 0.80, 0.80),
    Scenario("independantes_p05", 0.70, 0.50, 0.50),
    Scenario("v1_quasi_parfaite", 0.97, 0.80, 0.80),
    Scenario("echecs_rares_proches", 0.95, 0.90, 0.30),
]


def draw_world(sc: Scenario, n_v: int, rng):
    y1 = rng.random(n_v) < sc.pi
    y2 = np.where(y1, rng.random(n_v) < sc.q1, rng.random(n_v) < sc.q0)
    return y1, y2


def pick_probes(y1, m, rng, allocation="proportional", min_per_stratum=5):
    """proportional : tirage aléatoire simple dans l'historique.
    floor       : comme proportional, mais au moins min_per_stratum probes dans chaque strate de v
                  (quand elle existe). L'estimateur post-stratifié reste sans biais car les poids
                  des strates viennent de l'historique, pas des probes."""
    n_v = len(y1)
    if allocation == "proportional":
        return rng.choice(n_v, m, replace=False)
    if allocation == "floor":
        s, f = np.flatnonzero(y1), np.flatnonzero(~y1)
        k0 = min(len(f), max(min_per_stratum, int(round(m * len(f) / n_v))))
        k1 = min(len(s), m - k0)
        k0 = m - k1
        return np.concatenate([rng.choice(s, k1, replace=False), rng.choice(f, k0, replace=False)])
    raise ValueError(allocation)


ADAPTIVE_MIN_SHARE = 0.10  # règle v0.3.1 : plancher seulement si la plus petite strate pèse au moins 10 %


def resolve_allocation(y1, allocation):
    """'adaptive' -> 'floor' si la plus petite strate de v pèse >= 10 % de l'historique, sinon 'proportional'."""
    if allocation != "adaptive":
        return allocation
    share = y1.mean()
    return "floor" if min(share, 1 - share) >= ADAPTIVE_MIN_SHARE else "proportional"


def flips(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def evaluate(sc: Scenario, method: str, m=20, n_v=500, reps=1500, seed=0, draws=2000,
             allocation="proportional", prior="jeffreys", level=E.LEVEL, guard=None):
    """Renvoie l'erreur quadratique moyenne, la couverture et la largeur moyenne de l'intervalle."""
    rng = np.random.default_rng(seed + 10_000)  # tirages Monte Carlo propres à l'estimateur
    err, cover, width, guards = [], [], [], []
    for rep in range(reps):
        world = np.random.default_rng([seed, rep])  # même monde au même rang pour toutes les méthodes
        y1, y2 = draw_world(sc, n_v, world)
        # le reset tire ses probes au hasard dans le flux (sans biais) ; les autres suivent l'allocation
        alloc = "proportional" if method == "reset" else resolve_allocation(y1, allocation)
        idx = pick_probes(y1, m, world, alloc)
        f = flips(y1[idx], y2[idx])
        s_v = int(y1.sum())
        f_v = n_v - s_v
        if method == "reset":
            est = E.reset(*f, draws=draws, rng=rng, prior=prior)
        elif method == "full":
            est = E.full_inheritance(s_v, f_v, *f, draws=draws, rng=rng)
        elif method == "ds_bayes":
            g = (alloc == "proportional") if guard is None else guard  # garde-fou seulement si le reset de repli est sans biais
            est = E.double_sampling_bayes(s_v, f_v, *f, draws=draws, rng=rng, prior=prior, level=level, guard=g)
            guards.append(est.extra.get("guard", False))
        elif method == "ds_bma":
            est = E.double_sampling_bma(s_v, f_v, *f, draws=draws, rng=rng, level=level)
            guards.append(est.extra.get("p_link", 0.0))
        elif method == "ds_freq":
            est = E.double_sampling_freq(s_v, f_v, *f)
        else:
            raise ValueError(method)
        p = sc.p_true
        err.append((est.mean - p) ** 2)
        cover.append(est.lo <= p <= est.hi)
        width.append(est.hi - est.lo)
    return dict(mse=float(np.mean(err)), coverage=float(np.mean(cover)), width=float(np.mean(width)),
                guard_rate=float(np.mean(guards)) if guards else None)
