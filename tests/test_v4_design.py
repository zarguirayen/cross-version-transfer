"""Tests du plan stratifié et de l'estimateur assisté par modèle (GREG)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import graph as Gr  # noqa: E402


def _world(seed, base=-0.5, N=300):
    g = np.random.default_rng(seed)
    d = g.normal(0, 1.5, N)
    y = g.random(N) < 1 / (1 + np.exp(-(base + 1.2 * d)))
    return d, y


def _strat(score, m, rng):
    blocks = np.array_split(np.argsort(score, kind="stable"), m)
    return np.array([rng.choice(b) for b in blocks]), np.array([len(b) for b in blocks], float), blocks


def test_stratified_law_matches_monte_carlo():
    d, y = _world(1)
    m, N, rng = 20, len(y), np.random.default_rng(2)
    y = y.astype(float)
    srs = [y[rng.choice(N, m, replace=False)].mean() for _ in range(20000)]
    st = [y[_strat(d, m, rng)[0]].mean() for _ in range(20000)]
    blocks = _strat(d, m, rng)[2]
    G = ((1 - m / N) * y.var(ddof=1) / m) / (sum((1 - 1 / len(b)) * y[b].var(ddof=1) for b in blocks) / m ** 2)
    obs = np.var(srs) / np.var(st)
    assert abs(obs / G - 1) < 0.08


def test_greg_unbiased_under_misspecified_model_at_low_rate():
    """Écosystème trompeur (covariable sans lien) et taux bas : GREG reste sans biais sous le plan."""
    d, y = _world(3, base=-3.5)
    g = np.random.default_rng(4)
    X = np.column_stack([g.normal(size=len(y)), np.zeros(len(y))])
    ests = []
    for _ in range(600):
        idx, w, _ = _strat(X[:, 0], 20, g)
        ests.append(Gr.graph_greg(X, idx, y[idx], weights=w, rng=g, draws=400).mean)
    assert abs(np.mean(ests) - y.mean()) < 0.01


def test_greg_cf_covers_zero_rate_and_is_calibrated():
    """GREG croisé : borne basse 0 quand aucune probe ne réussit ; couverture proche de 90 % en régime normal."""
    g = np.random.default_rng(11)
    d, y = _world(5, base=-0.3)
    X = np.column_stack([d + g.normal(0, 0.5, len(d)), np.zeros(len(d))])
    cov = []
    for _ in range(300):
        idx, w, _ = _strat(X[:, 0], 20, g)
        e = Gr.graph_greg_cf(X, idx, y[idx], w, rng=g, draws=300)
        cov.append(e.lo <= y.mean() <= e.hi)
    assert 0.85 <= np.mean(cov) <= 0.99
    y0 = np.zeros(len(d), bool)
    idx, w, _ = _strat(X[:, 0], 20, g)
    e = Gr.graph_greg_cf(X, idx, y0[idx], w, rng=g, draws=300)
    assert e.lo == 0.0 and e.mean <= 0.05
