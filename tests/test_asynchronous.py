"""Tests of the deadline estimator (cvt.asynchronous)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import asynchronous as A  # noqa: E402


def _world(seed=0, A_=12, N=300, gamma=-1.0):
    """Earlier agents and a new agent whose slow tasks fail more often (true slowness effect ``gamma``)."""
    g = np.random.default_rng(seed)
    b = g.normal(0, 1.5, N); th = g.normal(-0.3, 1.0, A_ + 1)
    S = g.gamma(2.0, 10.0, (A_ + 1, N)) * np.exp(-0.4 * b)[None, :]      # hard tasks take more steps
    slow = S > np.median(S, axis=1, keepdims=True)
    P = 1 / (1 + np.exp(-(th[:, None] + b[None, :] + gamma * slow)))
    Y = g.random((A_ + 1, N)) < P
    return Y[:-1], S[:-1], Y[-1], S[-1]


def test_all_returned_equals_stratified_mean():
    """With every probe returned, the estimate is the stratum-weighted mean of the probes."""
    Ye, Se, y, _ = _world()
    b, th, g, ga = A.fit_slow_model(Ye, Se)
    idx = np.arange(0, 300, 15); w = np.full(20, 15.0)
    e = A.deadline_estimate(idx, np.ones(20, bool), y[idx], w, b, th, g, ga, 300)
    assert abs(e.mean - (w * y[idx]).sum() / 300) < 1e-12 and e.lo <= e.mean <= e.hi


def test_slowness_effect_is_recovered_and_corrects_bias():
    """The pooled effect is negative when slow tasks fail; at a deadline the correction beats the naive mean."""
    Ye, Se, y, s = _world(seed=1)
    b, th, g, ga = A.fit_slow_model(Ye, Se)
    assert g < -0.4
    rng = np.random.default_rng(3); naive, corr = [], []
    order_score = np.argsort(b)                                           # strata on task easiness
    for _ in range(200):
        blocks = np.array_split(order_score, 20)
        idx = np.array([rng.choice(bl) for bl in blocks]); w = np.array([len(bl) for bl in blocks], float)
        ret = np.zeros(20, bool); ret[np.argsort(s[idx])[:10]] = True
        naive.append(y[idx][ret].mean() - y.mean())
        corr.append(A.deadline_estimate(idx, ret, y[idx], w, b, th, g, ga, len(y)).mean - y.mean())
    assert np.mean(naive) > 0.03 and abs(np.mean(corr)) < abs(np.mean(naive))
