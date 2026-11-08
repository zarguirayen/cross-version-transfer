"""Synthetic tests of the predecessor-based estimator (``double_sampling_bma``).

Fast (under a minute), meant to be run after every change. Thresholds are slightly looser than the G1 validation
criteria because each test uses only 600 simulated worlds (standard error of a coverage of about 1.2 points)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import estimators as E  # noqa: E402
from cvt.simulate import Scenario, evaluate  # noqa: E402

REPS, M, N_V = 600, 20, 500


def gain_and_cov(sc, m=M):
    """Gain (MSE of random testing / MSE of ``ds_bma``) and coverage of ``ds_bma`` on scenario ``sc``."""
    base = evaluate(sc, "reset", m=m, n_v=N_V, reps=REPS, seed=3)
    r = evaluate(sc, "ds_bma", m=m, n_v=N_V, reps=REPS, seed=3)
    return base["mse"] / r["mse"], r["coverage"]


# ------------------------------------------------------------------ gain law
def test_gain_law_limits():
    """The finite-population law equals 1 at phi² = 0, n_v / m at phi² = 1, and 1/(1 - phi²) as n_v grows."""
    assert E.gain_law(0.0, 20, 500) == pytest.approx(1.0)
    assert E.gain_law(1.0, 20, 500) == pytest.approx(500 / 20)  # ceiling n_v / m
    assert E.gain_law(0.5, 20, 10**9) == pytest.approx(2.0, rel=1e-6)  # limit 1/(1-phi²)


# ------------------------------------------------------------------ scenarios
@pytest.mark.parametrize("sc", [
    Scenario("identiques", 0.70, 0.95, 0.05),
    Scenario("proches", 0.70, 0.90, 0.20),
    Scenario("inversees", 0.70, 0.10, 0.95),
])
def test_gain_follows_finite_law(sc):
    """With correlated versions, the observed gain is within 20% of the law and coverage stays above 0.86."""
    g, cov = gain_and_cov(sc)
    assert g == pytest.approx(E.gain_law(sc.phi2, M, N_V), rel=0.20)
    assert cov >= 0.86


@pytest.mark.parametrize("sc", [
    Scenario("independantes_p08", 0.70, 0.80, 0.80),
    Scenario("independantes_p05", 0.70, 0.50, 0.50),
    Scenario("v1_quasi_parfaite", 0.97, 0.80, 0.80),
])
def test_no_bonus_when_independent(sc):
    """With independent versions, the estimator loses at most 8% against random testing."""
    g, cov = gain_and_cov(sc)
    assert g >= 0.92          # no notable loss against random testing
    assert cov >= 0.86


def test_borrowed_information_zero_under_independence():
    """Under independence, the information borrowed from history (``i_hist``) is below one task in median."""
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(200):
        y1 = rng.random(N_V) < 0.7
        y2 = rng.random(N_V) < 0.8
        idx = rng.choice(N_V, M, replace=False)
        a, b = y1[idx], y2[idx]
        f = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
        vals.append(E.double_sampling_bma(int(y1.sum()), int((~y1).sum()), *f, rng=rng).extra["i_hist"])
    assert np.median(vals) < 1.0


def test_rare_failures_coverage():
    """Coverage holds when failures are rare and the versions are close."""
    _, cov = gain_and_cov(Scenario("echecs_rares_proches", 0.95, 0.90, 0.30))
    assert cov >= 0.86


def test_inverted_versions_have_negative_phi():
    """The signed phi coefficient is strongly negative when the new version fails where the old one succeeds."""
    assert E.phi_signed(1, 9, 9, 1) < -0.5


def test_degraded_version_prediction_drops():
    """The estimate falls when the probes show a regression."""
    # v solves 80%; on the probes, v' loses most of the successes of v
    est = E.double_sampling_bma(400, 100, n11=4, n10=12, n01=1, n00=3, rng=np.random.default_rng(1))
    assert est.mean < 0.5


def test_improved_version_prediction_rises():
    """The estimate rises when the probes show an improvement."""
    # v solves 60%; v' keeps its successes and fixes most of its failures
    est = E.double_sampling_bma(300, 200, n11=12, n10=0, n01=6, n00=2, rng=np.random.default_rng(1))
    assert est.mean > 0.8


def test_few_probes_wide_interval():
    """Five probes give a wide interval."""
    est = E.double_sampling_bma(350, 150, n11=3, n10=1, n01=0, n00=1, rng=np.random.default_rng(1))
    assert est.hi - est.lo > 0.25


def test_cold_start():
    """Without history, the estimator falls back to the probes alone and gives the history no weight."""
    est = E.double_sampling_bma(0, 0, 8, 0, 4, 0, rng=np.random.default_rng(1))  # no history
    assert est.method == "cold_start"
    assert est.extra["w_hist"] == 0.0


def test_own_evidence_moves_belief():
    """Extra results of the new version itself (30 failures) pull the estimate down."""
    rng = np.random.default_rng(2)
    a = E.double_sampling_bma(350, 150, 12, 2, 2, 4, rng=rng)
    b = E.double_sampling_bma(350, 150, 12, 2, 2, 4, s_own=0, f_own=30, rng=rng)
    assert b.mean < a.mean - 0.1
