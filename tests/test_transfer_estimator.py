"""Suite synthétique de l'estimateur de transfert entre versions (méthode : double_sampling_bma).
Rapide (< 1 min) : à lancer à chaque modification. Seuils volontairement un peu plus larges que G1,
car chaque test n'utilise que 600 mondes (erreur type d'une couverture ≈ 1,2 point)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import estimators as E  # noqa: E402
from cvt.simulate import Scenario, evaluate  # noqa: E402

REPS, M, N_V = 600, 20, 500


def gain_and_cov(sc, m=M):
    base = evaluate(sc, "reset", m=m, n_v=N_V, reps=REPS, seed=3)
    r = evaluate(sc, "ds_bma", m=m, n_v=N_V, reps=REPS, seed=3)
    return base["mse"] / r["mse"], r["coverage"]


# ------------------------------------------------------------------ loi de gain
def test_gain_law_limits():
    assert E.gain_law(0.0, 20, 500) == pytest.approx(1.0)
    assert E.gain_law(1.0, 20, 500) == pytest.approx(500 / 20)  # plafond n_v / m
    assert E.gain_law(0.5, 20, 10**9) == pytest.approx(2.0, rel=1e-6)  # limite 1/(1-phi²)


# ------------------------------------------------------------------ scénarios
@pytest.mark.parametrize("sc", [
    Scenario("identiques", 0.70, 0.95, 0.05),
    Scenario("proches", 0.70, 0.90, 0.20),
    Scenario("inversees", 0.70, 0.10, 0.95),
])
def test_gain_follows_finite_law(sc):
    g, cov = gain_and_cov(sc)
    assert g == pytest.approx(E.gain_law(sc.phi2, M, N_V), rel=0.20)
    assert cov >= 0.86


@pytest.mark.parametrize("sc", [
    Scenario("independantes_p08", 0.70, 0.80, 0.80),
    Scenario("independantes_p05", 0.70, 0.50, 0.50),
    Scenario("v1_quasi_parfaite", 0.97, 0.80, 0.80),
])
def test_no_bonus_when_independent(sc):
    g, cov = gain_and_cov(sc)
    assert g >= 0.92          # pas de perte notable face au reset
    assert cov >= 0.86


def test_borrowed_information_zero_under_independence():
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
    _, cov = gain_and_cov(Scenario("echecs_rares_proches", 0.95, 0.90, 0.30))
    assert cov >= 0.86


def test_inverted_versions_have_negative_phi():
    assert E.phi_signed(1, 9, 9, 1) < -0.5


def test_degraded_version_prediction_drops():
    # v réussit 80 % ; sur les probes, v' perd la plupart des succès de v
    est = E.double_sampling_bma(400, 100, n11=4, n10=12, n01=1, n00=3, rng=np.random.default_rng(1))
    assert est.mean < 0.5


def test_improved_version_prediction_rises():
    # v réussit 60 % ; v' garde ses succès et corrige la plupart de ses échecs
    est = E.double_sampling_bma(300, 200, n11=12, n10=0, n01=6, n00=2, rng=np.random.default_rng(1))
    assert est.mean > 0.8


def test_few_probes_wide_interval():
    est = E.double_sampling_bma(350, 150, n11=3, n10=1, n01=0, n00=1, rng=np.random.default_rng(1))
    assert est.hi - est.lo > 0.25


def test_cold_start():
    est = E.double_sampling_bma(0, 0, 8, 0, 4, 0, rng=np.random.default_rng(1))  # pas d'historique
    assert est.method == "cold_start"
    assert est.extra["w_hist"] == 0.0


def test_own_evidence_moves_belief():
    rng = np.random.default_rng(2)
    a = E.double_sampling_bma(350, 150, 12, 2, 2, 4, rng=rng)
    b = E.double_sampling_bma(350, 150, 12, 2, 2, 4, s_own=0, f_own=30, rng=rng)
    assert b.mean < a.mean - 0.1
