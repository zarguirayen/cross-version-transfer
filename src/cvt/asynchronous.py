"""Asynchronous evaluation: estimating a new version's success rate at a deadline, before every probe has returned.

When the probes of a new version run in parallel, their results arrive in an order that is not random: tasks on
which the agent takes many steps finish last, and on agent benchmarks those slow tasks are mostly failures. An
estimate computed on the probes that have returned by a deadline is therefore biased upwards.

The graph of earlier agents knows how outcome relates to slowness. ``fit_slow_model`` fits, on the earlier agents,

    logit P(y_aj = 1) = theta_a + b_j + gamma * slow_aj,

where slow_aj = 1 when agent a used more steps on task j than on half of the tasks. ``deadline_estimate`` then
estimates the new version's ability from its returned (fast) probes, imputes each unreturned (slow) probe by
P = sigmoid(theta + b_j + gamma), and combines returned and imputed outcomes with the stratum weights of the design.
The interval adds to the design variance the uncertainty of the imputed outcomes, of the ability and of gamma (the
spread of per-agent slowness effects in the graph).

Development: ``experiments/dev_v8_async.py`` (the 14 agents of the AppWorld paper). Confirmation: study v8.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .estimators import Estimate


def _sig(z):
    """Logistic function."""
    return 1 / (1 + np.exp(-z))


def fit_slow_model(Y: np.ndarray, S: np.ndarray, iters: int = 400, pen: float = 0.01):
    """Fit the slowness model on the earlier agents of the graph.

    Parameters
    ----------
    Y : ndarray (agents, tasks)
        Outcomes of the earlier agents (0/1).
    S : ndarray (agents, tasks)
        Number of steps each earlier agent used on each task.
    iters : int
        Gradient steps for the joint fit.
    pen : float
        Ridge penalty on all parameters.

    Returns
    -------
    b : ndarray (tasks,)
        Task easiness.
    theta : ndarray (agents,)
        Abilities of the earlier agents.
    gamma : float
        Pooled slowness effect (negative when slow tasks fail more often).
    gamma_a : ndarray (agents,)
        Per-agent slowness effects (with ``b`` and ``theta`` fixed), whose spread measures how well gamma transfers.
    """
    Y = np.asarray(Y, float)
    A, N = Y.shape
    Sl = (S > np.median(S, axis=1, keepdims=True)).astype(float)
    th, b, g = np.zeros(A), np.zeros(N), 0.0
    for _ in range(iters):
        R = Y - _sig(th[:, None] + b[None, :] + g * Sl)
        th += 0.5 * (R.mean(1) - pen * th)
        b += 0.5 * (R.mean(0) - pen * b)
        g += 0.5 * ((R * Sl).sum() / max(Sl.sum(), 1.0) - pen * g)
    ga = np.zeros(A)
    for _ in range(200):
        R = Y - _sig(th[:, None] + b[None, :] + ga[:, None] * Sl)
        ga += ((R * Sl).sum(1) / np.maximum(Sl.sum(1), 1.0) - pen * ga)
    return b, th, float(g), ga


def _fit_ability(b_ret, y_ret, prior_mu, prior_sd=2.0):
    """MAP ability of the new version from its returned probes (offsets b_ret), Gaussian prior; returns (theta, se)."""
    th = float(prior_mu)
    hess = -1.0 / prior_sd ** 2
    for _ in range(50):
        p = _sig(th + b_ret)
        grad = (y_ret - p).sum() - (th - prior_mu) / prior_sd ** 2
        hess = -(p * (1 - p)).sum() - 1 / prior_sd ** 2
        th -= grad / hess
    return th, float(np.sqrt(-1 / hess))


def deadline_estimate(idx, returned, y_probe, weights, b, theta_graph, gamma, gamma_a, N, level=0.90) -> Estimate:
    """Estimate the success rate at a deadline, with the graph's slowness correction.

    Parameters
    ----------
    idx : ndarray (m,)
        Indices of the probes (one per stratum of the stratified design, in stratum order).
    returned : ndarray of bool (m,)
        Which probes have returned by the deadline.
    y_probe : ndarray (m,)
        Outcomes of the probes; only the returned ones are read.
    weights : ndarray (m,)
        Stratum sizes.
    b, theta_graph, gamma, gamma_a
        Output of :func:`fit_slow_model` on the earlier agents.
    N : int
        Number of tasks in the suite.
    level : float
        Confidence level of the interval.

    Returns
    -------
    Estimate
        ``method = "deadline_graph"``; ``extra`` holds the ability and the number of imputed probes.
    """
    idx, returned, w = np.asarray(idx), np.asarray(returned, bool), np.asarray(weights, float)
    bj = b[idx]
    th, th_se = _fit_ability(bj[returned], np.asarray(y_probe, float)[returned], float(np.mean(theta_graph)))
    y = np.asarray(y_probe, float).copy()
    pu = _sig(th + bj[~returned] + gamma)
    y[~returned] = pu
    mean = float((w * y).sum() / N)
    m = len(idx)
    v_design = sum((w[i] * y[i] - w[i + 1] * y[i + 1]) ** 2 for i in range(0, m - 1, 2)) / N ** 2
    v_imp = float((w[~returned] ** 2 * pu * (1 - pu)).sum()) / N ** 2
    d = float((w[~returned] * pu * (1 - pu)).sum()) / N          # d mean / d (theta or gamma)
    ga = np.asarray(gamma_a, float)
    ga = ga[np.isfinite(ga)]
    v_g = d ** 2 * (np.var(ga, ddof=1) if len(ga) > 1 else 0.0)
    v_th = d ** 2 * th_se ** 2
    p_ = min(max(mean, 1e-4), 1 - 1e-4)
    var = max(v_design + v_imp + v_g + v_th, 0.25 * p_ * (1 - p_) / m)
    n_eff = p_ * (1 - p_) / var
    a = (1 - level) / 2
    lo = float(stats.beta.ppf(a, p_ * n_eff + 0.5, (1 - p_) * n_eff + 0.5))
    hi = float(stats.beta.ppf(1 - a, p_ * n_eff + 0.5, (1 - p_) * n_eff + 0.5))
    return Estimate(mean, lo, hi, "deadline_graph", {"theta": th, "n_imputed": int((~returned).sum())})
