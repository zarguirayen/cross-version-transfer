"""DEVELOPMENT for v7 (AppWorld, development data): is there structure in the graph once task difficulty is removed?

For each agent, the task-level residual is outcome minus the probability predicted by a one-dimensional model
(logit P = agent ability + task easiness, fitted on all agents). Residual correlations are compared across pairs of
agents by type of link: same model, same scaffold, neither. If the graph were only a difficulty scale, all these
correlations would be close to 0."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
import importlib.util  # noqa: E402

sys.argv = [sys.argv[0], "/home/claude/data/aw_root"]
spec = importlib.util.spec_from_file_location("v2", HERE / "v2_appworld_preregistered.py")
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)


def rasch(Y, iters=300):
    """Fit the one-dimensional model by penalised maximum likelihood (gradient ascent, ridge penalty 0.01).

    Parameters
    ----------
    Y : ndarray (agents, tasks)
        Binary outcomes.
    iters : int
        Number of gradient steps.

    Returns
    -------
    ndarray (agents, tasks)
        Fitted success probabilities.
    """
    A, N = Y.shape
    th, b = np.zeros(A), np.zeros(N)
    for _ in range(iters):
        P = 1 / (1 + np.exp(-(th[:, None] + b[None, :])))
        R = Y - P
        th += 0.5 * (R.mean(axis=1) - 0.01 * th)
        b += 0.5 * (R.mean(axis=0) - 0.01 * b)
    return 1 / (1 + np.exp(-(th[:, None] + b[None, :])))


def scaffold(a):
    """Scaffold of AppWorld agent ``a`` (the method prefix of its name)."""
    return next(m for m in v2.METHODS if a.startswith(m))


for split in v2.SPLITS:
    av = v2.agents_available(split)
    Y = np.array([v2.load(a, split)[1] for a in av], float)
    P = rasch(Y)
    Rz = (Y - P) / np.sqrt(P * (1 - P))
    raw = np.corrcoef(Y); res = np.corrcoef(Rz)
    groups = {"même modèle": [], "même échafaudage": [], "aucun lien": []}
    for i, j in combinations(range(len(av)), 2):
        sm, ss = v2.llm_of(av[i]) == v2.llm_of(av[j]), scaffold(av[i]) == scaffold(av[j])
        key = "même modèle" if sm else ("même échafaudage" if ss else "aucun lien")
        groups[key].append((raw[i, j], res[i, j]))
    var1 = 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum()
    print(f"\n{split} : {len(av)} agents, {Y.shape[1]} tâches ; part de variance expliquée par la difficulté (1 dimension) : {var1:.2f}")
    for k, v in groups.items():
        v = np.array(v)
        print(f"  {k:18s} n={len(v):3d} corrélation brute {v[:, 0].mean():.3f} | corrélation des résidus {v[:, 1].mean():+.3f}")
