"""DÉVELOPPEMENT v7 (AppWorld, données de développement) : y a-t-il une structure dans le graphe une fois la
difficulté des tâches retirée ? Pour chaque agent, résidu tâche par tâche = résultat − probabilité prédite par un
modèle à une dimension (logit P = capacité de l'agent + facilité de la tâche, ajusté sur tous les agents).
On compare la corrélation des résidus entre paires d'agents selon leur type de lien : même modèle, même
échafaudage, ni l'un ni l'autre. Si le graphe n'était qu'une échelle de difficulté, toutes ces corrélations
seraient proches de 0."""
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
    """Modèle à une dimension ajusté par maximum de vraisemblance pénalisé (descente de gradient)."""
    A, N = Y.shape
    th, b = np.zeros(A), np.zeros(N)
    for _ in range(iters):
        P = 1 / (1 + np.exp(-(th[:, None] + b[None, :])))
        R = Y - P
        th += 0.5 * (R.mean(axis=1) - 0.01 * th)
        b += 0.5 * (R.mean(axis=0) - 0.01 * b)
    return 1 / (1 + np.exp(-(th[:, None] + b[None, :])))


def scaffold(a):
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
