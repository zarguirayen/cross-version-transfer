"""DÉVELOPPEMENT (synthétique) : détection séquentielle d'une substitution dégradée.
v' a un historique connu (N tâches) ; une nouvelle version v'' est sondée par lots de 10 probes (jusqu'à 80).
Arrêt dès que l'intervalle tranche :
  - alerte  si la borne haute < p(v') - marge_alerte   (dégradation probable)
  - feu vert si la borne basse > p(v') - delta_ok      (pas de dégradation notable)
Scénarios : dégradation de 10 points ; aucun changement mais bruit de ré-exécution (bascules aléatoires
symétriques à 5 %) ; aucun changement, sans bruit.
Mesures : taux d'alerte, nombre moyen de probes. Comparé aux budgets fixes 40, 60, 80."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import estimators as E  # noqa: E402

N, BATCH, MMAX = 500, 10, 80


def make(rng, p, scenario):
    y2 = rng.random(N) < p
    y3 = y2.copy()
    if scenario == "degradee_10pts":
        succ = np.flatnonzero(y2)
        y3[rng.choice(succ, int(0.10 * N), replace=False)] = False
    elif scenario == "bruit_5pc":
        flip = rng.random(N) < 0.05
        y3[flip] = ~y3[flip]
    return y2, y3


def estimate(y2, y3, idx, rng, level):
    a, b = y2[idx], y3[idx]
    f = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
    return E.double_sampling_bma(int(y2.sum()), int((~y2).sum()), *f, rng=rng, draws=1500, level=level)


def run(scenario, policy, reps=300, seed=0, level=0.90, margin=0.0, delta_ok=0.05):
    rng = np.random.default_rng(seed)
    alerts, used = [], []
    for rep in range(reps):
        w = np.random.default_rng([seed, rep])
        p = w.uniform(0.25, 0.85)
        y2, y3 = make(w, p, scenario)
        p2 = y2.mean()
        order = w.permutation(N)
        if policy == "sequentiel":
            alert, m = False, 0
            for m in range(BATCH, MMAX + 1, BATCH):
                e = estimate(y2, y3, order[:m], rng, level)
                if e.hi < p2 - margin:
                    alert = True
                    break
                if e.lo > p2 - delta_ok:
                    break
            alerts.append(alert); used.append(m)
        else:
            m = int(policy)
            e = estimate(y2, y3, order[:m], rng, level)
            alerts.append(e.hi < p2 - margin); used.append(m)
    return float(np.mean(alerts)), float(np.mean(used))


print(f"{'politique':28s} | {'dégradation 10 pts':>22s} | {'bruit 5 % (pas de dégr.)':>26s} | {'identique':>18s}")
for policy, level in (("40", 0.90), ("60", 0.90), ("80", 0.90), ("sequentiel", 0.90), ("sequentiel", 0.95)):
    res = [run(s, policy, level=level) for s in ("degradee_10pts", "bruit_5pc", "identique")]
    lab = f"{policy} (intervalle {int(level*100)} %)"
    print(f"{lab:28s} | alerte {res[0][0]:.2f}, {res[0][1]:4.1f} probes | alerte {res[1][0]:.2f}, {res[1][1]:4.1f} probes  | alerte {res[2][0]:.2f}, {res[2][1]:4.1f} pr.", flush=True)
