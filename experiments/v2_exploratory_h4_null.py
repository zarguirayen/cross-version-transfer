"""EXPLORATORY (after the v2 freeze): false alerts of sequential detection under re-run noise that does NOT change
the mean rate (as many success -> failure flips as failure -> success).

Reason: the pre-registered symmetric noise raises the rate of agents below 50%, which makes the false-alert control
too easy. Same detection rule as v2, same agents (rate >= 0.20)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import estimators as E  # noqa: E402

ROOT = Path(sys.argv[1]) / "experiments" / "outputs"
rng = np.random.default_rng(20261112)


def flips(a, b):
    """Contingency counts (n11, n10, n01, n00) of old outcomes ``a`` against new outcomes ``b`` on the probes."""
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def detect(y, y3):
    """Sequential detection in batches of 10 probes (up to 80): return ``(alert, probes used)``."""
    n, order = len(y), rng.permutation(len(y))
    for m in range(10, 81, 10):
        idx = order[:m]
        e = E.double_sampling_bma(int(y.sum()), int((~y).sum()), *flips(y[idx], y3[idx]), rng=rng, draws=1500, level=0.95)
        if e.hi < y.mean():
            return True, m
        if e.lo > y.mean() - 0.08:
            return False, m
    return False, 80


for rate in (0.05, 0.10):
    alerts, used = [], []
    for split in ("test_normal", "test_challenge"):
        for p in sorted(ROOT.glob(f"*_{split}")):
            d = json.load((p / "evaluations" / f"{split}.json").open())["individual"]
            y = np.array([bool(d[t]["success"]) for t in sorted(d)])
            if y.mean() < 0.20:
                continue
            k = int(round(rate * len(y) / 2))
            for _ in range(200):
                y3 = y.copy()
                y3[rng.choice(np.flatnonzero(y), k, replace=False)] = False
                y3[rng.choice(np.flatnonzero(~y), k, replace=False)] = True
                a, m = detect(y, y3)
                alerts.append(a); used.append(m)
    print(f"bruit à taux constant, {int(rate*100)} % des tâches changent : fausses alertes {np.mean(alerts):.3f}, probes moyennes {np.mean(used):.1f}")
