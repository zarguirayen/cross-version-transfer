"""DEVELOPMENT for v8 (AppWorld, the 14 agents of the AppWorld paper: outcomes already seen in v2, step counts never
analysed before), not evidence.

Question: when probe results arrive asynchronously (in the order of the new agent's own number of steps) and the
evaluation stops at a deadline (k of the m = 20 stratified probes returned), how biased is the naive estimate, and
does the graph's slowness model (``cvt.asynchronous``) correct it with calibrated intervals?
Graph of each agent: the other paper agents that do not use its model (the v2 rule).

Usage
-----
    python experiments/dev_v8_async.py [R]     # writes results/dev_v8_async.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import asynchronous as A  # noqa: E402
from cvt import graph as Gr  # noqa: E402
import v8_data as D  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 60
M, KS = 20, (10, 15)
OUT = Path(__file__).resolve().parents[1] / "results" / "dev_v8_async.csv"


def llm(agent):
    """Model of a paper agent (suffix of its name)."""
    return agent.rsplit("_", 1)[1]


def main():
    """Replay each paper agent as a new version; record signed bias, squared error and coverage per deadline."""
    rows = []
    for split in D.SPLITS:
        ids, Y, S = D.load_split(split)
        old = [a for a in Y if a in D.OLD_AGENTS]
        for a in old:
            eco = [w for w in old if w != a and llm(w) != llm(a)]
            Ye, Se = np.array([Y[w] for w in eco]), np.array([S[w] for w in eco], float)
            b, th, g, ga = A.fit_slow_model(Ye, Se)
            X = Gr.ecosystem_features(Ye, np.zeros(len(ids), bool))
            y, s, N = Y[a], S[a].astype(float), len(ids)
            p = float(y.mean())
            rng = np.random.default_rng(2)
            acc = {k: {"naive": [], "graph": [], "cov": []} for k in KS}
            for _ in range(R):
                blocks = np.array_split(np.argsort(X[:, 0], kind="stable"), M)
                idx = np.array([rng.choice(bb) for bb in blocks]); w = np.array([len(bb) for bb in blocks], float)
                order = np.lexsort((rng.random(M), s[idx]))           # arrival order: fewest steps first
                for k in KS:
                    ret = np.zeros(M, bool); ret[order[:k]] = True
                    e = A.deadline_estimate(idx, ret, y[idx], w, b, th, g, ga, N)
                    acc[k]["naive"].append(y[idx][ret].mean() - p); acc[k]["graph"].append(e.mean - p)
                    acc[k]["cov"].append(e.lo <= p <= e.hi)
            for k in KS:
                nv, gr = np.array(acc[k]["naive"]), np.array(acc[k]["graph"])
                rows.append(dict(split=split, agent=a, k=k, p=p, gamma=g, bias_naive=nv.mean(), bias_graph=gr.mean(),
                                 rmse_naive=np.sqrt((nv ** 2).mean()), rmse_graph=np.sqrt((gr ** 2).mean()),
                                 cov_graph=np.mean(acc[k]["cov"])))
    with OUT.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    for k in KS:
        S_ = [r for r in rows if r["k"] == k]
        f = lambda key: np.array([r[key] for r in S_])
        print(f"k={k}: bias naive mean {f('bias_naive').mean():+.3f} | graph {f('bias_graph').mean():+.3f} ; "
              f"median RMSE ratio naive/graph {np.median(f('rmse_naive') / f('rmse_graph')):.2f} ; "
              f"coverage graph mean {f('cov_graph').mean():.3f} (min {f('cov_graph').min():.3f}) ; "
              f"gamma mean {f('gamma').mean():+.2f}")


if __name__ == "__main__":
    main()
