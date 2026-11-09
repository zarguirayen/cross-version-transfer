"""DEVELOPMENT for a prospective v9 (data already seen: the 26 v8 evaluations of new AppWorld agents), not evidence.

The v8 deadline correction transferred a slowness effect learned from all earlier agents; the paper's agents fail
where they are slow, newer agents do not. Candidates compared here, at the v8 deadline (k = 10 of m = 20):
  naive     mean of the returned probes;
  v8        slowness effect pooled over all earlier agents (frozen in v8);
  zero      no slowness effect (difficulty only);
  recent    slowness effect of the 5 most recently released earlier agents;
  hedge     point estimate of `recent`, interval = union of the `recent` and `zero` intervals.

Usage
-----
    python experiments/dev_v9_deadline.py [R]     # writes results/dev_v9_deadline.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import asynchronous as A  # noqa: E402
from cvt import graph as Gr  # noqa: E402
import v8_data as D  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
M, K, RECENT = 20, 10, 5
OUT = HERE.parent / "results" / "dev_v9_deadline.csv"


def main():
    """Replay each new v8 agent with the same design and arrival rule as v8; compare the candidates."""
    agents = D.all_agents(); rows = []
    for split in D.SPLITS:
        ids, Y, S = D.load_split(split)
        for kc, (a, (date, pat, parent, new)) in enumerate(agents.items()):
            if not new:
                continue
            eco = sorted((w for w, (d, _, _, _) in agents.items() if d < date), key=lambda w: agents[w][0])
            Ye, Se = np.array([Y[w] for w in eco]), np.array([S[w] for w in eco], float)
            X = Gr.ecosystem_features(Ye, Y[parent] if parent else np.zeros(len(ids), bool))
            b, th, g, ga = A.fit_slow_model(Ye, Se)
            rec = list(range(len(eco)))[-RECENT:]
            g_rec, ga_rec = float(np.mean(ga[rec])), ga[rec]
            y, s, N = Y[a], S[a].astype(float), len(ids); p = float(y.mean())
            rng = np.random.default_rng([20261125, kc, len(split)])
            acc = {c: [] for c in ("naive", "v8", "zero", "recent", "hedge")}; cov = {c: [] for c in acc}
            for _ in range(R):
                blocks = np.array_split(np.argsort(X[:, 0], kind="stable"), M)
                idx = np.array([rng.choice(bb) for bb in blocks]); w = np.array([len(bb) for bb in blocks], float)
                ret = np.zeros(M, bool); ret[np.lexsort((rng.random(M), s[idx]))[:K]] = True
                acc["naive"].append(y[idx][ret].mean() - p); cov["naive"].append(np.nan)
                e8 = A.deadline_estimate(idx, ret, y[idx], w, b, th, g, ga, N)
                e0 = A.deadline_estimate(idx, ret, y[idx], w, b, th, 0.0, np.zeros(2), N)
                er = A.deadline_estimate(idx, ret, y[idx], w, b, th, g_rec, ga_rec, N)
                for c, e in (("v8", e8), ("zero", e0), ("recent", er)):
                    acc[c].append(e.mean - p); cov[c].append(e.lo <= p <= e.hi)
                acc["hedge"].append(er.mean - p); cov["hedge"].append(min(er.lo, e0.lo) <= p <= max(er.hi, e0.hi))
            row = dict(split=split, agent=a, pattern=pat, p=p, gamma_all=g, gamma_recent=g_rec)
            for c in acc:
                v = np.array(acc[c]); row[f"bias_{c}"] = v.mean(); row[f"rmse_{c}"] = float(np.sqrt((v ** 2).mean()))
                row[f"cov_{c}"] = float(np.nanmean(cov[c])) if c != "naive" else np.nan
            rows.append(row)
    with OUT.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    f = lambda k: np.array([r[k] for r in rows])
    for c in ("naive", "v8", "zero", "recent", "hedge"):
        cv = f(f"cov_{c}")
        print(f"{c:7s} bias mean {f(f'bias_{c}').mean():+.3f} | RMSE ratio naive/{c} median "
              f"{np.median(f('rmse_naive') / f(f'rmse_{c}')):.2f} | coverage mean {np.nanmean(cv):.3f} min {np.nanmin(cv) if c != 'naive' else float('nan'):.3f}")


if __name__ == "__main__":
    main()
