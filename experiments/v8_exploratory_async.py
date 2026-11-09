"""EXPLORATORY (after the v8 freeze, not pre-registered): why did the graph's deadline correction fail (A2, A3)?

Compares, for each new AppWorld agent, the slowness effect in its own data (fitted with the task easiness of its
graph held fixed) with the effect the graph transferred to it, and relates the mismatch to the estimator's bias.
Reads results/v8/new_agents.csv; writes results/v8/exploratory_async.csv.

Usage
-----
    python experiments/v8_exploratory_async.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import asynchronous as A  # noqa: E402
import v8_data as D  # noqa: E402

OUT = HERE.parent / "results" / "v8"
sig = lambda z: 1 / (1 + np.exp(-z))


def own_gamma(y, s, b, iters=400, pen=0.01):
    """Slowness effect of one agent in its own data: logit P = theta + b_j + g * slow_j, with b fixed."""
    sl = (s > np.median(s)).astype(float); th, g = 0.0, 0.0
    for _ in range(iters):
        r = y - sig(th + b + g * sl)
        th += 0.5 * (r.mean() - pen * th); g += 0.5 * ((r * sl).sum() / max(sl.sum(), 1) - pen * g)
    return g


def main():
    """Print and save the diagnostic per new agent."""
    rows = {(r["split"], r["agent"]): r for r in csv.DictReader((OUT / "new_agents.csv").open())}
    agents = D.all_agents(); out = []
    for split in D.SPLITS:
        ids, Y, S = D.load_split(split)
        old_c = [np.corrcoef(S[a], Y[a])[0, 1] for a in D.OLD_AGENTS]
        print(f"{split}: corr(steps, success), paper agents median {np.median(old_c):+.2f}")
        for a, (date, pat, _, new) in agents.items():
            if not new:
                continue
            eco = [w for w, (d, _, _, _) in agents.items() if d < date]
            b, th, g, ga = A.fit_slow_model(np.array([Y[w] for w in eco]), np.array([S[w] for w in eco], float))
            r = rows[(split, a)]
            out.append(dict(split=split, agent=a, pattern=pat, p_new=float(r["p_new"]),
                            corr_steps_success=float(r["corr_steps_success"]), gamma_graph=g,
                            gamma_own=own_gamma(Y[a].astype(float), S[a].astype(float), b),
                            bias_naive_k10=float(r["bias_naive_k10"]), bias_graph_k10=float(r["bias_graph_k10"])))
    with (OUT / "exploratory_async.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    f = lambda k: np.array([r[k] for r in out])
    mis = f("gamma_own") - f("gamma_graph")
    rho = stats.spearmanr(mis, f("bias_graph_k10"))
    print(f"new agents: corr(steps, success) median {np.median(f('corr_steps_success')):+.2f}; "
          f"gamma graph median {np.median(f('gamma_graph')):+.2f} vs own {np.median(f('gamma_own')):+.2f}; "
          f"Spearman(mismatch, graph bias) = {rho.correlation:+.2f} (p = {rho.pvalue:.2g})")
    for r in sorted(out, key=lambda r: r["gamma_own"] - r["gamma_graph"]):
        print(f"  {r['split'][5:]:9s} {r['agent'][:34]:34s} {r['pattern']:12s} p={r['p_new']:.2f} "
              f"g_graph={r['gamma_graph']:+.2f} g_own={r['gamma_own']:+.2f} bias naive={r['bias_naive_k10']:+.3f} graph={r['bias_graph_k10']:+.3f}")


if __name__ == "__main__":
    main()


def difficulty_only(R=200, k=10, M=20, seed=20261123):
    """EXPLORATORY variant suggested by the diagnosis: impute unreturned probes with the graph's difficulty only
    (gamma = 0, no slowness effect). Same design, arrival rule and seed as the frozen analysis; not confirmatory."""
    from cvt import graph as Gr
    agents = D.all_agents(); res = []
    cases = [(sp, a) for sp in D.SPLITS for a, (_, _, _, new) in agents.items() if new]
    data = {sp: D.load_split(sp) for sp in D.SPLITS}
    for kc, (split, a) in enumerate(cases):
        ids, Y, S = data[split]; date, pat, parent, _ = agents[a]
        eco = [w for w, (d, _, _, _) in agents.items() if d < date]
        Ye, Se = np.array([Y[w] for w in eco]), np.array([S[w] for w in eco], float)
        X = Gr.ecosystem_features(Ye, Y[parent] if parent else np.zeros(len(ids), bool))
        b, th, g, ga = A.fit_slow_model(Ye, Se)
        y, s, N = Y[a], S[a].astype(float), len(ids); p = y.mean()
        rng = np.random.default_rng([seed, kc, 7]); e0, cv = [], []
        for _ in range(R):
            blocks = np.array_split(np.argsort(X[:, 0], kind="stable"), M)
            idx = np.array([rng.choice(bb) for bb in blocks]); w = np.array([len(bb) for bb in blocks], float)
            order = np.lexsort((rng.random(M), s[idx])); ret = np.zeros(M, bool); ret[order[:k]] = True
            e = A.deadline_estimate(idx, ret, y[idx], w, b, th, 0.0, np.zeros(2), N)
            e0.append(e.mean - p); cv.append(e.lo <= p <= e.hi)
        res.append(dict(split=split, agent=a, pattern=pat, bias_difficulty_only=float(np.mean(e0)),
                        rmse_difficulty_only=float(np.sqrt(np.mean(np.square(e0)))), cov_difficulty_only=float(np.mean(cv))))
    with (OUT / "exploratory_async_difficulty_only.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=res[0].keys()); wr.writeheader(); wr.writerows(res)
    frozen = {(r["split"], r["agent"]): r for r in csv.DictReader((OUT / "new_agents.csv").open())}
    f = lambda key: np.array([r[key] for r in res])
    rn = np.array([float(frozen[(r["split"], r["agent"])]["rmse_naive_k10"]) for r in res])
    print(f"difficulty-only imputation (k={k}): bias mean {f('bias_difficulty_only').mean():+.3f}, median |bias| "
          f"{np.median(np.abs(f('bias_difficulty_only'))):.3f}; RMSE ratio naive/difficulty-only median "
          f"{np.median(rn / f('rmse_difficulty_only')):.2f}; coverage mean {f('cov_difficulty_only').mean():.3f} "
          f"(min {f('cov_difficulty_only').min():.3f})")


if __name__ == "__main__" and "--difficulty-only" in sys.argv:
    difficulty_only()
