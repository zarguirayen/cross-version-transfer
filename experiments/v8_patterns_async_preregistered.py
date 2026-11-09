"""PRE-REGISTERED analysis v8: agentic patterns, multi-agent systems and asynchronous evaluation.

Frozen with docs/PREREGISTRATION_V8.md (tag prereg-v8) before any outcome statistic on the 13 new AppWorld agents
and before any pattern statistic on Multi-SWE-bench.

Part N  - the 13 agents submitted to the AppWorld leaderboard after the paper (never analysed), on both splits, each
          evaluated as a new version with the graph of all agents released strictly before it: stratified design +
          cross-fitted GREG against uniform random testing, by pattern class (multi-agent systems included).
Part A  - the same replays under asynchronous arrival: probes return in the order of the agent's own number of
          steps; at a deadline (k = 10 of m = 20 returned; k = 15 reported) the naive estimate is compared with the
          graph's slowness-corrected estimate (cvt.asynchronous).
Part P  - Multi-SWE-bench, 7 languages, initial crossed design (3 scaffolds x 9 models): do agents of the same
          pattern (two agent loops: MSWE-agent, MopenHands) fail together more than agents of different patterns
          (Agentless-style workflow MagentLess vs a loop)?

Usage
-----
    python experiments/v8_patterns_async_preregistered.py [R] [--fake]
Outputs: results/v8/{new_agents.csv, pattern_structure.csv, summary.txt}
"""
import csv
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import asynchronous as A  # noqa: E402
from cvt import estimators as E  # noqa: E402
from cvt import graph as Gr  # noqa: E402
import v5_general_data as D5  # noqa: E402
import v8_data as D  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FAKE = "--fake" in sys.argv
R = int(ARGS[0]) if ARGS else 200
SEED = 20261123
M, KS = 20, (10, 15)
OUT = HERE.parent / "results" / ("v8_fake" if FAKE else "v8")
OUT.mkdir(parents=True, exist_ok=True)
LOOPS = {"mswe-agent": "MSWE-agent", "mopenhands": "MopenHands"}      # agent-loop pattern
WORKFLOW = {"magentless": "MagentLess"}                              # Agentless-style fixed workflow


def fake_split(Y, S, seed):
    """Dry-run data: outcomes from a logistic model with a shared difficulty, steps independent of outcomes."""
    g = np.random.default_rng(seed); n = len(next(iter(Y.values()))); d = g.normal(0, 1.5, n)
    Yf = {a: g.random(n) < 1 / (1 + np.exp(-(g.normal(-0.5, 1.5) + d))) for a in sorted(Y)}
    Sf = {a: g.integers(1, 60, n) for a in sorted(Y)}
    return Yf, Sf


def stratified(score, m, rng):
    """One probe per stratum of the graph's difficulty; returns probe indices and stratum sizes."""
    blocks = np.array_split(np.argsort(score, kind="stable"), m)
    return np.array([rng.choice(b) for b in blocks]), np.array([len(b) for b in blocks], float)


def law_strat(y, score, m):
    """Exact gain of the stratified design over the simple mean (Proposition 2) and eta² between strata."""
    y = y.astype(float); N = len(y); S2 = y.var(ddof=1)
    blocks = np.array_split(np.argsort(score, kind="stable"), m)
    within = sum((1 - 1 / len(b)) * (y[b].var(ddof=1) if len(b) > 1 else 0.0) for b in blocks) / m ** 2
    eta2 = 1 - sum(len(b) * y[b].var() for b in blocks) / (N * y.var()) if y.var() > 0 else np.nan
    return (((1 - m / N) * S2 / m) / within if within > 0 else np.nan), eta2


DATA = {}
for _sp in D.SPLITS:
    _ids, _Y, _S = D.load_split(_sp)
    if FAKE:
        _Y, _S = fake_split(_Y, _S, D.SPLITS.index(_sp) + 1)
    DATA[_sp] = (_ids, _Y, _S)
AGENTS = D.all_agents()
CASES = [(sp, a) for sp in D.SPLITS for a, (_, _, _, new) in AGENTS.items() if new]


def run_case(arg):
    """R replays of one new agent on one split (parts N and A share the same probes)."""
    k_case, (split, a) = arg
    rng = np.random.default_rng([SEED, k_case])
    ids, Y, S = DATA[split]
    date, pattern, parent, _ = AGENTS[a]
    eco = [w for w, (d, _, _, _) in AGENTS.items() if d < date]
    Ye, Se = np.array([Y[w] for w in eco]), np.array([S[w] for w in eco], float)
    y_prev = Y[parent] if parent else np.zeros(len(ids), bool)
    X = Gr.ecosystem_features(Ye, y_prev)
    b, th, g, ga = A.fit_slow_model(Ye, Se)
    y, s, N = Y[a], S[a].astype(float), len(ids)
    p = float(y.mean())
    best = max(eco, key=lambda w: Y[w].mean()); p_best = float(Y[best].mean())
    err = {q: [] for q in ("reset_u", "reset_s", "greg_cf_s")}
    cov = {q: [] for q in err}; dec = {q: [] for q in err}
    dl = {k: {"naive": [], "graph": [], "cov": [], "width": []} for k in KS}
    for _ in range(R):
        iu = rng.choice(N, M, replace=False)
        is_, w = stratified(X[:, 0], M, rng)
        su, ss = int(y[iu].sum()), int(y[is_].sum())
        ests = {"reset_u": E.reset(su, 0, 0, M - su, rng=rng, draws=1500, prior="hybrid"),
                "reset_s": E.reset(ss, 0, 0, M - ss, rng=rng, draws=1500, prior="hybrid"),
                "greg_cf_s": Gr.graph_greg_cf(X, is_, y[is_], w, rng=rng, draws=1500)}
        for q, e in ests.items():
            err[q].append((e.mean - p) ** 2); cov[q].append(e.lo <= p <= e.hi)
            dec[q].append((e.mean >= p_best) != (p >= p_best))       # "does the new agent beat the best earlier one?"
        order = np.lexsort((rng.random(M), s[is_]))                 # asynchronous arrival: fewest steps first
        for k in KS:
            ret = np.zeros(M, bool); ret[order[:k]] = True
            e = A.deadline_estimate(is_, ret, y[is_], w, b, th, g, ga, N)
            dl[k]["naive"].append(y[is_][ret].mean() - p); dl[k]["graph"].append(e.mean - p)
            dl[k]["cov"].append(e.lo <= p <= e.hi); dl[k]["width"].append(e.hi - e.lo)
    G, eta2 = law_strat(y, X[:, 0], M)
    mr = np.mean(err["reset_u"])
    row = dict(split=split, agent=a, pattern=pattern, date=date, parent=parent or "", n_eco=len(eco), N=N, p_new=p,
               p_best_earlier=p_best, delta_best=p - p_best, eta2=eta2, gain_law_strat=G, gamma=g,
               corr_steps_success=float(np.corrcoef(s, y)[0, 1]) if y.std() > 0 and s.std() > 0 else np.nan,
               **{f"gain_{q}": mr / np.mean(err[q]) for q in err}, **{f"cov_{q}": float(np.mean(cov[q])) for q in err},
               **{f"decerr_{q}": float(np.mean(dec[q])) for q in err})
    for k in KS:
        nv, gr = np.array(dl[k]["naive"]), np.array(dl[k]["graph"])
        row.update({f"bias_naive_k{k}": nv.mean(), f"bias_graph_k{k}": gr.mean(),
                    f"rmse_naive_k{k}": float(np.sqrt((nv ** 2).mean())), f"rmse_graph_k{k}": float(np.sqrt((gr ** 2).mean())),
                    f"cov_graph_k{k}": float(np.mean(dl[k]["cov"])), f"width_graph_k{k}": float(np.mean(dl[k]["width"]))})
    print(f"done: {split} {a}", flush=True)
    return row


def rasch_residuals(Y, iters=300):
    """Standardised residuals after the one-dimensional model logit P = ability + easiness (penalty 0.01), as in v7."""
    Aa, N = Y.shape
    th, b = np.zeros(Aa), np.zeros(N)
    for _ in range(iters):
        P = 1 / (1 + np.exp(-(th[:, None] + b[None, :])))
        Rr = Y - P
        th += 0.5 * (Rr.mean(axis=1) - 0.01 * th)
        b += 0.5 * (Rr.mean(axis=0) - 0.01 * b)
    P = 1 / (1 + np.exp(-(th[:, None] + b[None, :])))
    return (Y - P) / np.sqrt(P * (1 - P))


def part_p():
    """Pattern structure on Multi-SWE-bench (initial crossed design of 20250329, as in v7)."""
    rows = []
    for lang in D5.MSWE_LANGS:
        ids, Y, dates, _ = D5.load_taskset(lang)
        if FAKE:
            g = np.random.default_rng(len(ids)); d = g.normal(0, 1.5, len(ids))
            Y = {k: g.random(len(ids)) < 1 / (1 + np.exp(-(g.normal(-0.5, 1.5) + d))) for k in sorted(Y)}
        sc = {}
        for k in sorted(Y):
            rest = k.split("_", 1)[1].lower()
            hit = next((v for kk, v in {**LOOPS, **WORKFLOW}.items() if rest.startswith(kk + "_")), None)
            if hit and dates[k] == "20250329":
                sc[k] = hit
        names = sorted(sc)
        model = {k: k.split("_", 2)[2].lower() for k in names}
        pat = {k: "loop" if sc[k] in LOOPS.values() else "workflow" for k in names}
        Ym = np.array([Y[k] for k in names], float)
        C = np.corrcoef(rasch_residuals(Ym))
        same_pat, diff_pat, phi_same, phi_diff = [], [], [], []
        for i, j in combinations(range(len(names)), 2):
            a, c = names[i], names[j]
            if sc[a] == sc[c]:
                continue                                             # only pairs of different scaffolds
            if model[a] != model[c]:
                (same_pat if pat[a] == pat[c] else diff_pat).append(C[i, j])
            else:                                                    # same model, different scaffold: raw phi²
                y1, y2 = Ym[i].astype(bool), Ym[j].astype(bool)
                ph = E.phi_signed(int((y1 & y2).sum()), int((y1 & ~y2).sum()), int((~y1 & y2).sum()), int((~y1 & ~y2).sum()))
                (phi_same if pat[a] == pat[c] else phi_diff).append(ph ** 2)
        rows.append(dict(language=lang, n_agents=len(names), n_tasks=len(ids),
                         resid_same_pattern=float(np.mean(same_pat)), resid_diff_pattern=float(np.mean(diff_pat)),
                         n_same=len(same_pat), n_diff=len(diff_pat),
                         phi2_same_pattern=float(np.nanmean(phi_same)), phi2_diff_pattern=float(np.nanmean(phi_diff)),
                         n_phi_same=len(phi_same), n_phi_diff=len(phi_diff)))
    return rows


def write(path, rows):
    """Write a list of dicts to CSV."""
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def wil(x, alt="greater"):
    """One-sided Wilcoxon signed-rank p-value against 0 (nan when every value is 0)."""
    x = np.asarray(x, float)
    return float(stats.wilcoxon(x, alternative=alt).pvalue) if np.any(x != 0) else float("nan")


def summarise(Nrows, Prows):
    """Apply the frozen criteria; write results/v8/summary.txt."""
    L = []
    f = lambda k, S: np.array([r[k] for r in S], float)
    mas = [r for r in Nrows if r["pattern"] == "multi-agent"]
    g_all, g_mas = f("gain_greg_cf_s", Nrows), f("gain_greg_cf_s", mas)
    p1 = wil(np.log(g_all)); n1 = np.median(g_all) > 1 and p1 < 0.05
    L.append(f"N1-v8 — new agents (n={len(Nrows)}): GREG-cf gain median x{np.median(g_all):.3f} (min x{g_all.min():.3f}); "
             f"Wilcoxon log-gain > 0 p = {p1:.3g} ; supported (median > 1 and p < 0.05): {n1}")
    p2 = wil(np.log(g_mas)); n2 = np.median(g_mas) > 1 and p2 < 0.05
    L.append(f"N2-v8 — multi-agent systems (n={len(mas)}): gain median x{np.median(g_mas):.3f} (min x{g_mas.min():.3f}); "
             f"Wilcoxon p = {p2:.3g} ; supported (median > 1 and p < 0.05): {n2}")
    c_all, c_mas = f("cov_greg_cf_s", Nrows), f("cov_greg_cf_s", mas)
    n3 = c_all.mean() >= 0.85 and c_mas.mean() >= 0.85
    L.append(f"N3-v8 — calibration: GREG-cf 90% coverage mean {c_all.mean():.3f} (min {c_all.min():.3f}); multi-agent "
             f"{c_mas.mean():.3f} ; supported (both >= 0.85): {n3}")
    ok = ~np.isnan(f("gain_law_strat", Nrows))
    e = np.abs(f("gain_reset_s", Nrows)[ok] / f("gain_law_strat", Nrows)[ok] - 1)
    L.append(f"N4-v8 — stratified law: median error {np.median(e):.3f} (n={ok.sum()}) ; supported (<= 0.20): {np.median(e) <= 0.20}")
    bn, bg = f("bias_naive_k10", Nrows), f("bias_graph_k10", Nrows)
    pa1 = wil(bn); a1 = np.median(bn) > 0 and pa1 < 0.05
    L.append(f"A1-v8 — deadline bias exists (k=10): naive signed bias median {np.median(bn):+.3f} (mean {bn.mean():+.3f}); "
             f"Wilcoxon > 0 p = {pa1:.3g} ; supported (median > 0 and p < 0.05): {a1}")
    pa2 = wil(np.abs(bn) - np.abs(bg)); ratio = f("rmse_naive_k10", Nrows) / f("rmse_graph_k10", Nrows)
    a2 = pa2 < 0.05 and np.median(ratio) > 1
    L.append(f"A2-v8 — graph corrects it: |bias| naive {np.median(np.abs(bn)):.3f} vs graph {np.median(np.abs(bg)):.3f} (medians); "
             f"Wilcoxon p = {pa2:.3g} ; RMSE ratio naive/graph median {np.median(ratio):.2f} ; supported (p < 0.05 and ratio > 1): {a2}")
    cg = f("cov_graph_k10", Nrows)
    L.append(f"A3-v8 — calibration at the deadline: coverage mean {cg.mean():.3f} (min {cg.min():.3f}) ; supported (>= 0.85): {cg.mean() >= 0.85}")
    d1 = f("resid_same_pattern", Prows) - f("resid_diff_pattern", Prows)
    pp1 = wil(d1)
    L.append(f"P1-v8 — same pattern fails together (residuals, different scaffold and model): difference median {np.median(d1):+.3f}, "
             f"positive in {(d1 > 0).sum()}/{len(d1)} languages ; Wilcoxon p = {pp1:.3g} ; supported (p < 0.05): {pp1 < 0.05}")
    d2 = f("phi2_same_pattern", Prows) - f("phi2_diff_pattern", Prows)
    pp2 = wil(d2)
    L.append(f"P2-v8 — same model, pattern kept vs changed (raw phi²): difference median {np.median(d2):+.3f}, positive in "
             f"{(d2 > 0).sum()}/{len(d2)} ; Wilcoxon p = {pp2:.3g} ; supported (p < 0.05): {pp2 < 0.05}")
    L.append("Reported without criterion: gains and coverage by pattern class, k = 15, decisions against the best earlier "
             "agent, step-success correlations, pattern-class descriptives.")
    for cls in ("multi-agent", "RL-trained", "single-agent"):
        S_ = [r for r in Nrows if r["pattern"] == cls]
        L.append(f"  {cls:12s} n={len(S_):2d} gain median x{np.median(f('gain_greg_cf_s', S_)):.3f} "
                 f"cov {f('cov_greg_cf_s', S_).mean():.3f} naive bias {f('bias_naive_k10', S_).mean():+.3f} "
                 f"graph bias {f('bias_graph_k10', S_).mean():+.3f} cov_deadline {f('cov_graph_k10', S_).mean():.3f}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


def main():
    """Run parts N/A in parallel, then part P; write CSVs and the summary."""
    with Pool(2) as pool:
        Nrows = pool.map(run_case, list(enumerate(CASES)), chunksize=1)
    write(OUT / "new_agents.csv", Nrows)
    Prows = part_p()
    write(OUT / "pattern_structure.csv", Prows)
    summarise(Nrows, Prows)


if __name__ == "__main__":
    main()
