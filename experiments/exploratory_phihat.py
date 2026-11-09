"""EXPLORATORY, RETROSPECTIVE (protocol: docs/EXPLORATORY_PHIHAT.md, committed before this script was run).

Does the predecessor law still announce the gain when phi² is estimated from the 20 tested tasks only, instead of the
true phi² computed on every task? Transitions of v3-v6 at m = 20, loaded with the frozen studies' own data loaders
and matched to their rows in results/<study>/replays.csv.

Usage
-----
    python experiments/exploratory_phihat.py /home/claude/data/swe/experiments
Outputs: results/exploratory_phihat/{transitions.csv, summary.txt}
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import estimators as E  # noqa: E402

ROOT = sys.argv[1]
M, R, SEED = 20, 400, 20261109
RES = HERE.parent / "results"
OUT = RES / "exploratory_phihat"


def flips(a, b):
    """Contingency counts (n11, n10, n01, n00) of old outcomes ``a`` against new outcomes ``b``."""
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def pairs():
    """Yield (study, y_old, y_new, replay_row) for every m = 20 transition of v3-v6, in the frozen scripts' order."""
    import v3_swebench_data as D3
    import v4_lite_data as D4
    import v5_general_data as D5
    import v6_data as D6
    rows = {s: [r for r in csv.DictReader((RES / s / "replays.csv").open())
                if r["m"] == "20" and r.get("tasks", "toutes") == "toutes"]       # main task set only
            for s in ("v3", "v4", "v5", "v6")}
    _, Y, dates, lin, _ = D3.load(ROOT)
    for (o, n), r in zip(((t[-2], t[-1]) for t in D3.transitions(dates, lin)), rows["v3"]):
        assert (o, n) == (r["old"], r["new"])
        yield "v3", Y[o], Y[n], r
    _, Y, dates, lin = D4.load(ROOT)
    for t, r in zip(D4.transitions(dates, lin), rows["v4"]):
        yield "v4", Y[t[-2]], Y[t[-1]], r
    trans5 = []
    for ts in D5.TASKSETS:
        _, Y, dates, lin = D5.load(ts)
        trans5 += [(Y[t[-2]], Y[t[-1]]) for t in D5.transitions(dates, lin)]
    for (a, b), r in zip(trans5, rows["v5"]):
        yield "v5", a, b, r
    trans6 = []
    for fn in (D6.load_lcb, D6.load_tac):
        _, Y, _, trans = fn()
        trans6 += [(Y[a], Y[b]) for _, a, b in trans]
    for (a, b), r in zip(trans6, rows["v6"]):
        yield "v6", a, b, r


def main():
    """Run the protocol once and write the per-transition table and the summary."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = []
    for study, yo, yn, r in pairs():
        yo, yn = np.asarray(yo, bool), np.asarray(yn, bool)
        N = len(yo)
        phi_true = E.phi_signed(*flips(yo, yn))
        assert abs(phi_true ** 2 - float(r["phi2"])) < 1e-9 and abs(yn.mean() - float(r["p_new"])) < 1e-9, (study, r)
        g_obs = float(r["gain_bma"] if study == "v3" else r["gain_bma_u"])
        g_true = float(r["gain_law"] if study == "v3" else r["gain_law_phi"])
        ph2, gh, deg = [], [], []
        for _ in range(R):
            idx = rng.choice(N, M, replace=False)
            a, b = yo[idx], yn[idx]
            d = a.all() or (~a).all() or b.all() or (~b).all()
            p2 = E.phi_signed(*flips(a, b)) ** 2
            ph2.append(p2); gh.append(E.gain_law(p2, M, N)); deg.append(d)
        ph2, gh, deg = np.array(ph2), np.array(gh), np.array(deg)
        nd = ~deg
        out.append(dict(study=study, N=N, p_old=float(yo.mean()), p_new=float(yn.mean()), phi2=phi_true ** 2,
                        gain_obs=g_obs, gain_law_true=g_true, gain_law_hat_median=float(np.median(gh)),
                        gain_law_hat_median_nondeg=float(np.median(gh[nd])) if nd.any() else np.nan,
                        phi2_hat_median=float(np.median(ph2)), phi2_hat_iqr=float(np.subtract(*np.percentile(ph2, [75, 25]))),
                        share_degenerate=float(deg.mean()),
                        err_single_median=float(np.median(np.abs(gh / g_obs - 1))),
                        err_single_median_nondeg=float(np.median(np.abs(gh[nd] / g_obs - 1))) if nd.any() else np.nan))
    with (OUT / "transitions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    f = lambda k, S=out: np.array([r[k] for r in S], float)
    L = []
    for lab, S in [("all", out)] + [(s, [r for r in out if r["study"] == s]) for s in ("v3", "v4", "v5", "v6")]:
        go = f("gain_obs", S)
        e_hat = np.abs(f("gain_law_hat_median", S) / go - 1)
        e_true = np.abs(f("gain_law_true", S) / go - 1)
        e_none = np.abs(1 / go - 1)
        nd = ~np.isnan(f("gain_law_hat_median_nondeg", S))
        e_hat_nd = np.abs(f("gain_law_hat_median_nondeg", S)[nd] / go[nd] - 1)
        L.append(f"{lab:4s} n={len(S):3d} | per-transition error: phi-hat {np.median(e_hat):.3f} (non-degenerate {np.median(e_hat_nd):.3f}) "
                 f"vs true phi {np.median(e_true):.3f} vs no-transfer {np.median(e_none):.3f} | single-evaluation error "
                 f"{np.median(f('err_single_median', S)):.3f} | phi2 bias {np.median(f('phi2_hat_median', S) - f('phi2', S)):+.3f} "
                 f"| phi2-hat IQR {np.median(f('phi2_hat_iqr', S)):.3f} | degenerate draws {np.mean(f('share_degenerate', S)):.3f}")
    big = f("gain_obs") >= 1.5
    eb = lambda k: np.median(np.abs(f(k)[big] / f("gain_obs")[big] - 1))
    L.append(f"observed gain >= 1.5 (n={big.sum()}): per-transition error phi-hat {eb('gain_law_hat_median'):.3f} vs true phi {eb('gain_law_true'):.3f}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
