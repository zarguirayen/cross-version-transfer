"""Analyse PRÉ-ENREGISTRÉE v4 (confirmation sur données jamais analysées) : SWE-bench Lite, 84 agents datés.
Figée avec docs/PREREGISTRATION_V4.md (tag prereg-v4) avant toute statistique de résultat sur ces données.
Question : le graphe doit-il aussi CHOISIR les tâches à tester (plan stratifié sur la difficulté vue par
l'écosystème) ? Et la loi d'efficacité s'étend-elle à ce plan ?
Usage : python experiments/v4_lite_preregistered.py /chemin/experiments [R] [--fake]
Sorties : results/v4/*.csv et results/v4/summary.txt"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import estimators as E  # noqa: E402
from cvt import graph as Gr  # noqa: E402
import v4_lite_data as D  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FAKE = "--fake" in sys.argv
ROOT = ARGS[0]
R = int(ARGS[1]) if len(ARGS) > 1 else 400
SEED = 20261116
PRIMARY_S = "v3_s"   # estimateur principal sous plan stratifié, fixé avant le gel
OUT = HERE.parent / "results" / ("v4_fake" if FAKE else "v4")
OUT.mkdir(parents=True, exist_ok=True)

IDS, Y, DATES, LIN = D.load(ROOT)
VER = D.verified_ids(ROOT)
NEWTASK = np.array([i not in VER for i in IDS])           # 207 tâches absentes de Verified (analyse secondaire)
if FAKE:
    g = np.random.default_rng(0); diff = g.normal(0, 1.5, len(IDS))
    Y = {k: g.random(len(IDS)) < 1 / (1 + np.exp(-(g.normal(-1, 1.5) + diff))) for k in sorted(Y)}
TRANS = D.transitions(DATES, LIN)
M_LIST = (20, 40)


def flips(a, b):
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def stratified(score, m, rng):
    """Plan stratifié : tâches triées par difficulté vue par l'écosystème, m strates contiguës de taille ~N/m,
    1 probe tirée au hasard par strate. Renvoie les probes (dans l'ordre des strates) et les tailles."""
    blocks = np.array_split(np.argsort(score, kind="stable"), m)
    return np.array([rng.choice(b) for b in blocks]), np.array([len(b) for b in blocks], float)


def law_strat(y, score, m):
    """Loi du plan stratifié (gain exact sur la moyenne simple) et η² entre strates."""
    y = y.astype(float); N = len(y); S2 = y.var(ddof=1)
    blocks = np.array_split(np.argsort(score, kind="stable"), m)
    within = sum((1 - 1 / len(b)) * (y[b].var(ddof=1) if len(b) > 1 else 0.0) for b in blocks) / m ** 2
    eta2 = 1 - sum(len(b) * y[b].var() for b in blocks) / (N * y.var()) if y.var() > 0 else np.nan
    G = ((1 - m / N) * S2 / m) / within if within > 0 else np.nan
    return G, eta2


def analyse(sub, y1, y2, X, rng, tag):
    """Rejeux sur le sous-ensemble de tâches `sub` (indices)."""
    y1, y2, X = y1[sub], y2[sub], X[sub]
    N = len(y2); p = y2.mean(); pv = y1.mean()
    score = X[:, 0] if X.shape[1] > 1 else rng.random(N)
    phi = E.phi_signed(*flips(y1, y2))
    rows = []
    for m in M_LIST:
        keys = ("reset_u", "bma_u", "v3_u", "reset_s", "v3_s", "greg_s")
        err = {q: [] for q in keys}; cov = {q: [] for q in keys}; sign = {q: [] for q in keys}
        for _ in range(R):
            iu = rng.choice(N, m, replace=False)
            is_, wts = stratified(score, m, rng)
            fu, fs = flips(y1[iu], y2[iu]), flips(y1[is_], y2[is_])
            ests = {"reset_u": E.reset(*fu, rng=rng, draws=1500, prior="hybrid"),
                    "bma_u": E.double_sampling_bma(int(y1.sum()), int((~y1).sum()), *fu, rng=rng, draws=1500),
                    "v3_u": Gr.graph_v3(X, iu, y2[iu], y1, rng=rng, draws=2000),
                    "reset_s": E.reset(*fs, rng=rng, draws=1500, prior="hybrid"),
                    "v3_s": Gr.graph_v3(X, is_, y2[is_], y1, rng=rng, draws=2000),
                    "greg_s": Gr.graph_greg(X, is_, y2[is_], weights=wts, rng=rng, draws=2000)}
            for q, e in ests.items():
                err[q].append((e.mean - p) ** 2); cov[q].append(e.lo <= p <= e.hi)
                sign[q].append((e.mean >= pv) != (p >= pv))          # mauvaise décision « garder / remplacer v »
        G, eta2 = law_strat(y2, score, m)
        mr = np.mean(err["reset_u"])
        rows.append(dict(tasks=tag, m=m, n=N, p_old=pv, p_new=p, delta=p - pv, phi2=phi ** 2, eta2=eta2,
                         gain_law_phi=E.gain_law(phi ** 2, m, N), gain_law_strat=G,
                         **{f"gain_{q}": mr / np.mean(err[q]) for q in keys},
                         **{f"cov_{q}": float(np.mean(cov[q])) for q in keys},
                         **{f"decerr_{q}": float(np.mean(sign[q])) for q in keys}))
    return rows


def run_transition(t):
    k, (lin, old, new) = t
    rng = np.random.default_rng([SEED, k])
    eco = D.ecosystem(DATES, old, new)
    X = Gr.ecosystem_features(np.array([Y[e] for e in eco]) if eco else None, Y[old])
    rows = analyse(np.arange(len(IDS)), Y[old], Y[new], X, rng, "toutes")
    rows += analyse(np.flatnonzero(NEWTASK), Y[old], Y[new], X, rng, "hors_verified")
    for r in rows:
        r.update(lineage=lin, old=old, new=new, n_eco=len(eco))
    print(f"fait : {lin} {old} -> {new}", flush=True)
    return rows


def main():
    with Pool(2) as pool:
        res = pool.map(run_transition, list(enumerate(TRANS)), chunksize=1)
    rows = [x for r in res for x in r]
    with (OUT / "replays.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summarise(rows)


def summarise(rows):
    L = []
    P = [r for r in rows if r["m"] == 20 and r["tasks"] == "toutes"]
    a = lambda k, S=P: np.array([r[k] for r in S], float)
    e1 = np.abs(a("gain_bma_u") / a("gain_law_phi") - 1)
    L.append(f"H1-v4 — {len(P)} transitions : erreur médiane de la loi phi² {np.median(e1):.3f} ; soutenue (<= 0,20) : {np.median(e1) <= 0.20}")
    ok = ~np.isnan(a("gain_law_strat"))
    e8 = np.abs(a("gain_reset_s")[ok] / a("gain_law_strat")[ok] - 1)
    L.append(f"H8-v4 — loi du plan stratifié : erreur médiane {np.median(e8):.3f} (n={ok.sum()}), η² médian {np.nanmedian(a('eta2')):.3f} ; "
             f"soutenue (<= 0,20) : {np.median(e8) <= 0.20}")
    gs, gu, cs = a(f"gain_{PRIMARY_S}"), a("gain_v3_u"), a(f"cov_{PRIMARY_S}")
    w = stats.wilcoxon(gs, gu, alternative="greater")
    h9 = w.pvalue < 0.05 and np.mean(gs >= 0.95) >= 0.90 and cs.mean() >= 0.88
    L.append(f"H9-v4 — {PRIMARY_S} x{np.median(gs):.3f} vs graphe uniforme (v3) x{np.median(gu):.3f} vs prédécesseur seul x{np.median(a('gain_bma_u')):.3f} ; "
             f"Wilcoxon p = {w.pvalue:.2g} ; >= 0,95 sur {np.mean(gs >= 0.95):.0%} ; couverture {cs.mean():.3f} (min {cs.min():.3f}) ; soutenue : {h9}")
    D_ = [r for r in P if abs(r["delta"]) >= 0.05]
    ds, du = a(f"decerr_{PRIMARY_S}", D_), a("decerr_reset_u", D_)
    w10 = stats.wilcoxon(ds, du, alternative="less") if np.any(ds != du) else None
    L.append(f"H10-v4 — {len(D_)} transitions avec |écart| >= 5 points : erreur de décision {PRIMARY_S} {ds.mean():.3f} vs reset uniforme {du.mean():.3f} ; "
             f"Wilcoxon p = {w10.pvalue if w10 else float('nan'):.2g} ; soutenue (p < 0,05) : {bool(w10 and w10.pvalue < 0.05)}")
    gg = a("gain_greg_s"); cg = a("cov_greg_s")
    L.append(f"H11-v4 — estimateur assisté par modèle (GREG, plan stratifié) : pire gain x{gg.min():.3f}, médiane x{np.median(gg):.3f} ; "
             f"couverture moy {cg.mean():.3f} (min {cg.min():.3f}, non testée) ; soutenue (pire gain >= 0,80) : {gg.min() >= 0.80}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
