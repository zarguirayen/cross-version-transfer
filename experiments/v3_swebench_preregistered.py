"""Analyse PRÉ-ENREGISTRÉE v3 (confirmation sur données jamais analysées) : SWE-bench Verified, 175 agents datés.
Figée avec docs/PREREGISTRATION_V3.md (tag prereg-v3) avant toute statistique sur ces données.
Usage : python experiments/v3_swebench_preregistered.py /chemin/experiments [R] [--fake]
  --fake : remplace les résultats par des tirages aléatoires de même forme (test à blanc du script).
Sorties : results/v3/*.csv et results/v3/summary.txt"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvt import estimators as E  # noqa: E402
from cvt import graph as Gr  # noqa: E402
import v3_swebench_data as D  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FAKE = "--fake" in sys.argv
ROOT = ARGS[0]
R = int(ARGS[1]) if len(ARGS) > 1 else 400
SEED = 20261114
OUT = Path(__file__).resolve().parents[1] / "results" / ("v3_fake" if FAKE else "v3")
OUT.mkdir(parents=True, exist_ok=True)

ids, Y, DATES, LIN, _ = D.load(ROOT)
if FAKE:
    g = np.random.default_rng(0)
    diff = g.normal(0, 1.5, len(ids))
    Y = {k: g.random(len(ids)) < 1 / (1 + np.exp(-(g.normal(-0.5, 1.5) + diff))) for k in sorted(Y)}
TRANS = D.transitions(DATES, LIN)
REPO = np.array([i.split("__")[0] for i in ids])        # grappe = dépôt GitHub de la tâche
N = len(ids)


def flips(a, b):
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def cluster_boot_phi(y1, y2, rng, B=2000):
    u = np.unique(REPO)
    idx_by = {c: np.flatnonzero(REPO == c) for c in u}
    vals = []
    for _ in range(B):
        ii = np.concatenate([idx_by[c] for c in rng.choice(u, len(u), replace=True)])
        vals.append(E.phi_signed(*flips(y1[ii], y2[ii])))
    return np.quantile(vals, [0.025, 0.975])


def control_of(old, new):
    """Témoin du lien de lignée : agent antérieur à v', hors lignée de v', différent de v, de taux le plus proche
    de celui de v (égalité : ordre alphabétique)."""
    cand = [e for e in Y if DATES[e] < DATES[new] and e != old and LIN[e] != LIN[new]]
    if not cand:
        return None
    pv = Y[old].mean()
    return min(cand, key=lambda e: (abs(Y[e].mean() - pv), e))


def run_transition(t):
    k, (lin, old, new) = t
    rng = np.random.default_rng([SEED, k])
    y1, y2 = Y[old], Y[new]
    eco = D.ecosystem(DATES, LIN, old, new)
    eco_s = D.ecosystem(DATES, LIN, old, new, exclude_lineage=True)
    X = Gr.ecosystem_features(np.array([Y[e] for e in eco]) if eco else None, y1)
    Xs = Gr.ecosystem_features(np.array([Y[e] for e in eco_s]) if eco_s else None, y1)
    phi = E.phi_signed(*flips(y1, y2)); lo, hi = cluster_boot_phi(y1, y2, rng)
    c = control_of(old, new)
    phi_c = E.phi_signed(*flips(Y[c], y2)) if c else np.nan
    atlas = dict(kind="mini" if lin.startswith("mini") else "produit", lineage=lin, old=old, new=new, n=N,
                 p_old=y1.mean(), p_new=y2.mean(), phi=phi, phi_lo=lo, phi_hi=hi, phi2=phi ** 2,
                 control=c, phi2_control=phi_c ** 2, n_eco=len(eco), n_eco_strict=len(eco_s))
    reps = []
    for m in (20, 40):
        keys = ("reset", "bma", "eco_only", "graph", "graph_v3", "graph_v3_strict")
        err = {q: [] for q in keys}; cov = {q: [] for q in keys}; fb = []
        for _ in range(R):
            idx = rng.choice(N, m, replace=False)
            f = flips(y1[idx], y2[idx])
            e3 = Gr.graph_v3(X, idx, y2[idx], y1, rng=rng, draws=2000)
            ests = {"reset": E.reset(*f, rng=rng, draws=1500, prior="hybrid"),
                    "bma": E.double_sampling_bma(int(y1.sum()), int((~y1).sum()), *f, rng=rng, draws=1500),
                    "graph": Gr.graph_logit(X, idx, y2[idx], rng=rng, draws=2000),
                    "graph_v3": e3,
                    "graph_v3_strict": Gr.graph_v3(Xs, idx, y2[idx], y1, rng=rng, draws=2000)}
            fb.append(e3.extra["fallback"])
            sy = int(y2[idx].sum())
            ests["eco_only"] = (Gr.graph_bma_logit(X[:, :1] if eco else X[:, :0], idx, y2[idx], rng=rng, draws=2000, average=False)
                                if 0 < sy < m else ests["reset"])
            for q, e in ests.items():
                err[q].append((e.mean - y2.mean()) ** 2); cov[q].append(e.lo <= y2.mean() <= e.hi)
        mse = {q: float(np.mean(v)) for q, v in err.items()}
        reps.append(dict(kind=atlas["kind"], lineage=lin, old=old, new=new, m=m, n=N, phi2=phi ** 2, p_new=y2.mean(),
                         gain_law=E.gain_law(phi ** 2, m, N), fallback_rate=float(np.mean(fb)),
                         **{f"gain_{q}": mse["reset"] / mse[q] for q in keys if q != "reset"},
                         **{f"cov_{q}": float(np.mean(v)) for q, v in cov.items()}))
    # H4-v3 : détection séquentielle sur la transition réelle (pas de dégradation synthétique)
    pv = y1.mean(); alerts, used = [], []
    for _ in range(200):
        order = rng.permutation(N); alert, mm = False, 0
        for mm in range(10, 81, 10):
            ii = order[:mm]
            e = E.double_sampling_bma(int(y1.sum()), int((~y1).sum()), *flips(y1[ii], y2[ii]), rng=rng, draws=1500, level=0.95)
            if e.hi < pv:
                alert = True; break
            if e.lo > pv - 0.08:
                break
        alerts.append(alert); used.append(mm)
    h4 = dict(lineage=lin, old=old, new=new, p_old=pv, p_new=y2.mean(), delta=y2.mean() - pv,
              alert_rate=float(np.mean(alerts)), mean_probes=float(np.mean(used)))
    print(f"fait : {lin} {old} -> {new}", flush=True)
    return atlas, reps, h4


def main():
    with Pool(2) as pool:
        res = pool.map(run_transition, list(enumerate(TRANS)), chunksize=1)
    atlas = [r[0] for r in res]; reps = [x for r in res for x in r[1]]; h4 = [r[2] for r in res]
    for name, rows in (("atlas", atlas), ("replays", reps), ("h4_real", h4)):
        with (OUT / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summarise(atlas, reps, h4)


def summarise(atlas, rep, h4):
    L = []
    R20 = [r for r in rep if r["m"] == 20]
    err = np.array([abs(r["gain_bma"] / r["gain_law"] - 1) for r in R20])
    L.append(f"H1-v3 — {len(R20)} transitions, m=20 : erreur relative médiane loi {np.median(err):.3f} ; soutenue (<= 0,20) : {np.median(err) <= 0.20}")
    g3 = np.array([r["gain_graph_v3"] for r in R20]); gb = np.array([r["gain_bma"] for r in R20])
    c3 = np.array([r["cov_graph_v3"] for r in R20])
    w = stats.wilcoxon(g3, gb, alternative="greater")
    nd = np.mean(g3 >= 0.95)
    h5 = w.pvalue < 0.05 and nd >= 0.90 and c3.mean() >= 0.88
    L.append(f"H5-v3 — gain médian graphe v3 x{np.median(g3):.3f} vs prédécesseur seul x{np.median(gb):.3f} ; Wilcoxon p = {w.pvalue:.2g} ; "
             f">= 0,95 x reset sur {nd:.0%} ; couverture moy {c3.mean():.3f} ; soutenue : {h5}")
    h6 = g3.min() >= 0.80 and c3.min() >= 0.80
    L.append(f"H6-v3 — pire gain v3 x{g3.min():.3f} ; pire couverture v3 {c3.min():.3f} ; soutenue (>= 0,80 et >= 0,80) : {h6}")
    ge = np.array([r["gain_eco_only"] for r in R20])
    w7 = stats.wilcoxon(g3, ge, alternative="greater")
    L.append(f"H7-v3 — graphe v3 x{np.median(g3):.3f} vs écosystème seul (type IRT, sans arête de lignée) x{np.median(ge):.3f} ; "
             f"Wilcoxon p = {w7.pvalue:.2g} ; soutenue (p < 0,05) : {w7.pvalue < 0.05}")
    a = [x for x in atlas if not np.isnan(x["phi2_control"])]
    pl = np.array([x["phi2"] for x in a]); pc = np.array([x["phi2_control"] for x in a])
    w2 = stats.wilcoxon(pl, pc, alternative="greater")
    L.append(f"H2-v3 — {len(a)} transitions : phi² médian lignée {np.median(pl):.3f} vs témoin de même taux {np.median(pc):.3f} ; "
             f"Wilcoxon p = {w2.pvalue:.2g} ; soutenue (p < 0,05) : {w2.pvalue < 0.05}")
    reg = [h for h in h4 if h["delta"] <= -0.10]; ok = [h for h in h4 if h["delta"] >= 0]
    if len(reg) >= 5:
        pw = np.mean([h["alert_rate"] for h in reg]); fa = np.mean([h["alert_rate"] for h in ok])
        mp = np.mean([h["mean_probes"] for h in reg])
        L.append(f"H4-v3 — {len(reg)} régressions réelles >= 10 points : alerte {pw:.3f} avec {mp:.1f} probes ; fausses alertes sur {len(ok)} "
                 f"transitions sans baisse {fa:.3f} ; soutenue (>= 0,80 et <= 0,10) : {pw >= 0.80 and fa <= 0.10}")
    else:
        L.append(f"H4-v3 — {len(reg)} régressions réelles >= 10 points (< 5) : non testée, descriptif seulement")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
