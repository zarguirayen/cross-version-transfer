"""Analyse PRÉ-ENREGISTRÉE v2 (confirmation sur données jamais analysées) : agents AppWorld.
Figée avec docs/PREREGISTRATION_V2.md (tag prereg-v2) avant toute statistique sur ces données.
Usage : python experiments/v2_appworld_preregistered.py /chemin/aw_root [R]
Sorties : results/v2/*.csv et results/v2/summary.txt"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import estimators as E  # noqa: E402
from cvt import graph as Gr  # noqa: E402

ROOT = Path(sys.argv[1]) / "experiments" / "outputs"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SEED = 20261111
OUT = Path(__file__).resolve().parents[1] / "results" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
SPLITS = ["test_normal", "test_challenge"]
METHODS = ["react", "plan_exec", "full_code_refl", "ipfuncall"]
LLMS = ["gpt4turbo", "gpt4o", "llama3", "deepseekcoder"]


def agents_available(split):
    return sorted(p.name[: -len(split) - 1] for p in ROOT.glob(f"*_{split}") if (p / "evaluations" / f"{split}.json").exists())


def load(agent, split):
    d = json.load((ROOT / f"{agent}_{split}" / "evaluations" / f"{split}.json").open())["individual"]
    ids = sorted(d)
    return ids, np.array([bool(d[t]["success"]) for t in ids])


def llm_of(agent):
    return next(l for l in LLMS if agent.endswith("_" + l))


def transitions():
    """Liste fixée a priori : (type, ancien agent, nouvel agent)."""
    T = []
    for m in METHODS:                                   # A : montée de version du modèle, architecture fixe
        T.append(("A_modele", f"{m}_gpt4turbo", f"{m}_gpt4o"))
    for llm in ("gpt4turbo", "gpt4o"):                  # B : changement d'architecture, modèle fixe
        for m in ("plan_exec", "full_code_refl", "ipfuncall"):
            T.append(("B_architecture", f"react_{llm}", f"{m}_{llm}"))
    for llm in ("llama3", "deepseekcoder"):
        for m in ("plan_exec", "full_code_refl"):
            T.append(("B_architecture", f"react_{llm}", f"{m}_{llm}"))
    for m in ("react", "plan_exec", "full_code_refl"):  # C : changement de famille de modèle, architecture fixe
        for old in ("llama3", "deepseekcoder"):
            T.append(("C_famille", f"{m}_{old}", f"{m}_gpt4o"))
    return T


def flips(a, b):
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def cluster_boot_phi(y1, y2, clusters, rng, B=2000):
    u = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in u}
    vals = []
    for _ in range(B):
        pick = rng.choice(u, len(u), replace=True)
        ii = np.concatenate([idx_by[c] for c in pick])
        vals.append(E.phi_signed(*flips(y1[ii], y2[ii])))
    return np.quantile(vals, [0.025, 0.975])


def main():
    rng = np.random.default_rng(SEED)
    atlas, rep_rows, h4_rows = [], [], []
    for split in SPLITS:
        av = agents_available(split)
        data = {a: load(a, split) for a in av}
        ids = data[av[0]][0]
        assert all(data[a][0] == ids for a in av), "tâches non alignées"
        Y = {a: data[a][1] for a in av}
        clusters = np.array([t.split("_")[0] for t in ids])     # scénario = préfixe de l'identifiant
        n = len(ids)
        for kind, old, new in transitions():
            y1, y2 = Y[old], Y[new]
            aux_strict = [a for a in av if a not in (old, new) and llm_of(a) != llm_of(new)]
            aux_all = [a for a in av if a not in (old, new)]
            X_strict = Gr.ecosystem_features(np.array([Y[a] for a in aux_strict]), y1)
            X_all = Gr.ecosystem_features(np.array([Y[a] for a in aux_all]), y1)
            phi = E.phi_signed(*flips(y1, y2))
            lo, hi = cluster_boot_phi(y1, y2, clusters, rng)
            p1, p2 = y1.mean(), y2.mean()
            atlas.append(dict(split=split, type=kind, old=old, new=new, n=n, p_old=p1, p_new=p2, phi=phi,
                              phi_lo=lo, phi_hi=hi, phi2=phi ** 2, n_aux_strict=len(aux_strict)))
            for m in (20, 40):
                if m > n // 2:
                    continue
                err = {k: [] for k in ("reset", "bma", "bma09", "graph", "graph_all")}
                cov = {k: [] for k in err}
                for _ in range(R):
                    idx = rng.choice(n, m, replace=False)
                    f = flips(y1[idx], y2[idx])
                    ests = {
                        "reset": E.reset(*f, rng=rng, draws=1500, prior="hybrid"),
                        "bma": E.double_sampling_bma(int(y1.sum()), int((~y1).sum()), *f, rng=rng, draws=1500),
                        "bma09": E.double_sampling_bma(int(y1.sum()), int((~y1).sum()), *f, rng=rng, draws=1500, prior_link=0.9),
                        "graph": Gr.graph_logit(X_strict, idx, y2[idx], rng=rng, draws=2000),
                        "graph_all": Gr.graph_logit(X_all, idx, y2[idx], rng=rng, draws=2000),
                    }
                    for k, e in ests.items():
                        err[k].append((e.mean - p2) ** 2)
                        cov[k].append(e.lo <= p2 <= e.hi)
                mse = {k: float(np.mean(v)) for k, v in err.items()}
                rep_rows.append(dict(split=split, type=kind, old=old, new=new, m=m, n=n, phi2=phi ** 2, p_new=p2,
                                     gain_law=E.gain_law(phi ** 2, m, n),
                                     **{f"gain_{k}": mse["reset"] / mse[k] for k in err if k != "reset"},
                                     **{f"cov_{k}": float(np.mean(v)) for k, v in cov.items()}))
            print(f"fait : {split} {kind} {old} -> {new}", flush=True)
        # H4 v2 : détection séquentielle, agents avec p >= 0,20
        for agent in av:
            y = Y[agent]
            if y.mean() < 0.20:
                continue
            for scen in ("degradee_10pts", "bruit_5pc", "identique"):
                alerts, used = [], []
                for _ in range(200):
                    y3 = y.copy()
                    if scen == "degradee_10pts":
                        y3[rng.choice(np.flatnonzero(y), int(round(0.10 * n)), replace=False)] = False
                    elif scen == "bruit_5pc":
                        fl = rng.random(n) < 0.05
                        y3[fl] = ~y3[fl]
                    order = rng.permutation(n)
                    alert, used_m = False, 0
                    for used_m in range(10, 81, 10):
                        idx = order[:used_m]
                        e = E.double_sampling_bma(int(y.sum()), int((~y).sum()), *flips(y[idx], y3[idx]), rng=rng,
                                                  draws=1500, level=0.95)
                        if e.hi < y.mean():
                            alert = True
                            break
                        if e.lo > y.mean() - 0.08:
                            break
                    alerts.append(alert); used.append(used_m)
                h4_rows.append(dict(split=split, agent=agent, p=float(y.mean()), scenario=scen,
                                    alert_rate=float(np.mean(alerts)), mean_probes=float(np.mean(used))))
    for name, rows in (("atlas", atlas), ("replays", rep_rows), ("h4_sequential", h4_rows)):
        with (OUT / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summarise(atlas, rep_rows, h4_rows)


def summarise(atlas, rep, h4):
    L = []
    R20 = [r for r in rep if r["m"] == 20]
    # H1-v2 : la loi sur des agents
    err = np.array([abs(r["gain_bma"] / r["gain_law"] - 1) for r in R20])
    L.append(f"H1-v2 — {len(R20)} transitions, m=20 : erreur relative médiane loi {np.median(err):.3f} ; soutenue (<= 0,20) : {np.median(err) <= 0.20}")
    # H5 : le graphe fait mieux que le prédécesseur seul
    gg = np.array([r["gain_graph"] for r in R20]); gb = np.array([r["gain_bma"] for r in R20])
    w = stats.wilcoxon(gg, gb, alternative="greater")
    nondeg = np.mean(gg >= 0.95)
    covg = np.array([r["cov_graph"] for r in R20])
    h5 = w.pvalue < 0.05 and nondeg >= 0.90 and covg.mean() >= 0.88
    L.append(f"H5 — gain médian graphe x{np.median(gg):.3f} vs prédécesseur seul x{np.median(gb):.3f} ; Wilcoxon p = {w.pvalue:.4f} ; "
             f"graphe >= 0,95 x reset sur {nondeg:.0%} ; couverture graphe moy {covg.mean():.3f} min {covg.min():.3f} ; soutenue : {h5}")
    # H2-v2 : phi² selon le type de transition (une valeur par transition et par split)
    groups = {t: [a["phi2"] for a in atlas if a["type"] == t] for t in ("A_modele", "B_architecture", "C_famille")}
    kw = stats.kruskal(*groups.values())
    L.append("H2-v2 — phi² médian : " + ", ".join(f"{t} {np.median(v):.3f} (n={len(v)})" for t, v in groups.items())
             + f" ; Kruskal-Wallis p = {kw.pvalue:.4f} ; soutenue (p < 0,05) : {kw.pvalue < 0.05}")
    # H4-v2 : détection séquentielle
    deg = [h for h in h4 if h["scenario"] == "degradee_10pts"]
    noi = [h for h in h4 if h["scenario"] == "bruit_5pc"]
    pw = np.mean([h["alert_rate"] for h in deg]); mp = np.mean([h["mean_probes"] for h in deg])
    fa = np.mean([h["alert_rate"] for h in noi])
    L.append(f"H4-v2 — {len(deg)} agents x split : puissance {pw:.3f} avec {mp:.1f} probes en moyenne ; fausses alertes (bruit 5 %) {fa:.3f} ; "
             f"soutenue (puissance >= 0,80, probes <= 60, fausses alertes <= 0,10) : {pw >= 0.80 and mp <= 60 and fa <= 0.10}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
