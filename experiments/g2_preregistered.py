"""Analyse PRÉ-ENREGISTRÉE de la porte G2 et des hypothèses H1-H4 sur les données de Sheng & Lu.
Ce fichier est figé avec docs/PREREGISTRATION.md (tag git prereg-v1) AVANT toute lecture des résultats.
Toute modification ultérieure doit être notée dans docs/DEVIATIONS.md.

Usage : python experiments/g2_preregistered.py /chemin/vers/llm-regression-signals [R]
Sorties : results/g2/*.csv et results/g2/summary.txt"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import estimators as E  # noqa: E402

DATA = Path(sys.argv[1]) / "results"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SEED = 20261107
MS = (10, 20, 40)
DELTA_H4 = 0.10
OUT = Path(__file__).resolve().parents[1] / "results" / "g2"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = {  # classement fixé avant les données
    "qwen": ("Qwen2.5-7B -> Qwen3-8B", "generation", True),
    "llama": ("Llama-3-8B -> Llama-3.1-8B", "revision_mineure", True),
    "mistral": ("Mistral-7B v0.2 -> v0.3", "revision_mineure", True),
    "qwen2": ("Qwen2-7B -> Qwen2.5-7B", "generation", True),
    "gemma12": ("gemma-1.1-7b -> gemma-2-9b", "generation", True),
    "llama23": ("Llama-2-7b -> Llama-3-8B", "generation", False),  # hors périmètre : analyse secondaire
}
BENCHES = ["mcq_n3000", "gpqa", "gsm8k", "mathfull", "humaneval", "mbpp"]


def load(pair, bench):
    rows = list(csv.DictReader((DATA / f"results_{pair}_{bench}.csv").open()))
    y1 = np.array([int(r["old_correct"]) for r in rows], bool)
    y2 = np.array([int(r["new_correct"]) for r in rows], bool)
    return y1, y2


def flips(a, b):
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def phi2_of(y1, y2):
    return E.phi_signed(*flips(y1, y2)) ** 2


def boot_phi(y1, y2, rng, B=2000):
    n = len(y1)
    vals = [E.phi_signed(*flips(y1[i], y2[i])) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.quantile(vals, [0.025, 0.975])


def estimators_for(y1h, n11, n10, n01, n00, rng):
    s_v, f_v = int(y1h.sum()), int((~y1h).sum())
    f = (n11, n10, n01, n00)
    return {
        "reset": E.reset(*f, rng=rng, draws=2000, prior="hybrid"),
        "full": E.full_inheritance(s_v, f_v, *f, rng=rng, draws=2000),
        "power_025": E.power_prior(s_v, f_v, *f, a=0.25, rng=rng, draws=2000),
        "power_05": E.power_prior(s_v, f_v, *f, a=0.5, rng=rng, draws=2000),
        "ds_freq": E.double_sampling_freq(s_v, f_v, *f),
        "bma": E.double_sampling_bma(s_v, f_v, *f, rng=rng, draws=2000),
    }


def main():
    rng = np.random.default_rng(SEED)
    atlas, h1, h3, h4 = [], [], [], []
    for pair, (label, change, in_scope) in PAIRS.items():
        for bench in BENCHES:
            y1, y2 = load(pair, bench)
            n = len(y1)
            p1, p2 = y1.mean(), y2.mean()
            phi = E.phi_signed(*flips(y1, y2))
            lo, hi = boot_phi(y1, y2, rng)
            n1 = max(int(y1.sum()), 1)
            n0 = max(int((~y1).sum()), 1)
            atlas.append(dict(pair=pair, label=label, change=change, in_scope=in_scope, bench=bench, n=n,
                              p_old=p1, p_new=p2, phi=phi, phi_lo=lo, phi_hi=hi, phi2=phi ** 2,
                              q1=(y1 & y2).sum() / n1, q0=(~y1 & y2).sum() / n0))
            for m in MS:
                if m > n // 2:
                    continue
                err = {k: [] for k in ("reset", "full", "power_025", "power_05", "ds_freq", "bma")}
                cov = {k: [] for k in err}
                regret = {k: [] for k in err}
                pred = []
                for _ in range(R):
                    idx = rng.choice(n, m, replace=False)
                    fl = flips(y1[idx], y2[idx])
                    ests = estimators_for(y1, *fl, rng)
                    for k, e in ests.items():
                        err[k].append((e.mean - p2) ** 2)
                        cov[k].append(e.lo <= p2 <= e.hi)
                        chosen = p2 if e.mean >= p1 else p1          # H3 : garder v ou passer à v' ?
                        regret[k].append(max(p1, p2) - chosen)
                    pred.append(E.gain_law(ests["bma"].extra["phi2_median"], m, n))
                mse = {k: float(np.mean(v)) for k, v in err.items()}
                h1.append(dict(pair=pair, bench=bench, in_scope=in_scope, m=m, n=n, phi2_true=phi ** 2,
                               gain_obs=mse["reset"] / mse["bma"],
                               gain_pred_true=E.gain_law(phi ** 2, m, n),
                               gain_pred_probe=float(np.median(pred)),
                               **{f"mse_{k}": v for k, v in mse.items()},
                               **{f"cov_{k}": float(np.mean(v)) for k, v in cov.items()}))
                h3.append(dict(pair=pair, bench=bench, in_scope=in_scope, m=m, gap=p2 - p1,
                               **{f"regret_{k}": float(np.mean(v)) for k, v in regret.items()}))
            # H4 : substitution silencieuse v' -> v'' dégradée de DELTA_H4 (succès de v' basculés en échecs)
            k = int(round(DELTA_H4 * n))
            succ = np.flatnonzero(y2)
            if len(succ) >= k:
                for m in (20, 40):
                    det = {"reset": [], "full": [], "bma": []}
                    fa = {"reset": [], "full": [], "bma": []}
                    for _ in range(R // 2):
                        y3 = y2.copy()
                        y3[rng.choice(succ, k, replace=False)] = False
                        for target, store in ((y3, det), (y2, fa)):   # dégradée, puis contrôle sans dégradation
                            idx = rng.choice(n, m, replace=False)
                            fl = flips(y2[idx], target[idx])           # v' sert d'historique pour v''
                            ests = estimators_for(y2, *fl, rng)
                            for key in store:
                                store[key].append(ests[key].hi < p2)   # alerte : UCB90 sous le niveau établi de v'
                    h4.append(dict(pair=pair, bench=bench, in_scope=in_scope, m=m,
                                   **{f"power_{x}": float(np.mean(v)) for x, v in det.items()},
                                   **{f"false_alarm_{x}": float(np.mean(v)) for x, v in fa.items()}))
            print(f"fait : {pair} {bench}", flush=True)

    for name, rows in (("atlas", atlas), ("h1", h1), ("h3", h3), ("h4", h4)):
        with (OUT / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    summarise(atlas, h1, h3, h4)


def summarise(atlas, h1, h3, h4):
    L = []
    A = [a for a in atlas if a["in_scope"]]
    phi2 = np.array([a["phi2"] for a in A])
    L.append(f"G2 — phi² sur {len(A)} transitions du périmètre : médiane {np.median(phi2):.3f}, "
             f"min {phi2.min():.3f}, max {phi2.max():.3f}, étendue {phi2.max() - phi2.min():.3f}")
    g2 = (phi2.max() - phi2.min() >= 0.20) and (0.05 <= np.median(phi2) <= 0.90)
    L.append(f"G2 franchie (étendue >= 0,20 et médiane dans [0,05 ; 0,90]) : {g2}")
    # H2 : modèle additif paire + benchmark sur atanh(phi)
    z = np.arctanh(np.clip([a["phi"] for a in A], -0.999, 0.999))
    pairs = sorted({a["pair"] for a in A}); benches = sorted({a["bench"] for a in A})
    X = np.column_stack([np.ones(len(A))] + [[a["pair"] == p for a in A] for p in pairs[1:]]
                        + [[a["bench"] == b for a in A] for b in benches[1:]]).astype(float)

    def rss(Xm):
        beta, *_ = np.linalg.lstsq(Xm, z, rcond=None)
        return float(((z - Xm @ beta) ** 2).sum())
    full = rss(X); df_res = len(A) - X.shape[1]
    kp, kb = len(pairs) - 1, len(benches) - 1
    X_no_p = np.delete(X, range(1, 1 + kp), axis=1)
    X_no_b = np.delete(X, range(1 + kp, 1 + kp + kb), axis=1)
    Fp = ((rss(X_no_p) - full) / kp) / (full / df_res); pp = 1 - stats.f.cdf(Fp, kp, df_res)
    Fb = ((rss(X_no_b) - full) / kb) / (full / df_res); pb = 1 - stats.f.cdf(Fb, kb, df_res)
    h2 = min(pp, pb) < 0.025  # Holm, 2 tests : au moins un effet significatif au seuil alpha/2
    L.append(f"H2 — effet paire p = {pp:.4f}, effet benchmark p = {pb:.4f} ; soutenue (Holm, alpha 0,05) : {h2}")
    rm = [a["phi2"] for a in A if a["change"] == "revision_mineure"]
    ge = [a["phi2"] for a in A if a["change"] == "generation"]
    L.append(f"H2 (direction, exploratoire) — phi² médian révision mineure {np.median(rm):.3f} vs génération {np.median(ge):.3f}")
    # H1
    for m in MS:
        H = [h for h in h1 if h["in_scope"] and h["m"] == m]
        if not H:
            continue
        e_true = np.array([abs(h["gain_obs"] / h["gain_pred_true"] - 1) for h in H])
        e_probe = np.array([abs(h["gain_obs"] / h["gain_pred_probe"] - 1) for h in H])
        cov = np.array([h["cov_bma"] for h in H]); covr = np.array([h["cov_reset"] for h in H])
        L.append(f"H1 m={m} — erreur relative médiane loi(phi² vrai) {np.median(e_true):.3f}, "
                 f"loi(phi² des probes) {np.median(e_probe):.3f} ; soutenue (<= 0,20, phi² vrai) : {np.median(e_true) <= 0.20}"
                 f" | couverture BMA moy {cov.mean():.3f} min {cov.min():.3f} ; reset moy {covr.mean():.3f}")
    # H3
    H = [h for h in h3 if h["in_scope"] and h["m"] == 20]
    rb = np.array([h["regret_bma"] for h in H]); rr = np.array([h["regret_reset"] for h in H])
    w = stats.wilcoxon(rb, rr, alternative="less") if np.any(rb != rr) else None
    ok = (w is not None and w.pvalue < 0.05) and np.mean(rb <= 1.05 * rr + 1e-12) >= 0.90
    L.append(f"H3 m=20 — regret moyen BMA {rb.mean():.4f} vs reset {rr.mean():.4f}, "
             f"Wilcoxon p = {w.pvalue if w else float('nan'):.4f} ; soutenue : {ok}")
    # H4
    for m in (20, 40):
        H = [h for h in h4 if h["in_scope"] and h["m"] == m]
        if H:
            pw = np.mean([h["power_bma"] for h in H]); fa = np.mean([h["false_alarm_bma"] for h in H])
            pf = np.mean([h["power_full"] for h in H]); pr = np.mean([h["power_reset"] for h in H])
            L.append(f"H4 m={m} — puissance BMA {pw:.3f} (fausses alertes {fa:.3f}), héritage complet {pf:.3f}, "
                     f"reset {pr:.3f} ; soutenue (puissance >= 0,80 et fausses alertes <= 0,10) : {pw >= 0.8 and fa <= 0.10}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
