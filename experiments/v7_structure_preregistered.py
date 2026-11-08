"""Analyse PRÉ-ENREGISTRÉE v7 — la structure du graphe (analyse secondaire pré-enregistrée de données déjà utilisées
pour d'autres questions ; aucune statistique de structure n'y a été calculée avant le gel).
Figée avec docs/PREREGISTRATION_V7.md (tag prereg-v7).
Partie A — la structure existe-t-elle ? Corrélation des résidus (après un modèle à une dimension, capacité + facilité)
  entre paires d'agents, selon le type de lien : Multi-SWE-bench (7 langages, plan croisé échafaudage × modèle) et
  LiveCodeBench (même fournisseur ou non).
Partie B — sert-elle à évaluer ? Arêtes apprises (voisins les plus semblables sur les probes) dans GREG croisé et dans
  l'estimateur fondé sur le modèle, sur les 90 transitions des confirmations v4 à v6, plan stratifié, m = 20.
Usage : python experiments/v7_structure_preregistered.py [R] [--fake]"""
import csv
import re
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import estimators as E  # noqa: E402
from cvt import graph as Gr  # noqa: E402
import v4_lite_data as D4  # noqa: E402
import v5_general_data as D5  # noqa: E402
import v6_data as D6  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FAKE = "--fake" in sys.argv
R = int(ARGS[0]) if ARGS else 200
SEED = 20261121
M = 20
OUT = HERE.parent / "results" / ("v7_fake" if FAKE else "v7")
OUT.mkdir(parents=True, exist_ok=True)
SCAFF = {"mswe-agent": "MSWE-agent", "magentless": "MagentLess", "mopenhands": "MopenHands"}
LCB_PROVIDER = [(r"^(GPT|O1|O3|O4)", "openai"), (r"^Claude", "anthropic"), (r"^Gemini", "google"),
                (r"^(DSCoder|DeepSeek)", "deepseek"), (r"^(Qwen|QwQ)", "alibaba"), (r"Nemotron", "nvidia"),
                (r"^LLama", "meta"), (r"^Grok", "xai"), (r"^EXAONE", "lg"), (r"^DeepCoder", "agentica")]


def fake(Y, n):
    g = np.random.default_rng(n); diff = g.normal(0, 1.5, n)
    return {k: g.random(n) < 1 / (1 + np.exp(-(g.normal(-0.5, 1.5) + diff))) for k in sorted(Y)}


def rasch_residuals(Y, iters=300):
    """Résidus standardisés après le modèle à une dimension logit P = capacité + facilité (pénalisé, 0,01)."""
    A, N = Y.shape
    th, b = np.zeros(A), np.zeros(N)
    for _ in range(iters):
        P = 1 / (1 + np.exp(-(th[:, None] + b[None, :])))
        Rr = Y - P
        th += 0.5 * (Rr.mean(axis=1) - 0.01 * th)
        b += 0.5 * (Rr.mean(axis=0) - 0.01 * b)
    P = 1 / (1 + np.exp(-(th[:, None] + b[None, :])))
    return (Y - P) / np.sqrt(P * (1 - P))


def pair_stats(names, Y, key_a, key_b):
    """Corrélation moyenne des résidus : paires de même clé A (clé B différente), de même clé B (A différente),
    sans lien. key_* : fonction nom -> étiquette."""
    Rz = rasch_residuals(Y)
    C = np.corrcoef(Rz)
    g = {"A": [], "B": [], "none": []}
    for i, j in combinations(range(len(names)), 2):
        a, b = key_a(names[i]) == key_a(names[j]), key_b(names[i]) == key_b(names[j]) if key_b else False
        if a and not b:
            g["A"].append(C[i, j])
        elif b and not a:
            g["B"].append(C[i, j])
        elif not a and not b:
            g["none"].append(C[i, j])
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in g.items()}, C


def perm_p(names, C, key, n_perm=2000, rng=None):
    """Permutation des étiquettes : p unilatéral pour « paires de même étiquette > autres paires »."""
    lab = np.array([key(n) for n in names]); iu = np.triu_indices(len(names), 1)
    def stat(l):
        same = l[iu[0]] == l[iu[1]]
        return C[iu][same].mean() - C[iu][~same].mean()
    obs = stat(lab)
    null = [stat(rng.permutation(lab)) for _ in range(n_perm)]
    return obs, (1 + np.sum(np.array(null) >= obs)) / (n_perm + 1)


def part_a(rng):
    rows = []
    for lang in D5.MSWE_LANGS:
        ids, Y, dates, lin = D5.load_taskset(lang)
        if FAKE:
            Y = fake(Y, len(ids))
        names = []
        for d in sorted(Y):
            rest = d.split("_", 1)[1].lower()
            sc = next((v for k, v in SCAFF.items() if rest.startswith(k + "_")), None)
            if sc and dates[d] == "20250329":          # plan croisé initial : 3 échafaudages × 9 modèles
                names.append(d)
        sc_of = lambda d: next(v for k, v in SCAFF.items() if d.split("_", 1)[1].lower().startswith(k + "_"))
        model_of = lambda d: d.split("_", 2)[2].lower()
        Ym = np.array([Y[d] for d in names], float)
        s, C = pair_stats(names, Ym, sc_of, model_of)
        obs, p = perm_p(names, C, sc_of, rng=rng)
        rows.append(dict(source=f"mswe-{lang}", n_agents=len(names), n_tasks=len(ids), same_scaffold=s["A"],
                         same_model=s["B"], unrelated=s["none"], perm_stat=obs, perm_p=p))
    ids, Y, dates, _ = D6.load_lcb()
    if FAKE:
        Y = fake(Y, len(ids))
    prov = lambda d: next((v for pat, v in LCB_PROVIDER if re.search(pat, d)), None)
    variant = re.compile(r"\((Low|Med|Medium|Thinking)\)|-high$")       # un seul nœud par modèle
    names = [d for d in sorted(Y) if prov(d) and not variant.search(d)]
    s, C = pair_stats(names, np.array([Y[d] for d in names], float), prov, None)
    obs, p = perm_p(names, C, prov, rng=rng)
    rows.append(dict(source="lcb", n_agents=len(names), n_tasks=len(ids), same_scaffold=np.nan, same_model=s["A"],
                     unrelated=s["none"], perm_stat=obs, perm_p=p))
    return rows


def b_cases():
    ids, Y, dates, lin = D4.load("/home/claude/data/swe/experiments")
    if FAKE:
        Y = fake(Y, len(ids))
    for L, old, new in D4.transitions(dates, lin):
        eco = D4.ecosystem(dates, old, new)
        yield "lite", f"{L}|{new}", Y[old], Y[new], (np.array([Y[e] for e in eco]) if eco else None)
    for ts in D5.TASKSETS:
        ids, Y, dates, lin = D5.load(ts)
        if FAKE:
            Y = fake(Y, len(ids))
        for L, old, new in D5.transitions(dates, lin):
            eco = D5.ecosystem(dates, old, new)
            yield ts, f"{L}|{new}", Y[old], Y[new], (np.array([Y[e] for e in eco]) if eco else None)
    for fn, tag in ((D6.load_lcb, "lcb"), (D6.load_tac, "tac")):
        ids, Y, dates, trans = fn()
        if FAKE:
            Y = fake(Y, len(ids))
        for L, old, new in trans:
            eco = D6.ecosystem(dates, old, new)
            yield tag, f"{L}|{new}", Y[old], Y[new], (np.array([Y[e] for e in eco]) if eco else None)


KEYS = ("reset_u", "greg_cf", "greg_knn10", "v3_s", "v3_knn5")


def run_b(arg):
    k, (ts, name, y1, y2, aux) = arg
    rng = np.random.default_rng([SEED, k])
    X = Gr.ecosystem_features(aux, y1)
    score = X[:, 0] if aux is not None else rng.random(len(y2))
    n, p, pv = len(y2), y2.mean(), y1.mean()
    err = {q: [] for q in KEYS}; cov = {q: [] for q in KEYS}
    for _ in range(R):
        iu = rng.choice(n, M, replace=False)
        a, b = y1[iu], y2[iu]
        r = E.reset(int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum()), rng=rng, draws=1000, prior="hybrid")
        blocks = np.array_split(np.argsort(score, kind="stable"), M)
        idx = np.array([rng.choice(bl) for bl in blocks]); w = np.array([len(bl) for bl in blocks], float)
        ests = {"reset_u": r,
                "greg_cf": Gr.graph_greg_cf(X, idx, y2[idx], w, rng=rng, draws=1000),
                "greg_knn10": Gr.graph_greg_cf_knn(X, aux, idx, y2[idx], w, k=10, rng=rng, draws=1000),
                "v3_s": Gr.graph_v3(X, idx, y2[idx], y1, rng=rng, draws=1500),
                "v3_knn5": Gr.graph_v3_knn(X, aux, idx, y2[idx], y1, k=5, rng=rng, draws=1500)}
        for q, e in ests.items():
            err[q].append((e.mean - p) ** 2); cov[q].append(e.lo <= p <= e.hi)
    mr = np.mean(err["reset_u"])
    print(f"fait : {ts} {name}", flush=True)
    return dict(taskset=ts, transition=name, n=n, p_new=p, n_eco=0 if aux is None else len(aux),
                **{f"gain_{q}": mr / np.mean(err[q]) for q in KEYS}, **{f"cov_{q}": float(np.mean(cov[q])) for q in KEYS})


def main():
    A = part_a(np.random.default_rng(SEED))
    with Pool(2) as pool:
        B = pool.map(run_b, list(enumerate(b_cases())), chunksize=1)
    for name, rows in (("structure", A), ("replays", B)):
        with (OUT / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summarise(A, B)


def summarise(A, B):
    L = []
    ms = [r for r in A if r["source"].startswith("mswe")]
    d_sc = np.array([r["same_scaffold"] - r["unrelated"] for r in ms]); d_mo = np.array([r["same_model"] - r["unrelated"] for r in ms])
    w1 = stats.wilcoxon(d_sc, alternative="greater")
    w1b = stats.wilcoxon(d_sc - d_mo, alternative="greater")
    L.append(f"S1 — même échafaudage vs sans lien (7 langages) : écart médian {np.median(d_sc):+.3f}, positif dans {np.sum(d_sc > 0)}/7 ; "
             f"Wilcoxon p = {w1.pvalue:.3g} ; permutations p par langage {', '.join(f'{r['perm_p']:.3f}' for r in ms)} ; soutenue (p < 0,05) : {w1.pvalue < 0.05}")
    L.append(f"S2 — même échafaudage plus semblable que même modèle : écart médian {np.median(d_sc - d_mo):+.3f}, positif dans {np.sum(d_sc - d_mo > 0)}/7 ; "
             f"Wilcoxon p = {w1b.pvalue:.3g} ; soutenue (p < 0,05) : {w1b.pvalue < 0.05}")
    lc = next(r for r in A if r["source"] == "lcb")
    L.append(f"S3 — LiveCodeBench, même fournisseur vs autres : écart {lc['perm_stat']:+.3f}, permutation p = {lc['perm_p']:.3g} ; soutenue (p < 0,05) : {lc['perm_p'] < 0.05}")
    gk, gc = np.array([r["gain_greg_knn10"] for r in B]), np.array([r["gain_greg_cf"] for r in B])
    w4 = stats.wilcoxon(gk, gc, alternative="greater"); ratio = np.median(gk / gc)
    ck = np.array([r["cov_greg_knn10"] for r in B])
    L.append(f"S4 — arêtes apprises dans GREG croisé ({len(B)} transitions) : x{np.median(gk):.3f} vs x{np.median(gc):.3f}, rapport médian {ratio:.3f} ; "
             f"Wilcoxon p = {w4.pvalue:.3g} ; couverture {ck.mean():.3f} (min {ck.min():.3f}) ; soutenue (p < 0,05) : {w4.pvalue < 0.05}")
    L.append(f"S5 — l'apport de la structure est de second ordre : rapport médian {ratio:.3f} ; soutenue (<= 1,10) : {ratio <= 1.10}")
    vk, vs = np.array([r["gain_v3_knn5"] for r in B]), np.array([r["gain_v3_s"] for r in B])
    L.append(f"(rapporté) estimateur fondé sur le modèle : x{np.median(vk):.3f} vs x{np.median(vs):.3f}, rapport médian {np.median(vk / vs):.3f}, "
             f"Wilcoxon p = {stats.wilcoxon(vk, vs, alternative='greater').pvalue:.3g}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
