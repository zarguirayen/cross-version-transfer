"""DÉVELOPPEMENT v7 (données de développement de cette question : SWE-bench Verified et AppWorld uniquement).
Question : la STRUCTURE du graphe (qui est relié à qui) apporte-t-elle de l'information au-delà de la difficulté
moyenne vue par tous les agents ? Arêtes apprises : les k agents antérieurs qui se comportent le plus comme la
nouvelle version sur les probes. Plan stratifié par la difficulté du graphe, comme la méthode unifiée.
Usage : python experiments/dev_v7_structure.py verified|appworld [R]"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import estimators as E  # noqa: E402
from cvt import graph as Gr  # noqa: E402

WHICH = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 100
M = 20
OUT = HERE.parent / "results" / f"dev_v7_{WHICH}.csv"


def cases():
    if WHICH == "verified":
        import v3_swebench_data as D
        ids, Y, dates, lin, _ = D.load("/home/claude/data/swe/experiments")
        for L, old, new in D.transitions(dates, lin):
            eco = D.ecosystem(dates, lin, old, new)
            yield f"{L}|{old}->{new}", Y[old], Y[new], (np.array([Y[e] for e in eco]) if eco else None)
    else:
        import importlib.util
        sys.argv = [sys.argv[0], "/home/claude/data/aw_root"]
        spec = importlib.util.spec_from_file_location("v2", HERE / "v2_appworld_preregistered.py")
        v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
        for split in v2.SPLITS:
            av = v2.agents_available(split); Y = {a: v2.load(a, split)[1] for a in av}
            for kind, old, new in v2.transitions():
                aux = [a for a in av if a not in (old, new)]          # graphe complet (pas d'ordre temporel)
                yield f"{split}|{kind}|{old}->{new}", Y[old], Y[new], np.array([Y[a] for a in aux])


def stratified(score, m, rng):
    blocks = np.array_split(np.argsort(score, kind="stable"), m)
    return np.array([rng.choice(b) for b in blocks]), np.array([len(b) for b in blocks], float)


def flips(a, b):
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


KEYS = ("reset_u", "greg_cf", "greg_knn5", "greg_knn10", "v3_s", "v3_knn5")


def run(arg):
    k, (name, y1, y2, aux) = arg
    rng = np.random.default_rng([20261120, k])
    X = Gr.ecosystem_features(aux, y1)
    score = X[:, 0] if aux is not None else rng.random(len(y2))
    n, p, pv = len(y2), y2.mean(), y1.mean()
    err = {q: [] for q in KEYS}; cov = {q: [] for q in KEYS}; wid = {q: [] for q in KEYS}; dec = {q: [] for q in KEYS}
    for _ in range(R):
        iu = rng.choice(n, M, replace=False)
        r = E.reset(*flips(y1[iu], y2[iu]), rng=rng, draws=1000, prior="hybrid")
        idx, w = stratified(score, M, rng)
        ests = {"reset_u": r,
                "greg_cf": Gr.graph_greg_cf(X, idx, y2[idx], w, rng=rng, draws=1000),
                "greg_knn5": Gr.graph_greg_cf_knn(X, aux, idx, y2[idx], w, k=5, rng=rng, draws=1000),
                "greg_knn10": Gr.graph_greg_cf_knn(X, aux, idx, y2[idx], w, k=10, rng=rng, draws=1000),
                "v3_s": Gr.graph_v3(X, idx, y2[idx], y1, rng=rng, draws=1500),
                "v3_knn5": Gr.graph_v3_knn(X, aux, idx, y2[idx], y1, k=5, rng=rng, draws=1500)}
        for q, e in ests.items():
            err[q].append((e.mean - p) ** 2); cov[q].append(e.lo <= p <= e.hi); wid[q].append(e.hi - e.lo)
            dec[q].append((e.mean >= pv) != (p >= pv))
    mr = np.mean(err["reset_u"])
    row = dict(case=name, n=n, p_new=p, delta=p - pv, n_eco=0 if aux is None else len(aux),
               **{f"gain_{q}": mr / np.mean(err[q]) for q in KEYS}, **{f"cov_{q}": float(np.mean(cov[q])) for q in KEYS},
               **{f"width_{q}": float(np.mean(wid[q])) for q in KEYS}, **{f"dec_{q}": float(np.mean(dec[q])) for q in KEYS})
    print(f"{name[:60]:60s} p'={p:.3f} | greg x{row['gain_greg_cf']:.2f} knn5 x{row['gain_greg_knn5']:.2f} knn10 x{row['gain_greg_knn10']:.2f} "
          f"| v3_s x{row['gain_v3_s']:.2f} v3_knn5 x{row['gain_v3_knn5']:.2f}", flush=True)
    return row


if __name__ == "__main__":
    with Pool(2) as pool:
        rows = pool.map(run, list(enumerate(cases())), chunksize=1)
    with OUT.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    from scipy import stats
    for lab, S in (("tous", rows), ("p' 10-90 %", [r for r in rows if 0.1 <= r["p_new"] < 0.9])):
        print(f"\n{lab} ({len(S)})")
        for q in KEYS:
            g = np.array([r[f"gain_{q}"] for r in S]); c = np.array([r[f"cov_{q}"] for r in S])
            wd = np.mean([r[f"width_{q}"] for r in S]); B = [r for r in S if abs(r["delta"]) >= 0.05]
            de = np.mean([r[f"dec_{q}"] for r in B]) if B else float("nan")
            print(f"  {q:10s} x{np.median(g):.3f} (min {g.min():.2f}) couv {c.mean():.3f}/{c.min():.3f} largeur {wd:.3f} décision {de:.3f}")
        for a, b in (("greg_knn5", "greg_cf"), ("greg_knn10", "greg_cf"), ("v3_knn5", "v3_s")):
            ga = np.array([r[f"gain_{a}"] for r in S]); gb = np.array([r[f"gain_{b}"] for r in S])
            print(f"  {a} > {b} : {np.mean(ga > gb):.0%} ; Wilcoxon p = {stats.wilcoxon(ga, gb, alternative='greater').pvalue:.3g} ; rapport médian {np.median(ga / gb):.3f}")
