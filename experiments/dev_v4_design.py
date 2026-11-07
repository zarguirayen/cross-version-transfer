"""DÉVELOPPEMENT v4 (données déjà vues : SWE-bench Verified après v3, AppWorld après v2) — pas une preuve.
Question : choisir les probes en les répartissant sur l'échelle de difficulté connue du graphe (plan stratifié,
1 probe par strate de taille N/m) améliore-t-il l'estimation, en particulier pour les versions faibles ?
La sélection ne dépend que de l'historique (agents antérieurs), jamais de v'.
Usage : python experiments/dev_v4_design.py verified|appworld [R]"""
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
R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
M = 20
OUT = HERE.parent / "results" / f"dev_v4b_{WHICH}.csv"


def flips(a, b):
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


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
                aux = [a for a in av if a not in (old, new) and v2.llm_of(a) != v2.llm_of(new)]
                yield f"{split}|{kind}|{old}->{new}", Y[old], Y[new], np.array([Y[a] for a in aux])


def stratified(score, m, rng):
    """1 probe par strate, dans l'ordre des strates ; renvoie aussi la taille de chaque strate."""
    order = np.argsort(score, kind="stable")
    blocks = np.array_split(order, m)
    return np.array([rng.choice(b) for b in blocks]), np.array([len(b) for b in blocks], float)


def run(case):
    k, (name, y1, y2, aux) = case
    rng = np.random.default_rng([20261115, k])
    X = Gr.ecosystem_features(aux, y1)
    score = X[:, 0] if aux is not None else rng.random(len(y2))
    n, p = len(y2), y2.mean()
    keys = ("reset_u", "v3_u", "greg_u", "reset_s", "v3_s", "greg_s")
    err = {q: [] for q in keys}; cov = {q: [] for q in keys}; wid = {q: [] for q in keys}
    for _ in range(R):
        for design in ("u", "s"):
            if design == "u":
                idx, wts = rng.choice(n, M, replace=False), None
            else:
                idx, wts = stratified(score, M, rng)
            f = flips(y1[idx], y2[idx])
            ests = {f"reset_{design}": E.reset(*f, rng=rng, draws=1500, prior="hybrid"),
                    f"v3_{design}": Gr.graph_v3(X, idx, y2[idx], y1, rng=rng, draws=2000),
                    f"greg_{design}": Gr.graph_greg(X, idx, y2[idx], weights=wts, rng=rng, draws=2000)}
            for q, e in ests.items():
                err[q].append((e.mean - p) ** 2); cov[q].append(e.lo <= p <= e.hi); wid[q].append(e.hi - e.lo)
    mr = np.mean(err["reset_u"])
    row = dict(case=name, n=n, p_new=p, **{f"gain_{q}": mr / np.mean(err[q]) for q in keys},
               **{f"cov_{q}": float(np.mean(cov[q])) for q in keys}, **{f"width_{q}": float(np.mean(wid[q])) for q in keys})
    print(f"{name[:70]:70s} p'={p:.3f} | v3_u x{row['gain_v3_u']:.2f} v3_s x{row['gain_v3_s']:.2f} greg_s x{row['gain_greg_s']:.2f} reset_s x{row['gain_reset_s']:.2f} "
          f"| couv v3_s {row['cov_v3_s']:.2f} greg_s {row['cov_greg_s']:.2f}", flush=True)
    return row


if __name__ == "__main__":
    with Pool(2) as pool:
        rows = pool.map(run, list(enumerate(cases())), chunksize=1)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for lab, S in (("tous", rows), ("p' < 0,10", [r for r in rows if r["p_new"] < 0.10]),
                   ("p' >= 0,10", [r for r in rows if r["p_new"] >= 0.10])):
        if not S:
            continue
        print(f"\n{lab} ({len(S)})")
        for q in ("reset_u", "v3_u", "greg_u", "reset_s", "v3_s", "greg_s"):
            g = np.array([r[f"gain_{q}"] for r in S]); c = np.array([r[f"cov_{q}"] for r in S])
            wd = np.array([r[f"width_{q}"] for r in S])
            print(f"  {q:8s} gain médian x{np.median(g):.3f} min x{g.min():.2f} >=0,95 {np.mean(g >= 0.95):.0%} "
                  f"| couv moy {c.mean():.3f} min {c.min():.3f} | largeur moy {wd.mean():.3f}")
