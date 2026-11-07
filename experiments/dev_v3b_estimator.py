"""DÉVELOPPEMENT (variante b : prior empirique de pente tiré du graphe) (données déjà vues : AppWorld après v2, Sheng & Lu après v1) — pas une preuve.
Compare l'estimateur graphe v2 (figé) à ses deux corrections de principe :
  graph_fp  : population finie seule ;
  graph_v3  : population finie + moyenne sur les sous-ensembles de covariables (vraisemblance marginale).
Usage : python experiments/dev_v3_estimator.py appworld /chemin/aw_root [R]
        python experiments/dev_v3_estimator.py sheng /chemin/llm-regression-signals [R]"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import estimators as E  # noqa: E402
from cvt import graph as Gr  # noqa: E402

WHICH, SRC = sys.argv[1], Path(sys.argv[2])
R = int(sys.argv[3]) if len(sys.argv) > 3 else 200
M = 20
OUT = Path(__file__).resolve().parents[1] / "results" / f"dev_v3b_{WHICH}.csv"


def flips(a, b):
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def cases_appworld():
    sys.argv = [sys.argv[0], str(SRC)]
    import importlib.util
    spec = importlib.util.spec_from_file_location("v2", Path(__file__).parent / "v2_appworld_preregistered.py")
    v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
    for split in v2.SPLITS:
        av = v2.agents_available(split)
        Y = {a: v2.load(a, split)[1] for a in av}
        for kind, old, new in v2.transitions():
            aux = [a for a in av if a not in (old, new) and v2.llm_of(a) != v2.llm_of(new)]
            yield f"{split}|{kind}|{old}->{new}", Y[old], Y[new], np.array([Y[a] for a in aux])


def cases_sheng():
    sys.argv = [sys.argv[0], str(SRC)]
    import importlib.util
    spec = importlib.util.spec_from_file_location("ds", Path(__file__).parent / "dev_graph_sheng.py")
    ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
    for bench in ds.BENCHES:
        res = ds.outcomes(bench)
        for pair, (old, new) in ds.PAIRS.items():
            if pair == "llama23":
                continue
            names = [k for k in res if k not in (old, new) and ds.RELEASE[k] < ds.RELEASE[new]]
            yield f"{bench}|{pair}", res[old], res[new], (np.array([res[k] for k in names]) if names else None)


def main():
    rng = np.random.default_rng(20261113)
    rows, reps = [], []
    for name, y1, y2, aux in (cases_appworld() if WHICH == "appworld" else cases_sheng()):
        X = Gr.ecosystem_features(aux, y1)
        eb = Gr.ecosystem_slope_prior(aux)
        sm = None if eb is None else np.array([eb[0], 0.0]); ss = None if eb is None else np.array([eb[1], 1.0])
        n, p = len(y2), y2.mean()
        keys = ("reset", "bma", "graph_fp", "graph_eb", "graph_eb_avg")
        err = {k: [] for k in keys}; cov = {k: [] for k in keys}; wid = {k: [] for k in keys}
        for _ in range(R):
            idx = rng.choice(n, M, replace=False)
            f = flips(y1[idx], y2[idx])
            ests = {"reset": E.reset(*f, rng=rng, draws=1500, prior="hybrid"),
                    "bma": E.double_sampling_bma(int(y1.sum()), int((~y1).sum()), *f, rng=rng, draws=1500),
                    "graph_fp": Gr.graph_bma_logit(X, idx, y2[idx], rng=rng, draws=2000, average=False),
                    "graph_eb": Gr.graph_bma_logit(X, idx, y2[idx], rng=rng, draws=2000, average=False, slope_mean=sm, slope_sds=ss),
                    "graph_eb_avg": Gr.graph_bma_logit(X, idx, y2[idx], rng=rng, draws=2000, slope_mean=sm, slope_sds=ss)}
            reps.append(dict(case=name, p_new=p, s_probe=int(y2[idx].sum()),
                             **{f"{k}_{a}": getattr(e, a) for k, e in ests.items() for a in ("mean", "lo", "hi")}))
            for k, e in ests.items():
                err[k].append((e.mean - p) ** 2); cov[k].append(e.lo <= p <= e.hi); wid[k].append(e.hi - e.lo)
        mr = np.mean(err["reset"])
        row = dict(case=name, n=n, p_new=p, **{f"gain_{k}": mr / np.mean(err[k]) for k in keys if k != "reset"},
                   **{f"cov_{k}": float(np.mean(cov[k])) for k in keys}, **{f"width_{k}": float(np.mean(wid[k])) for k in keys})
        rows.append(row)
        print(f"{name:55s} p'={p:.3f} eb={eb} | bma x{row['gain_bma']:.2f} fp x{row['gain_graph_fp']:.2f} eb x{row['gain_graph_eb']:.2f} "
              f"eb_avg x{row['gain_graph_eb_avg']:.2f} | couv fp {row['cov_graph_fp']:.2f} eb {row['cov_graph_eb']:.2f} eba {row['cov_graph_eb_avg']:.2f}", flush=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with OUT.with_name(OUT.stem + "_replays.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=reps[0].keys()); w.writeheader(); w.writerows(reps)
    for lab, S in (("tous", rows), ("p' < 0,10", [r for r in rows if r["p_new"] < 0.10]), ("p' >= 0,10", [r for r in rows if r["p_new"] >= 0.10])):
        if not S:
            continue
        print(f"\n{lab} ({len(S)})")
        for k in ("bma", "graph_fp", "graph_eb", "graph_eb_avg"):
            g = np.array([r[f"gain_{k}"] for r in S]); c = np.array([r[f"cov_{k}"] for r in S])
            print(f"  {k:9s} gain médian x{np.median(g):.3f} min x{g.min():.2f} >=0,95 sur {np.mean(g >= 0.95):.0%} | couv moy {c.mean():.3f} min {c.min():.3f}")


if __name__ == "__main__":
    main()
