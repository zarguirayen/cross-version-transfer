"""DEVELOPMENT (data already seen, not evidence): logistic graph estimator on the Sheng & Lu data.

The ecosystem of a new version v' is the set of models released before v', excluding v' and its predecessor v,
which enters separately as a covariate. Release dates are approximate and were fixed before computing anything.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt import estimators as E  # noqa: E402
from cvt import graph as Gr  # noqa: E402

DATA = Path(sys.argv[1]) / "results"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 400
M = 20
RELEASE = {  # approximate release date (year-month-day)
    "Llama-2-7b": "2023-07-18", "Mistral-7B-v0.2": "2023-12-11", "gemma-1.1-7b": "2024-04-05",
    "Llama-3-8B": "2024-04-18", "Mistral-7B-v0.3": "2024-05-22", "Qwen2-7B": "2024-06-07",
    "gemma-2-9b": "2024-06-27", "Llama-3.1-8B": "2024-07-23", "Qwen2.5-7B": "2024-09-19", "Qwen3-8B": "2025-04-29",
}
PAIRS = {"qwen": ("Qwen2.5-7B", "Qwen3-8B"), "llama": ("Llama-3-8B", "Llama-3.1-8B"),
         "mistral": ("Mistral-7B-v0.2", "Mistral-7B-v0.3"), "qwen2": ("Qwen2-7B", "Qwen2.5-7B"),
         "gemma12": ("gemma-1.1-7b", "gemma-2-9b"), "llama23": ("Llama-2-7b", "Llama-3-8B")}
BENCHES = ["mcq_n3000", "gpqa", "gsm8k", "mathfull", "humaneval", "mbpp"]


def outcomes(bench):
    """Outcome vector of each of the 10 models on ``bench`` (questions aligned across models, checked)."""
    res = {}
    for pair, (old, new) in PAIRS.items():
        rows = list(csv.DictReader((DATA / f"results_{pair}_{bench}.csv").open()))
        res[old] = np.array([r["old_correct"] == "1" for r in rows])
        res[new] = np.array([r["new_correct"] == "1" for r in rows])
    return res


def main():
    """Replay every pilot transition and print gains and coverages of random testing, BMA and the graph."""
    rng = np.random.default_rng(20261110)
    out = []
    for bench in BENCHES:
        res = outcomes(bench)
        for pair, (old, new) in PAIRS.items():
            y_prev, y_new = res[old], res[new]
            aux_names = [k for k in res if k not in (old, new) and RELEASE[k] < RELEASE[new]]
            aux = np.array([res[k] for k in aux_names]) if aux_names else None
            X = Gr.ecosystem_features(aux, y_prev)
            n, p = len(y_new), y_new.mean()
            er, eb, eg, cb, cg, cr = [], [], [], [], [], []
            for _ in range(R):
                idx = rng.choice(n, M, replace=False)
                a, b = y_prev[idx], y_new[idx]
                f = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
                r = E.reset(*f, rng=rng, draws=1500, prior="hybrid")
                x = E.double_sampling_bma(int(y_prev.sum()), int((~y_prev).sum()), *f, rng=rng, draws=1500)
                g = Gr.graph_logit(X, idx, b, rng=rng, draws=3000)
                er.append((r.mean - p) ** 2); eb.append((x.mean - p) ** 2); eg.append((g.mean - p) ** 2)
                cr.append(r.lo <= p <= r.hi); cb.append(x.lo <= p <= x.hi); cg.append(g.lo <= p <= g.hi)
            mr = np.mean(er)
            out.append(dict(pair=pair, bench=bench, in_scope=pair != "llama23", n_aux=len(aux_names), p_new=p,
                            gain_bma=mr / np.mean(eb), gain_graph=mr / np.mean(eg),
                            cov_reset=np.mean(cr), cov_bma=np.mean(cb), cov_graph=np.mean(cg)))
            o = out[-1]
            print(f"{pair:8s} {bench:10s} aux={o['n_aux']} p'={p:.2f} | BMA x{o['gain_bma']:.2f} | graphe x{o['gain_graph']:.2f} | couv graphe {o['cov_graph']:.3f}", flush=True)
    dst = Path(__file__).resolve().parents[1] / "results" / "dev_graph_sheng.csv"
    with dst.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    S = [o for o in out if o["in_scope"]]
    gb = np.array([o["gain_bma"] for o in S]); gg = np.array([o["gain_graph"] for o in S])
    print(f"\nPérimètre (30) : BMA gain médian x{np.median(gb):.3f} (min x{gb.min():.2f}) | graphe gain médian x{np.median(gg):.3f} "
          f"(min x{gg.min():.2f}) | graphe > BMA sur {np.mean(gg > gb):.0%} | couverture graphe moy {np.mean([o['cov_graph'] for o in S]):.3f} "
          f"min {min(o['cov_graph'] for o in S):.3f}")


if __name__ == "__main__":
    main()
