"""PROSPECTIVE PRE-REGISTERED analysis v9: the difficulty-only deadline estimator on agents not yet released.

Frozen with docs/PREREGISTRATION_V9.md (tag prereg-v9) on 2026-11-09, before any of the agents it will analyse
existed on the AppWorld leaderboard. Confirmation set: every agent whose leaderboard date is after 2026-11-04 (the
last snapshot analysed in v8), on both splits. Graph of each agent: every agent released strictly before it.

Run rule (frozen): run once, on the first leaderboard snapshot that holds at least 5 such agents, or on the
snapshot of 2027-05-01 if that comes first; with fewer than 5 agents the results are reported as descriptive only.

Usage
-----
    python experiments/v9_prospective_deadline.py <aw_root> <aw_snapshot_dir> <leaderboard.json> [R]
    python experiments/v9_prospective_deadline.py --fake [R]      # dry run on synthetic data
Outputs: results/v9/{evaluations.csv, summary.txt}
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from cvt import asynchronous as A  # noqa: E402
from cvt import estimators as E  # noqa: E402
from cvt import graph as Gr  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FAKE = "--fake" in sys.argv
R = int(ARGS[3]) if len(ARGS) > 3 else (int(ARGS[0]) if FAKE and ARGS else 200)
CUTOFF = "2026-11-04"
SEED = 20261127
M, K, MIN_AGENTS = 20, 10, 5
SPLITS = ("test_normal", "test_challenge")
OUT = HERE.parent / "results" / ("v9_fake" if FAKE else "v9")
_STEP = re.compile(r"^### Environment Interaction", re.M)
_REPORT = re.compile(r"Num Passed Tests : (\d+)\s+Num Failed Tests : (\d+)\s+Num Total\s+Tests : (\d+)")


def load_agent(d: Path, split: str):
    """Task ids, success and steps of one unpacked agent directory (same rules as v8)."""
    ev = d / "evaluations" / f"{split}.json"
    if ev.exists():
        ind = json.load(ev.open())["individual"]; ids = sorted(ind)
        y = np.array([bool(ind[t]["success"]) for t in ids])
    else:
        ids = sorted(p.name for p in (d / "tasks").iterdir()); y = []
        for t in ids:
            m = _REPORT.search((d / "tasks" / t / "evaluation" / "report.md").read_text())
            y.append(m is not None and int(m.group(1)) == int(m.group(3)) > 0 and int(m.group(2)) == 0)
        y = np.array(y)
    st = np.array([len(_STEP.findall((d / "tasks" / t / "logs" / "environment_io.md").read_text()))
                   if (d / "tasks" / t / "logs" / "environment_io.md").exists() else 0 for t in ids])
    return ids, y, st


def discover(aw_root: Path, snap: Path, board: Path):
    """All agents with a date: the paper's (aw_root) and every unpacked leaderboard agent (snap). Dates come from the
    leaderboard file (matched on method and model names), else from the agent's metadata.json."""
    entries = json.load(board.open())
    by_name = {(e["method"]["name"], e["llm"]["name"]): e["date"] for e in entries}
    agents = {}
    for root in (aw_root, snap):
        for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_test_normal")):
            name = d.name[: -len("_test_normal")]
            if not (root / f"{name}_test_challenge").exists() or not (d / "metadata.json").exists():
                continue
            md = json.load((d / "metadata.json").open())
            date = by_name.get((md["method"]["name"], md["llm"]["name"]), md.get("date", ""))
            agents[name] = (date, root)
    return agents


def fake_world(seed):
    """Synthetic dry-run world: 30 agents with dates, a shared difficulty, steps longer on hard tasks."""
    g = np.random.default_rng(seed); N = 200; d = g.normal(0, 1.5, N)
    dates = [f"2026-{1 + i // 3:02d}-{1 + i % 28:02d}" for i in range(24)] + [f"2026-12-{10 + i:02d}" for i in range(6)]
    data = {}
    for i, dt in enumerate(dates):
        y = g.random(N) < 1 / (1 + np.exp(-(g.normal(0, 1.2) + d)))
        st = np.maximum(1, (g.gamma(2, 8, N) * np.exp(-0.4 * d)).astype(int))
        data[f"agent{i:02d}"] = (dt, [f"t{j}" for j in range(N)], y, st)
    return data


def build(split):
    """Agent -> (date, ids, y, steps) for one split."""
    if FAKE:
        return fake_world(SPLITS.index(split) + 1)
    aw_root, snap, board = Path(ARGS[0]), Path(ARGS[1]), Path(ARGS[2])
    out = {}
    for name, (date, root) in discover(aw_root, snap, board).items():
        ids, y, st = load_agent(root / f"{name}_{split}", split)
        out[name] = (date, ids, y, st)
    ids0 = next(iter(out.values()))[1]
    assert all(v[1] == ids0 for v in out.values()), "task ids differ"
    return out


def run(split, a, data, kc):
    """R replays of one new agent: GREG-cf vs random testing, and the deadline estimators at k = 10."""
    date, ids, y, s = data[a]
    eco = [w for w, v in data.items() if v[0] < date]
    Ye, Se = np.array([data[w][2] for w in eco]), np.array([data[w][3] for w in eco], float)
    X = Gr.ecosystem_features(Ye, np.zeros(len(ids), bool))
    b, th, _, _ = A.fit_slow_model(Ye, Se)
    N, p = len(ids), float(np.mean(y)); rng = np.random.default_rng([SEED, kc])
    err_u, err_g, cov_g, bn, bz, cz = [], [], [], [], [], []
    for _ in range(R):
        iu = rng.choice(N, M, replace=False); su = int(y[iu].sum())
        err_u.append((E.reset(su, 0, 0, M - su, rng=rng, draws=1500, prior="hybrid").mean - p) ** 2)
        blocks = np.array_split(np.argsort(X[:, 0], kind="stable"), M)
        idx = np.array([rng.choice(bb) for bb in blocks]); w = np.array([len(bb) for bb in blocks], float)
        e = Gr.graph_greg_cf(X, idx, y[idx], w, rng=rng, draws=1500)
        err_g.append((e.mean - p) ** 2); cov_g.append(e.lo <= p <= e.hi)
        ret = np.zeros(M, bool); ret[np.lexsort((rng.random(M), s[idx]))[:K]] = True
        bn.append(y[idx][ret].mean() - p)
        ez = A.deadline_estimate(idx, ret, y[idx], w, b, th, 0.0, np.zeros(2), N)     # difficulty only
        bz.append(ez.mean - p); cz.append(ez.lo <= p <= ez.hi)
    bn, bz = np.array(bn), np.array(bz)
    return dict(split=split, agent=a, date=date, n_eco=len(eco), p_new=p, gain_greg_cf=np.mean(err_u) / np.mean(err_g),
                cov_greg_cf=float(np.mean(cov_g)), bias_naive=bn.mean(), bias_zero=bz.mean(),
                rmse_naive=float(np.sqrt((bn ** 2).mean())), rmse_zero=float(np.sqrt((bz ** 2).mean())),
                cov_zero=float(np.mean(cz)))


def wil(x):
    """One-sided Wilcoxon signed-rank p-value (> 0)."""
    x = np.asarray(x, float)
    return float(stats.wilcoxon(x, alternative="greater").pvalue) if np.any(x != 0) else float("nan")


def main():
    """Select the agents after the cutoff, replay them, apply the frozen criteria."""
    OUT.mkdir(parents=True, exist_ok=True)
    cutoff = "2026-12-01" if FAKE else CUTOFF
    rows = []
    for split in SPLITS:
        data = build(split)
        new = sorted(a for a, v in data.items() if v[0] > cutoff)
        for kc, a in enumerate(new):
            rows.append(run(split, a, data, kc + 100 * SPLITS.index(split)))
            print(f"done: {split} {a}", flush=True)
    if not rows:
        txt = f"v9 — no agent released after {cutoff} in this snapshot: nothing to test yet."
        (OUT / "summary.txt").write_text(txt + "\n"); print(txt); return
    with (OUT / "evaluations.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    f = lambda k: np.array([r[k] for r in rows], float)
    n_agents = len({r["agent"] for r in rows})
    conf = n_agents >= MIN_AGENTS
    L = [f"v9 — {n_agents} agents released after {cutoff}, {len(rows)} evaluations ; "
         f"{'confirmatory' if conf else 'DESCRIPTIVE ONLY (fewer than 5 agents)'}"]
    g = f("gain_greg_cf"); p0 = wil(np.log(g))
    L.append(f"V1-v9 — method beats random testing: gain median x{np.median(g):.3f} ; p = {p0:.3g} ; supported (median > 1 and p < 0.05): {conf and np.median(g) > 1 and p0 < 0.05}")
    bn, bz = f("bias_naive"), f("bias_zero"); p1 = wil(bn)
    L.append(f"D1-v9 — deadline bias exists: naive bias median {np.median(bn):+.3f} ; p = {p1:.3g} ; supported (median > 0 and p < 0.05): {conf and np.median(bn) > 0 and p1 < 0.05}")
    ratio = f("rmse_naive") / f("rmse_zero"); p2 = wil(f("rmse_naive") - f("rmse_zero"))
    L.append(f"D2-v9 — difficulty-only correction reduces the error: RMSE ratio naive/corrected median {np.median(ratio):.2f} ; p = {p2:.3g} ; supported (ratio > 1 and p < 0.05): {conf and np.median(ratio) > 1 and p2 < 0.05}")
    cz = f("cov_zero")
    L.append(f"D3-v9 — calibrated at the deadline: coverage mean {cz.mean():.3f} (min {cz.min():.3f}) ; supported (>= 0.85): {conf and cz.mean() >= 0.85}")
    txt = "\n".join(L); (OUT / "summary.txt").write_text(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
