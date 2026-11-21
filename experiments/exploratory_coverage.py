"""EXPLORATORY, RETROSPECTIVE (protocol: docs/EXPLORATORY_COVERAGE.md, committed before this script was run).

Coverage of the SWE-bench-family outcome matrices of studies v3, v4 and v5: how many 0s are not evaluated
attempts (no patch, no log, never run), and how much treating them as missing moves the inputs of the
predecessor law (p_new, phi², law gain at m = 20). Matrices, lineages and transitions come from the frozen
loaders, unchanged.

Usage
-----
    python experiments/exploratory_coverage.py /home/claude/data/swe/experiments
Outputs: results/exploratory_coverage/{status_counts.csv, transitions.csv, summary.txt}
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from cvt import estimators as E  # noqa: E402
import v3_swebench_data as D3  # noqa: E402
import v4_lite_data as D4  # noqa: E402
import v5_general_data as D5  # noqa: E402

ROOT = Path(sys.argv[1])
OUT = HERE.parent / "results" / "exploratory_coverage"
M = 20
STATUSES = ("success", "attempted_failure", "no_patch", "no_log", "not_run")


def statuses(d: Path, ids, y):
    """Status of every task of ``ids`` for the submission directory ``d`` (rules of the protocol)."""
    st = {i: "attempted_failure" for i in ids}
    det, rj = d / "per_instance_details.json", d / "results" / "results.json"
    if det.exists():
        dd = json.load(det.open())
        for i in ids:
            if i not in dd:
                st[i] = "not_run"
    elif rj.exists():
        r = json.load(rj.open())
        L = lambda *ks: set().union(*[set(x for x in r.get(k, []) if isinstance(x, str)) for k in ks])
        if "submitted_ids" in r:                                   # Multi-SWE-bench
            sub = L("submitted_ids")
            for i in ids:
                if i not in sub:
                    st[i] = "not_run"
            for i in L("error_ids", "incomplete_ids") & set(ids):
                st[i] = "no_log"
            for i in L("empty_patch_ids", "empty_error_patch_ids") & set(ids):
                st[i] = "no_patch"
        else:                                                      # SWE-bench
            if "generated" in r:
                gen, ng = L("generated"), L("no_generation")
                for i in ids:
                    if i not in gen and i not in ng:
                        st[i] = "not_run"
                if "with_logs" in r:
                    wl = L("with_logs")
                    for i in gen & set(ids):
                        if i not in wl:
                            st[i] = "no_log"
            for i in L("no_logs") & set(ids):
                st[i] = "no_log"
            for i in L("no_generation") & set(ids):
                st[i] = "no_patch"
    for i, v in zip(ids, y):
        if v:
            st[i] = "success"
    return np.array([st[i] for i in ids])


def tasksets():
    """Yield (study, taskset, ids, Y, dirs, transitions) with the frozen loaders."""
    ids, Y, dates, lin, _ = D3.load(ROOT)
    yield "v3", "verified", ids, Y, ROOT / "evaluation" / "verified", [(t[-2], t[-1]) for t in D3.transitions(dates, lin)]
    ids, Y, dates, lin = D4.load(ROOT)
    yield "v4", "lite", ids, Y, ROOT / "evaluation" / "lite", [(t[-2], t[-1]) for t in D4.transitions(dates, lin)]
    for ts in D5.TASKSETS:
        ids, Y, dates, lin = D5.load(ts)
        name = ts.split("-", 1)[1]
        base = D5.MSWE_ROOT / name / "verified" if name in D5.MSWE_LANGS else D5.SWE_ROOT / name
        yield "v5", ts, ids, Y, base, [(t[-2], t[-1]) for t in D5.transitions(dates, lin)]


def law_inputs(a, b):
    """p_old, p_new, phi², law gain at m = 20 on the kept tasks."""
    n11, n10 = int((a & b).sum()), int((a & ~b).sum())
    n01, n00 = int((~a & b).sum()), int((~a & ~b).sum())
    phi2 = E.phi_signed(n11, n10, n01, n00) ** 2
    return float(a.mean()), float(b.mean()), phi2, float(E.gain_law(phi2, M, len(a)))


def main():
    """Count statuses, run the sensitivity analysis, write the outputs."""
    OUT.mkdir(parents=True, exist_ok=True)
    counts, rows = [], []
    for study, ts, ids, Y, base, trans in tasksets():
        S = {d: statuses(base / d, ids, Y[d]) for d in Y}
        c = {s: int(sum((S[d] == s).sum() for d in S)) for s in STATUSES}
        counts.append(dict(study=study, taskset=ts, agents=len(Y), tasks=len(ids), cells=len(Y) * len(ids), **c))
        for o, n in trans:
            a, b = Y[o].astype(bool), Y[n].astype(bool)
            base_vals = law_inputs(a, b)
            row = dict(study=study, taskset=ts, old=o, new=n, N=len(ids))
            for lab, drop in (("s1", ("no_log", "not_run")), ("s2", ("no_patch", "no_log", "not_run"))):
                keep = ~(np.isin(S[o], drop) | np.isin(S[n], drop))
                row[f"removed_{lab}"] = int((~keep).sum())
                v = law_inputs(a[keep], b[keep]) if keep.sum() >= M else (np.nan,) * 4
                row[f"dp_new_{lab}"] = v[1] - base_vals[1]
                row[f"dphi2_{lab}"] = v[2] - base_vals[2]
                row[f"rgain_{lab}"] = v[3] / base_vals[3] - 1
            rows.append(row)
    with (OUT / "status_counts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=counts[0].keys()); w.writeheader(); w.writerows(counts)
    with (OUT / "transitions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    L = []
    for lab, sel in (("v3", ("v3",)), ("v4", ("v4",)), ("v5", ("v5",)), ("all", ("v3", "v4", "v5"))):
        C = [c for c in counts if c["study"] in sel]
        cells = sum(c["cells"] for c in C)
        part = {s: sum(c[s] for c in C) for s in STATUSES}
        L.append(f"{lab:3s} cells {cells:6d} | " + " | ".join(f"{s} {part[s]} ({100 * part[s] / cells:.2f}%)" for s in STATUSES))
    for lab in ("s1", "s2"):
        T = [r for r in rows]
        hit = [r for r in T if r[f"removed_{lab}"] > 0]
        f = lambda k: np.abs(np.array([r[k] for r in T], float))
        L.append(f"{lab}: transitions {len(T)}, with removed cells {len(hit)} | |d p_new| median {np.nanmedian(f(f'dp_new_{lab}')):.4f} "
                 f"max {np.nanmax(f(f'dp_new_{lab}')):.4f} | |d phi2| median {np.nanmedian(f(f'dphi2_{lab}')):.4f} max {np.nanmax(f(f'dphi2_{lab}')):.4f} "
                 f"| |rel d gain| median {np.nanmedian(f(f'rgain_{lab}')):.4f} max {np.nanmax(f(f'rgain_{lab}')):.4f}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
