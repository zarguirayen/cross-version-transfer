"""Tests du noyau : journal, graphe, plan, estimate, decide, lint, pages."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt.kernel import Kernel  # noqa: E402


def _world(tmp_path, n_tasks=200, n_old=12, seed=0):
    g = np.random.default_rng(seed)
    diff = g.normal(0, 1.5, n_tasks)
    tasks = [f"t{i:03d}" for i in range(n_tasks)]
    K = Kernel(tmp_path)
    K.register_tasks(tasks)
    K.register_slot("coder", "corriger des tickets")
    K.register_slot("other", "agents de référence")
    for j in range(n_old):
        v = f"ref{j}"
        K.register_version("other", v, f"2024-01-{j + 1:02d}")
        y = g.random(n_tasks) < 1 / (1 + np.exp(-(g.normal(-0.3, 1) + 1.3 * diff)))
        for t, s in zip(tasks, y):
            K.record(v, t, bool(s))
    K.register_version("coder", "v1", "2024-02-01")
    y1 = g.random(n_tasks) < 1 / (1 + np.exp(-(-0.5 + 1.3 * diff)))
    for t, s in zip(tasks, y1):
        K.record("v1", t, bool(s))
    K.register_version("coder", "v2", "2024-03-01", parent="v1")
    y2 = g.random(n_tasks) < 1 / (1 + np.exp(-(0.3 + 1.3 * diff)))
    return K, tasks, y1, y2


def test_journal_replay_restores_state(tmp_path):
    K, tasks, _, _ = _world(tmp_path)
    K2 = Kernel(tmp_path)
    assert K2.versions.keys() == K.versions.keys()
    assert K2.results["v1"] == K.results["v1"]


def test_plan_estimate_decide(tmp_path):
    K, tasks, y1, y2 = _world(tmp_path)
    assert K.estimate("v2")["status"] == "needs_probes"
    plan = K.plan("v2", m=20)
    assert len(plan) == 20 and len(set(plan)) == 20
    truth = dict(zip(tasks, y2))
    for t in plan:
        K.record("v2", t, bool(truth[t]), role="probe")
    e = K.estimate("v2")
    assert e["status"] == "estimated" and e["method"] == "graph_greg_cf" and 0 <= e["lo"] <= e["mean"] <= e["hi"] <= 1
    d = K.decide("v2")
    assert d["action"] in ("promote", "keep") and d["current"] == "v1"
    assert K.lint() == []
    pages = K.render()
    assert pages and "v2" in pages[0].read_text()


def test_lint_flags_probe_outside_plan_and_bad_parent(tmp_path):
    K, tasks, _, y2 = _world(tmp_path)
    plan = K.plan("v2", m=20)
    outside = next(t for t in tasks if t not in plan)
    K.record("v2", outside, True, role="probe")
    K.register_version("other", "late", "2025-01-01")
    K.register_version("coder", "v0", "2024-04-01", parent="late")
    codes = {f["code"] for f in K.lint()}
    assert {"L1", "L2", "L3"} <= codes


def test_record_rejects_unknown_objects(tmp_path):
    K, tasks, _, _ = _world(tmp_path)
    with pytest.raises(KeyError):
        K.record("nope", tasks[0], True)
    with pytest.raises(KeyError):
        K.record("v1", "unknown-task", True)
    with pytest.raises(ValueError):
        K.record("v1", tasks[0], True, role="bad")
