"""Tests of the kernel: event log, graph, plan, estimate, decide, lint and pages."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvt.kernel import Kernel  # noqa: E402


def _world(tmp_path, n_tasks=200, n_old=12, seed=0):
    """Build a small synthetic graph in ``tmp_path``.

    Twelve reference agents and a version ``v1`` of slot ``coder`` are fully measured on ``n_tasks`` tasks drawn
    from a logistic model with a shared task difficulty; ``v2`` (parent ``v1``) is registered but not measured.

    Returns
    -------
    K : Kernel
    tasks : list of str
    y1, y2 : ndarray of bool
        True outcomes of ``v1`` and ``v2`` on every task.
    """
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
    """A new kernel opened on the same directory rebuilds the same state from the log."""
    K, tasks, _, _ = _world(tmp_path)
    K2 = Kernel(tmp_path)
    assert K2.versions.keys() == K.versions.keys()
    assert K2.results["v1"] == K.results["v1"]


def test_plan_estimate_decide(tmp_path):
    """End-to-end: plan 20 probes, record them, estimate with an ordered interval, decide, lint and render."""
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
    """Lint reports a parent in another slot (L1), a parent released later (L2) and a probe outside the plan (L3)."""
    K, tasks, _, y2 = _world(tmp_path)
    plan = K.plan("v2", m=20)
    outside = next(t for t in tasks if t not in plan)
    K.record("v2", outside, True, role="probe")
    K.register_version("other", "late", "2025-01-01")
    K.register_version("coder", "v0", "2024-04-01", parent="late")
    codes = {f["code"] for f in K.lint()}
    assert {"L1", "L2", "L3"} <= codes


def test_record_rejects_unknown_objects(tmp_path):
    """Recording a result for an unknown version or task, or with an invalid role, raises."""
    K, tasks, _, _ = _world(tmp_path)
    with pytest.raises(KeyError):
        K.record("nope", tasks[0], True)
    with pytest.raises(KeyError):
        K.record("v1", "unknown-task", True)
    with pytest.raises(ValueError):
        K.record("v1", tasks[0], True, role="bad")


# ---------------------------------------------------------------- kernel 0.2 (review fixes)
from cvt.kernel import as_outcome  # noqa: E402


def test_outcome_parsing_is_strict(tmp_path):
    """'false' is a failure, not a success; ambiguous values are rejected."""
    assert as_outcome("false") is False and as_outcome(" TRUE ") is True and as_outcome(1) is True
    for bad in ("no", "", 2, None, 0.5):
        with pytest.raises(ValueError):
            as_outcome(bad)
    K, tasks, y1, y2 = _world(tmp_path)
    K.record("v2", tasks[0], "false")
    assert K.results["v2"][tasks[0]] is False


def test_plan_is_bound_to_the_task_suite(tmp_path):
    """Adding tasks after a plan makes estimate refuse the plan (stale) and lint report L6."""
    K, tasks, y1, y2 = _world(tmp_path)
    probes = K.plan("v2", m=20)
    K.record_many("v2", {t: bool(y2[tasks.index(t)]) for t in probes}, role="probe")
    assert K.estimate("v2")["status"] == "estimated"
    K.register_tasks(["new-task"])
    assert K.estimate("v2")["status"] == "stale_plan"
    assert K.decide("v2")["action"] == "stale_plan"
    assert any(f["code"] == "L6" for f in K.lint())


def test_odd_budget_is_refused(tmp_path):
    """The paired-strata variance needs an even number of probes."""
    K, *_ = _world(tmp_path)
    with pytest.raises(ValueError):
        K.plan("v2", m=21)


def test_confidence_agrees_with_action(tmp_path):
    """A decision is never 'sure' when the interval lies on the other side of the threshold than the action."""
    K, tasks, y1, y2 = _world(tmp_path)
    probes = K.plan("v2", m=20)
    K.record_many("v2", {t: bool(y2[tasks.index(t)]) for t in probes}, role="probe")
    d = K.decide("v2")
    lo, hi = d["p_new_interval"]
    thr = d["p_current"]
    if d["confident"]:
        assert (d["action"] == "promote" and lo > thr) or (d["action"] == "keep" and hi < thr)
    assert d["estimators_agree"] or not d["confident"]


def test_repeated_runs_are_kept_and_conflicts_flagged(tmp_path):
    """Every run is kept with its provenance; disagreeing runs raise L7 and the latest run is used."""
    K, tasks, y1, y2 = _world(tmp_path)
    K.record("v2", tasks[0], True, origin="benchmark", verifier="harness-a")
    K.record("v2", tasks[0], False, origin="production", verifier="harness-b")
    runs = K.runs[("v2", tasks[0])]
    assert [r["verifier"] for r in runs] == ["harness-a", "harness-b"] and K.results["v2"][tasks[0]] is False
    assert any(f["code"] == "L7" for f in K.lint())
