"""Minimal kernel for the graph of agents.

The kernel observes; it does not orchestrate. Each node is a *version* of a *slot* (an agent, a model, a tool);
lineage edges link a version to its predecessor; the evidence is a set of task results. Every change is written to
an append-only event log (``events.jsonl``) and the in-memory state is rebuilt by replaying that log, so the log is
the single source of truth.

Operations
----------
register_slot, register_version, register_tasks
    Build the graph: slots, versions with their release date and lineage, and the shared task suite.
record, record_many
    Store task results (one at a time, or a whole evaluation), with their role and origin.
plan
    Let the graph choose which tasks to test for a new version: a stratified design on the task difficulty seen by
    earlier versions of every slot (paper, Section 4, "Choose").
estimate
    Success rate of a version with an interval, by cross-fitted GREG on the planned probes (paper, Section 4,
    "Estimate"); falls back to a Jeffreys-type random-testing estimate when the graph is empty.
decide
    Keep the current version or promote the new one, and say whether the decision is sure (paper, Section 4,
    "Decide").
lint
    Consistency checks on the log (codes L1 to L5).
render
    One Markdown page per slot.

Examples
--------
>>> k = Kernel("state")                                    # doctest: +SKIP
>>> k.register_slot("coder", "fixes GitHub issues")        # doctest: +SKIP
>>> k.register_version("coder", "coder-v2", "2026-03-01", parent="coder-v1")   # doctest: +SKIP
>>> probes = k.plan("coder-v2", m=20)                      # doctest: +SKIP
>>> k.record_many("coder-v2", {t: run(t) for t in probes}, role="probe")      # doctest: +SKIP
>>> k.decide("coder-v2")                                   # doctest: +SKIP
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import estimators as E
from . import graph as Gr

#: Kernel version. 0.2 (after the paper's v0.4 review): strict outcome parsing, plans bound to the task suite,
#: decisions whose confidence agrees with the action, even probe budgets, repeated runs and provenance kept.
__version__ = "0.2.0"

#: Allowed roles of a result. ``probe`` results come from a plan; ``train`` marks tasks used for training, which
#: must never serve as evidence (lint L5).
ROLES = {"probe", "full", "train", "other"}
#: Allowed origins of a result.
ORIGINS = {"production", "benchmark", "synthetic"}


def as_outcome(v) -> bool:
    """Parse one task outcome strictly.

    Accepts booleans, the integers 0 and 1, and the strings "true"/"false" (any case, surrounding spaces ignored).
    Anything else raises ``ValueError``: ``bool("false")`` is ``True`` in Python, so a lenient conversion would
    silently record failures as successes.
    """
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    raise ValueError(f"résultat invalide : {v!r} (attendu : booléen, 0/1, 'true' ou 'false')")


def suite_hash(ids) -> str:
    """SHA-256 of the sorted task ids: the identity of the task suite a plan was made on.

    It detects tasks added or removed, not a task whose content or verifier changed under the same id; version the
    task ids (or the suite name) when tasks are edited.
    """
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


@dataclass
class Version:
    """A node of the graph.

    Attributes
    ----------
    slot : str
        Slot the version belongs to.
    id : str
        Unique version identifier.
    parent : str or None
        Predecessor in the same slot (lineage edge), or ``None`` for a first version.
    released_at : str
        Release date in ISO format (``YYYY-MM-DD``); string comparison gives chronological order.
    meta : dict
        Free-form metadata (model, scaffold, prompt, ...).
    """

    slot: str
    id: str
    parent: str | None
    released_at: str
    meta: dict = field(default_factory=dict)


class Kernel:
    """Graph of agents backed by an append-only event log.

    Parameters
    ----------
    root : str or Path
        Directory holding ``events.jsonl``; created if missing. An existing log is replayed on construction.

    Attributes
    ----------
    slots : dict
        Slot name -> ``{"responsibility": str}``.
    versions : dict
        Version id -> :class:`Version`.
    results : dict
        Version id -> {task id -> success}.
    roles : dict
        (version id, task id) -> role of the stored result.
    plans : dict
        Version id -> stored probe plan (tasks, stratum weights, predecessor, seed).
    tasks : set
        The shared task suite.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / "events.jsonl"
        self.slots: dict[str, dict] = {}
        self.versions: dict[str, Version] = {}
        self.results: dict[str, dict[str, bool]] = {}      # version -> task -> success
        self.roles: dict[tuple[str, str], str] = {}
        self.plans: dict[str, dict] = {}
        self.tasks: set[str] = set()
        self.runs: dict[tuple[str, str], list[dict]] = {}   # every recorded run, in log order
        self._replay()

    # ------------------------------------------------------------------ event log
    def _append(self, ev: dict):
        """Timestamp an event, append it to the log, then apply it to the in-memory state."""
        ev = {"ts": time.time(), **ev}
        with self.log.open("a") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self._apply(ev)

    def _replay(self):
        """Rebuild the state by applying every event of the log in order.

        Raises
        ------
        ValueError
            If a line is not valid JSON or carries an invalid outcome; the message gives the line number, so a
            corrupted or hand-edited log is detected instead of being silently misread.
        """
        if self.log.exists():
            for no, line in enumerate(self.log.open(), 1):
                if line.strip():
                    try:
                        self._apply(json.loads(line))
                    except (ValueError, KeyError) as exc:
                        raise ValueError(f"journal invalide, ligne {no} : {exc}") from exc

    def _apply(self, ev: dict):
        """Apply one event to the in-memory state (no validation: events were validated when written)."""
        k = ev["event"]
        if k == "slot":
            self.slots[ev["slot"]] = {"responsibility": ev.get("responsibility", "")}
        elif k == "version":
            self.versions[ev["version"]] = Version(ev["slot"], ev["version"], ev.get("parent"), ev["released_at"],
                                                   ev.get("meta", {}))
            self.results.setdefault(ev["version"], {})
        elif k == "task":
            self.tasks.add(ev["task"])
        elif k == "results":
            # Bulk import: one event carries a whole evaluation.
            self.results.setdefault(ev["version"], {}).update({t: as_outcome(v) for t, v in ev["outcomes"].items()})
            for t, v in ev["outcomes"].items():
                self.roles[(ev["version"], t)] = ev.get("role", "full")
                self.runs.setdefault((ev["version"], t), []).append(
                    {"success": as_outcome(v), "role": ev.get("role", "full"), "origin": ev.get("origin"), "verifier": ev.get("verifier"), "ts": ev.get("ts")})
        elif k == "result":
            self.results.setdefault(ev["version"], {})[ev["task"]] = as_outcome(ev["success"])
            self.roles[(ev["version"], ev["task"])] = ev.get("role", "other")
            self.runs.setdefault((ev["version"], ev["task"]), []).append(
                {"success": as_outcome(ev["success"]), "role": ev.get("role", "other"), "origin": ev.get("origin"),
                 "verifier": ev.get("verifier"), "ts": ev.get("ts")})
        elif k == "plan":
            self.plans[ev["version"]] = {"tasks": ev["tasks"], "weights": ev["weights"], "against": ev["against"],
                                         "seed": ev["seed"], "suite_hash": ev.get("suite_hash")}

    # ------------------------------------------------------------------ graph
    def register_slot(self, slot: str, responsibility: str = ""):
        """Declare a slot (idempotent).

        Parameters
        ----------
        slot : str
            Slot name.
        responsibility : str, optional
            One-line description shown on the slot's page.
        """
        if slot not in self.slots:
            self._append({"event": "slot", "slot": slot, "responsibility": responsibility})

    def register_version(self, slot: str, version: str, released_at: str, parent: str | None = None, **meta):
        """Add a version (a node) to a slot.

        Parameters
        ----------
        slot : str
            An already registered slot.
        version : str
            New, unique version id.
        released_at : str
            Release date, ISO format. Only versions released strictly earlier enter its graph.
        parent : str, optional
            Predecessor version (lineage edge); must already be registered.
        **meta
            Free-form metadata stored with the version.

        Raises
        ------
        KeyError
            If the slot or the parent is unknown.
        ValueError
            If the version id is already used.
        """
        if slot not in self.slots:
            raise KeyError(f"slot inconnu : {slot}")
        if parent is not None and parent not in self.versions:
            raise KeyError(f"version parente inconnue : {parent}")
        if version in self.versions:
            raise ValueError(f"version déjà enregistrée : {version}")
        self._append({"event": "version", "slot": slot, "version": version, "parent": parent,
                      "released_at": released_at, "meta": meta})

    def register_tasks(self, tasks):
        """Add tasks to the shared suite; tasks already known are ignored.

        Parameters
        ----------
        tasks : iterable of str
            Task identifiers.
        """
        for t in sorted(set(tasks) - self.tasks):
            self._append({"event": "task", "task": t})

    def record(self, version: str, task: str, success: bool, role: str = "full", origin: str = "benchmark",
               verifier: str = "unknown"):
        """Store one task result.

        Parameters
        ----------
        version, task : str
            Registered version and task.
        success : bool
            Whether the version solved the task.
        role : {"probe", "full", "train", "other"}
            Why the task was run. Use ``"probe"`` for tasks chosen by :meth:`plan`.
        origin : {"production", "benchmark", "synthetic"}
            Where the result comes from.
        verifier : str
            Free-form name of the checker that produced the outcome.

        Raises
        ------
        KeyError
            If the version or the task is unknown.
        ValueError
            If the role or the origin is not allowed.
        """
        if version not in self.versions:
            raise KeyError(f"version inconnue : {version}")
        if task not in self.tasks:
            raise KeyError(f"tâche inconnue : {task}")
        if role not in ROLES or origin not in ORIGINS:
            raise ValueError("rôle ou origine invalide")
        self._append({"event": "result", "version": version, "task": task, "success": as_outcome(success), "role": role,
                      "origin": origin, "verifier": verifier})

    def record_many(self, version: str, outcomes: dict, role: str = "full", origin: str = "benchmark",
                    verifier: str = "unknown"):
        """Store a whole evaluation at once (for example the full history of a version).

        Parameters
        ----------
        version : str
            Registered version.
        outcomes : dict
            Task id -> success. Every task must be registered.
        role, origin, verifier
            As in :meth:`record`.

        Raises
        ------
        KeyError
            If the version or any task is unknown.
        ValueError
            If the role or the origin is not allowed.
        """
        if version not in self.versions:
            raise KeyError(f"version inconnue : {version}")
        unknown = set(outcomes) - self.tasks
        if unknown:
            raise KeyError(f"{len(unknown)} tâches inconnues")
        if role not in ROLES or origin not in ORIGINS:
            raise ValueError("rôle ou origine invalide")
        self._append({"event": "results", "version": version, "outcomes": {t: as_outcome(v) for t, v in outcomes.items()},
                      "role": role, "origin": origin, "verifier": verifier})

    # ------------------------------------------------------------------ views
    def task_list(self) -> list[str]:
        """Return the task suite in a fixed (sorted) order, used to index every feature matrix."""
        return sorted(self.tasks)

    def _complete(self, v: str, ids: list[str]) -> bool:
        """Whether version ``v`` has a result on every task in ``ids``."""
        r = self.results.get(v, {})
        return all(t in r for t in ids)

    def ecosystem(self, version: str) -> list[str]:
        """Earlier versions that form the graph of a new version.

        Parameters
        ----------
        version : str
            The new version.

        Returns
        -------
        list of str
            Versions of any slot released strictly before ``version`` and measured on every task, excluding its
            parent, which enters the estimators as a separate covariate.
        """
        v = self.versions[version]; ids = self.task_list()
        return [w for w, x in self.versions.items() if x.released_at < v.released_at and w != v.parent
                and w != version and self._complete(w, ids)]

    def _features(self, version: str):
        """Build the task-level features the graph provides for ``version``.

        Returns
        -------
        ids : list of str
            Task order.
        X : ndarray of shape (N, d)
            Features from :func:`cvt.graph.ecosystem_features`; column 0 is the ecosystem difficulty.
        y_prev : ndarray of bool, shape (N,)
            Parent's outcomes (all ``False`` when there is no fully measured parent).
        eco : list of str
            The ecosystem, as returned by :meth:`ecosystem`.
        """
        ids = self.task_list()
        v = self.versions[version]
        eco = self.ecosystem(version)
        aux = np.array([[self.results[w][t] for t in ids] for w in eco]) if eco else None
        if v.parent and self._complete(v.parent, ids):
            y_prev = np.array([self.results[v.parent][t] for t in ids])
        else:
            y_prev = np.zeros(len(ids), bool)
        return ids, Gr.ecosystem_features(aux, y_prev), y_prev, eco

    # ------------------------------------------------------------------ plan
    def plan(self, version: str, m: int = 20, seed: int | None = None) -> list[str]:
        """Let the graph choose the ``m`` tasks to test for a new version.

        Tasks are sorted by the difficulty seen by the ecosystem, cut into ``m`` contiguous strata of (almost)
        equal size, and one task is drawn at random in each stratum. The plan is written to the log because its
        stratum sizes are the weights used by :meth:`estimate`.

        Parameters
        ----------
        version : str
            The new version.
        m : int, default 20
            Number of probes (and of strata). Must be even: the variance pairs adjacent strata.
        seed : int, optional
            Random seed. By default it is derived from a hash of the version id, so that the plan is reproducible.

        Returns
        -------
        list of str
            The tasks to run, one per stratum.
        """
        if m < 2 or m % 2:
            raise ValueError("m doit être pair : la variance compare des paires de strates voisines")
        ids, X, _, eco = self._features(version)
        if seed is None:
            seed = int(hashlib.sha256(version.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        # Without an ecosystem there is no difficulty scale: strata are then random, i.e. a uniform design.
        score = X[:, 0] if eco else rng.random(len(ids))
        blocks = np.array_split(np.argsort(score, kind="stable"), m)
        pick = [int(rng.choice(b)) for b in blocks]
        tasks = [ids[i] for i in pick]
        self._append({"event": "plan", "version": version, "tasks": tasks, "weights": [len(b) for b in blocks],
                      "against": self.versions[version].parent, "seed": seed, "n_ecosystem": len(eco),
                      "suite_hash": suite_hash(ids), "n_tasks": len(ids)})
        return tasks

    # ------------------------------------------------------------------ estimate
    def estimate(self, version: str, level: float = 0.90) -> dict:
        """Estimate the success rate of a version on the whole task suite.

        Parameters
        ----------
        version : str
            Version to estimate.
        level : float, default 0.90
            Confidence level of the interval.

        Returns
        -------
        dict
            ``status`` is one of

            - ``"measured"``: every task has a result; ``mean`` is the exact rate (method ``"census"``);
            - ``"needs_probes"``: no plan, or some planned probes have no result yet (listed in ``missing``);
            - ``"estimated"``: ``mean``, ``lo``, ``hi`` from cross-fitted GREG on the planned probes, or from the
              random-testing estimator when the ecosystem is empty;
            - ``"stale_plan"``: the task suite changed after the plan was made (its weights no longer apply).
        """
        ids, X, y_prev, eco = self._features(version)
        pl = self.plans.get(version)
        r = self.results.get(version, {})
        if self._complete(version, ids):                       # everything is measured: nothing to estimate
            p = float(np.mean([r[t] for t in ids]))
            return dict(version=version, status="measured", mean=p, lo=p, hi=p, n_probes=len(ids), method="census")
        if pl is not None and pl.get("suite_hash") and pl["suite_hash"] != suite_hash(ids):
            return dict(version=version, status="stale_plan",
                        reason="la suite de tâches a changé depuis le plan : refaire plan()")
        if pl is None or not all(t in r for t in pl["tasks"]):
            missing = [] if pl is None else [t for t in pl["tasks"] if t not in r]
            return dict(version=version, status="needs_probes", missing=missing)
        # Only the planned probes count: extra results would break the stratified weights (see lint L3).
        idx = np.array([ids.index(t) for t in pl["tasks"]])
        y = np.array([r[t] for t in pl["tasks"]])
        rng = np.random.default_rng(pl["seed"] + 1)
        if eco:
            e = Gr.graph_greg_cf(X, idx, y, np.array(pl["weights"], float), rng=rng, level=level)
        else:
            s = int(y.sum()); e = E.reset(s, 0, 0, len(y) - s, rng=rng, prior="hybrid", level=level)
        return dict(version=version, status="estimated", mean=e.mean, lo=e.lo, hi=e.hi, n_probes=len(idx),
                    method=e.method, n_ecosystem=len(eco), level=level)

    # ------------------------------------------------------------------ decide
    def decide(self, new: str, current: str | None = None, margin: float = 0.0) -> dict:
        """Decide whether to keep the current version or promote the new one.

        The decision uses the point estimate of the stratified graph estimator (:func:`cvt.graph.graph_v3`) on the
        planned probes; the cross-fitted GREG interval says whether the decision is sure, i.e. whether it excludes
        the current version's rate.

        Parameters
        ----------
        new : str
            Candidate version.
        current : str, optional
            Version in production; defaults to the parent of ``new``. Must be measured on every task.
        margin : float, default 0.0
            The new version is promoted if its estimate is at least ``p_current - margin``.

        Returns
        -------
        dict
            ``action`` is ``"promote"``, ``"keep"``, ``"needs_probes"``, ``"stale_plan"`` or ``"undetermined"``.
            When a decision is made, ``confident`` is true only if the GREG-cf interval excludes the threshold on the
            side of the action, and ``interval_not_contrary`` is false when the interval lies entirely on the other side (an
            interval that straddles the threshold neither contradicts nor confirms the action).
        """
        current = current or self.versions[new].parent
        ids, X, y_prev, eco = self._features(new)
        pl = self.plans.get(new)
        if pl is not None and pl.get("suite_hash") and pl["suite_hash"] != suite_hash(ids):
            return dict(new=new, current=current, action="stale_plan",
                        reason="la suite de tâches a changé depuis le plan : refaire plan()")
        if current is None or not self._complete(current, ids):
            return dict(new=new, current=current, action="undetermined", reason="version courante non mesurée")
        p_cur = float(np.mean([self.results[current][t] for t in ids]))
        est = self.estimate(new)
        if est["status"] == "needs_probes":
            return dict(new=new, current=current, action="needs_probes", missing=est["missing"])
        if est["status"] == "stale_plan":
            return dict(new=new, current=current, action="stale_plan", reason=est["reason"])
        if est["status"] == "measured":
            point = est["mean"]
        else:
            pl = self.plans[new]
            idx = np.array([ids.index(t) for t in pl["tasks"]])
            y = np.array([self.results[new][t] for t in pl["tasks"]])
            point = Gr.graph_v3(X, idx, y, y_prev, rng=np.random.default_rng(pl["seed"] + 2)).mean
        action = "promote" if point >= p_cur - margin else "keep"
        thr = p_cur - margin
        # Sure only if the GREG-cf interval excludes the threshold on the side of the action; when the two estimators
        # disagree (graph_v3 point on one side, interval entirely on the other), the decision is reported as not sure.
        sure = (action == "promote" and est["lo"] > thr) or (action == "keep" and est["hi"] < thr)
        not_contrary = not ((action == "promote" and est["hi"] < thr) or (action == "keep" and est["lo"] > thr))
        return dict(new=new, current=current, action=action, confident=bool(sure), interval_not_contrary=bool(not_contrary),
                    p_current=p_cur, p_new_point=float(point), p_new_interval=(est["lo"], est["hi"]))

    # ------------------------------------------------------------------ lint
    def lint(self) -> list[dict]:
        """Check the consistency of the log.

        Returns
        -------
        list of dict
            One finding per problem, with a ``code``:

            - L1: the parent belongs to another slot;
            - L2: the parent was released at the same time as or after the version;
            - L3: probe results outside the version's plan (ignored by :meth:`estimate`);
            - L4: the plan was made against another parent than the current one;
            - L5: a training task is stored as evidence;
            - L6: the plan was made on another task suite than the current one (estimate refuses it);
            - L7: repeated runs of the same version and task disagree (the latest run is used).
        """
        out = []
        for v, x in self.versions.items():
            if x.parent and self.versions[x.parent].slot != x.slot:
                out.append(dict(code="L1", version=v, msg="parent dans une autre slot"))
            if x.parent and self.versions[x.parent].released_at >= x.released_at:
                out.append(dict(code="L2", version=v, msg="parent publié après ou en même temps que la version"))
            pl = self.plans.get(v)
            if pl:
                extra = [t for t in self.results.get(v, {}) if t not in pl["tasks"] and self.roles.get((v, t)) == "probe"]
                if extra:
                    out.append(dict(code="L3", version=v, msg=f"{len(extra)} probes hors plan : l'estimation les ignore"))
                if pl["against"] != x.parent:
                    out.append(dict(code="L4", version=v, msg="plan établi contre un autre parent"))
        ids_hash = suite_hash(self.task_list())
        for v, pl in self.plans.items():
            if pl.get("suite_hash") and pl["suite_hash"] != ids_hash:
                out.append(dict(code="L6", version=v, msg="plan établi sur une autre suite de tâches"))
        for (v, t), rs in self.runs.items():
            if len({r["success"] for r in rs}) > 1:
                out.append(dict(code="L7", version=v, task=t, msg=f"{len(rs)} exécutions en désaccord : la dernière est utilisée"))
        for (v, t), role in self.roles.items():
            if role == "train":
                out.append(dict(code="L5", version=v, task=t, msg="tâche d'entraînement utilisée comme preuve"))
        return out

    # ------------------------------------------------------------------ pages
    def render(self, out_dir: str | Path | None = None) -> list[Path]:
        """Write one Markdown page per slot, listing its versions with their rates and intervals.

        Parameters
        ----------
        out_dir : str or Path, optional
            Output directory; defaults to ``<root>/wiki``. Pages go to ``<out_dir>/slots/<slot>.md``.

        Returns
        -------
        list of Path
            The pages written.
        """
        out_dir = Path(out_dir or self.root / "wiki")
        (out_dir / "slots").mkdir(parents=True, exist_ok=True)
        paths = []
        for slot in sorted(self.slots):
            vs = sorted((x for x in self.versions.values() if x.slot == slot), key=lambda x: x.released_at)
            lines = [f"# {slot}", "", self.slots[slot]["responsibility"], "",
                     "| Version | Publiée | Parent | Taux | Intervalle à 90 % | Tâches mesurées | Méthode |",
                     "|---|---|---|---|---|---|---|"]
            for x in vs:
                e = self.estimate(x.id)
                if e["status"] in ("measured", "estimated"):
                    lines.append(f"| {x.id} | {x.released_at} | {x.parent or '—'} | {e['mean']:.3f} | "
                                 f"[{e['lo']:.3f} ; {e['hi']:.3f}] | {e['n_probes']} | {e['method']} |")
                else:
                    lines.append(f"| {x.id} | {x.released_at} | {x.parent or '—'} | — | — | 0 | à tester |")
            p = out_dir / "slots" / f"{slot}.md"
            p.write_text("\n".join(lines) + "\n")
            paths.append(p)
        return paths
