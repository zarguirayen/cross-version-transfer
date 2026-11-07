"""Noyau minimal du graphe d'agents (spec v0.7, sections Modèle de données, record, estimate, decide, lint).

Le noyau observe, il n'orchestre pas. Chaque nœud est une version d'un slot (un agent, un modèle, un outil) ; les
arêtes de lignée relient une version à la précédente ; les preuves sont des résultats de tâches. Tout passe par
un journal d'événements append-only (events.jsonl) ; l'état est reconstruit en le relisant.

Opérations :
- register_slot / register_version : le graphe (lignée, date de sortie).
- record : un résultat (version, tâche, succès), avec rôle et origine.
- plan : le graphe choisit les tâches à tester pour une nouvelle version (plan stratifié sur la difficulté vue
  par les versions antérieures, toutes slots confondues).
- estimate : taux de réussite de la version et intervalle, par GREG croisé sur les probes du plan
  (méthode confirmée en v5 et v6) ; reset de Jeffreys si le graphe est vide.
- decide : garder la version courante ou passer à la nouvelle (graphe stratifié, méthode confirmée en v4–v6).
- lint : contrôles de cohérence du journal.
- render : une page Markdown par slot.
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

ROLES = {"probe", "full", "train", "other"}
ORIGINS = {"production", "benchmark", "synthetic"}


@dataclass
class Version:
    slot: str
    id: str
    parent: str | None
    released_at: str
    meta: dict = field(default_factory=dict)


class Kernel:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / "events.jsonl"
        self.slots: dict[str, dict] = {}
        self.versions: dict[str, Version] = {}
        self.results: dict[str, dict[str, bool]] = {}      # version -> tâche -> succès
        self.roles: dict[tuple[str, str], str] = {}
        self.plans: dict[str, dict] = {}
        self.tasks: set[str] = set()
        self._replay()

    # ------------------------------------------------------------------ journal
    def _append(self, ev: dict):
        ev = {"ts": time.time(), **ev}
        with self.log.open("a") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self._apply(ev)

    def _replay(self):
        if self.log.exists():
            for line in self.log.open():
                if line.strip():
                    self._apply(json.loads(line))

    def _apply(self, ev: dict):
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
            self.results.setdefault(ev["version"], {}).update({t: bool(v) for t, v in ev["outcomes"].items()})
            for t in ev["outcomes"]:
                self.roles[(ev["version"], t)] = ev.get("role", "full")
        elif k == "result":
            self.results.setdefault(ev["version"], {})[ev["task"]] = bool(ev["success"])
            self.roles[(ev["version"], ev["task"])] = ev.get("role", "other")
        elif k == "plan":
            self.plans[ev["version"]] = {"tasks": ev["tasks"], "weights": ev["weights"], "against": ev["against"],
                                         "seed": ev["seed"]}

    # ------------------------------------------------------------------ graphe
    def register_slot(self, slot: str, responsibility: str = ""):
        if slot not in self.slots:
            self._append({"event": "slot", "slot": slot, "responsibility": responsibility})

    def register_version(self, slot: str, version: str, released_at: str, parent: str | None = None, **meta):
        if slot not in self.slots:
            raise KeyError(f"slot inconnu : {slot}")
        if parent is not None and parent not in self.versions:
            raise KeyError(f"version parente inconnue : {parent}")
        if version in self.versions:
            raise ValueError(f"version déjà enregistrée : {version}")
        self._append({"event": "version", "slot": slot, "version": version, "parent": parent,
                      "released_at": released_at, "meta": meta})

    def register_tasks(self, tasks):
        for t in sorted(set(tasks) - self.tasks):
            self._append({"event": "task", "task": t})

    def record(self, version: str, task: str, success: bool, role: str = "full", origin: str = "benchmark",
               verifier: str = "unknown"):
        if version not in self.versions:
            raise KeyError(f"version inconnue : {version}")
        if task not in self.tasks:
            raise KeyError(f"tâche inconnue : {task}")
        if role not in ROLES or origin not in ORIGINS:
            raise ValueError("rôle ou origine invalide")
        self._append({"event": "result", "version": version, "task": task, "success": bool(success), "role": role,
                      "origin": origin, "verifier": verifier})

    def record_many(self, version: str, outcomes: dict, role: str = "full", origin: str = "benchmark",
                    verifier: str = "unknown"):
        """Import en bloc d'une évaluation (par exemple l'historique complet d'une version)."""
        if version not in self.versions:
            raise KeyError(f"version inconnue : {version}")
        unknown = set(outcomes) - self.tasks
        if unknown:
            raise KeyError(f"{len(unknown)} tâches inconnues")
        if role not in ROLES or origin not in ORIGINS:
            raise ValueError("rôle ou origine invalide")
        self._append({"event": "results", "version": version, "outcomes": {t: bool(v) for t, v in outcomes.items()},
                      "role": role, "origin": origin, "verifier": verifier})

    # ------------------------------------------------------------------ vues
    def task_list(self) -> list[str]:
        return sorted(self.tasks)

    def _complete(self, v: str, ids: list[str]) -> bool:
        r = self.results.get(v, {})
        return all(t in r for t in ids)

    def ecosystem(self, version: str) -> list[str]:
        """Versions publiées strictement avant `version` (toutes slots), avec résultats sur toutes les tâches,
        sauf son parent (qui entre comme covariable séparée)."""
        v = self.versions[version]; ids = self.task_list()
        return [w for w, x in self.versions.items() if x.released_at < v.released_at and w != v.parent
                and w != version and self._complete(w, ids)]

    def _features(self, version: str):
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
        """Le graphe choisit les m tâches à tester : tri par difficulté vue par l'écosystème, m strates contiguës,
        1 tâche tirée par strate. Le plan est enregistré (il fixe les poids de l'estimation)."""
        ids, X, _, eco = self._features(version)
        if seed is None:
            seed = int(hashlib.sha256(version.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        score = X[:, 0] if eco else rng.random(len(ids))
        blocks = np.array_split(np.argsort(score, kind="stable"), m)
        pick = [int(rng.choice(b)) for b in blocks]
        tasks = [ids[i] for i in pick]
        self._append({"event": "plan", "version": version, "tasks": tasks, "weights": [len(b) for b in blocks],
                      "against": self.versions[version].parent, "seed": seed, "n_ecosystem": len(eco)})
        return tasks

    # ------------------------------------------------------------------ estimate
    def estimate(self, version: str, level: float = 0.90) -> dict:
        ids, X, y_prev, eco = self._features(version)
        pl = self.plans.get(version)
        r = self.results.get(version, {})
        if self._complete(version, ids):                       # tout est mesuré : pas d'estimation à faire
            p = float(np.mean([r[t] for t in ids]))
            return dict(version=version, status="measured", mean=p, lo=p, hi=p, n_probes=len(ids), method="census")
        if pl is None or not all(t in r for t in pl["tasks"]):
            missing = [] if pl is None else [t for t in pl["tasks"] if t not in r]
            return dict(version=version, status="needs_probes", missing=missing)
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
        """Garder `current` (par défaut : le parent) ou passer à `new`. Point de décision : graphe stratifié sur
        les probes du plan ; l'intervalle GREG croisé dit si la décision est sûre."""
        current = current or self.versions[new].parent
        ids, X, y_prev, eco = self._features(new)
        if current is None or not self._complete(current, ids):
            return dict(new=new, current=current, action="undetermined", reason="version courante non mesurée")
        p_cur = float(np.mean([self.results[current][t] for t in ids]))
        est = self.estimate(new)
        if est["status"] == "needs_probes":
            return dict(new=new, current=current, action="needs_probes", missing=est["missing"])
        if est["status"] == "measured":
            point = est["mean"]
        else:
            pl = self.plans[new]
            idx = np.array([ids.index(t) for t in pl["tasks"]])
            y = np.array([self.results[new][t] for t in pl["tasks"]])
            point = Gr.graph_v3(X, idx, y, y_prev, rng=np.random.default_rng(pl["seed"] + 2)).mean
        action = "promote" if point >= p_cur - margin else "keep"
        sure = est["lo"] > p_cur - margin or est["hi"] < p_cur - margin
        return dict(new=new, current=current, action=action, confident=bool(sure), p_current=p_cur,
                    p_new_point=float(point), p_new_interval=(est["lo"], est["hi"]))

    # ------------------------------------------------------------------ lint
    def lint(self) -> list[dict]:
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
        for (v, t), role in self.roles.items():
            if role == "train":
                out.append(dict(code="L5", version=v, task=t, msg="tâche d'entraînement utilisée comme preuve"))
        return out

    # ------------------------------------------------------------------ pages
    def render(self, out_dir: str | Path | None = None) -> list[Path]:
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
