"""Données de la confirmation v3 : soumissions publiques à SWE-bench Verified (dépôt SWE-bench/experiments).
Ce module ne calcule AUCUNE statistique de résultat : chargement, univers des 500 tâches, dates, lignées.

Règles fixées avant toute statistique :
- Une soumission = un agent (nœud du graphe), datée par le préfixe AAAAMMJJ de son dossier.
- Résultat par tâche : 1 si l'identifiant figure dans `resolved` (results/results.json) ou si
  per_instance_details.json indique resolved = true ; 0 sinon (convention du classement officiel).
- Lignée = même produit (table PRODUCTS, écrite à partir des seuls noms de dossiers). Pour mini-SWE-agent,
  qui sert de banc commun, la lignée = même fournisseur de modèle (champ tags.model_org ou, à défaut, org).
- Transition v -> v' : pour chaque soumission v' d'une lignée, v = la soumission la plus récente de la même
  lignée à une date STRICTEMENT antérieure ; à égalité de date, celle dont le nom suit l'ordre alphabétique en
  dernier. Les variantes du même jour (TTS, 7B/72B) ne forment donc pas de transition entre elles.
- Écosystème de v' (analyse principale) : toutes les soumissions de date strictement antérieure à v', sauf v.
  Analyse secondaire : en excluant aussi la lignée de v'.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import yaml

PRODUCTS = [  # (motif sur le nom du dossier sans la date, lignée)
    (r"^rag_", "rag"), (r"^sweagent_", "swe-agent"), (r"^amazon-q-developer", "amazon-q"),
    (r"^autocoderover", "autocoderover"), (r"^agentless-1\.5", "agentless"), (r"^agentless_lite", "agentless"),
    (r"^(openhands|OpenHands)", "openhands"), (r"^epam-ai-run", "epam"), (r"^nfactorial", "nfactorial"),
    (r"^solver$", "solver"), (r"^lingma-agent_lingma-swe-gpt-72b", "lingma-72b"),
    (r"^lingma-agent_lingma-swe-gpt-7b", "lingma-7b"), (r"^composio", "composio"), (r"^gru$", "gru"),
    (r"^devlo$", "devlo"), (r"^emergent$", "emergent"), (r"^tools_", "anthropic-tools"),
    (r"^augment_agent", "augment"), (r"^cortexa", "cortexa"), (r"^trae", "trae"), (r"^Refact_Agent", "refact"),
    (r"^warp$", "warp"), (r"^patchpilot", "patchpilot"), (r"^sonar-foundation-agent", "sonar"),
    (r"^livesweagent", "live-swe-agent"), (r"^Prometheus", "prometheus"), (r"^SalesforceAIResearch_SAGE", "sage"),
    (r"^Lingxi", "lingxi"), (r"^artemis_agent", "artemis"), (r"^SWE-Fixer", "swe-fixer"),
    (r"^zai_glm", "zai-glm"), (r"^appmap-navie|^navie-2", "appmap-navie"), (r"^mini-", "MINI"),
]


def lineage(dirname: str, meta: dict) -> str | None:
    rest = dirname.split("_", 1)[1]
    for pat, lab in PRODUCTS:
        if re.search(pat, rest):
            if lab != "MINI":
                return lab
            t = meta.get("tags", {}) or {}
            org = t.get("model_org") or t.get("org")
            org = org[0] if isinstance(org, list) else org
            return f"mini:{str(org).lower().replace('-', '').replace('.', '').replace(' ', '')}"
    return None


def load(root: str | Path):
    """Renvoie (ids des 500 tâches triés, dict nom -> vecteur booléen, dict nom -> date, dict nom -> lignée)."""
    base = Path(root) / "evaluation" / "verified"
    resolved, universe, metas = {}, set(), {}
    for d in sorted(os.listdir(base)):
        p = base / d
        meta = yaml.safe_load((p / "metadata.yaml").open()) if (p / "metadata.yaml").exists() else {}
        if (p / "per_instance_details.json").exists():
            det = json.load((p / "per_instance_details.json").open())
            universe |= set(det)
            resolved[d] = {k for k, v in det.items() if v.get("resolved")}
        elif (p / "results" / "results.json").exists():
            resolved[d] = set(json.load((p / "results" / "results.json").open())["resolved"])
        else:
            continue
        metas[d] = meta
    ids = sorted(universe)
    Y = {d: np.array([i in r for i in ids]) for d, r in resolved.items()}
    extra = {d: len(r - universe) for d, r in resolved.items()}
    dates = {d: d[:8] for d in Y}
    lin = {d: lineage(d, metas[d]) for d in Y}
    return ids, Y, dates, lin, extra


def transitions(dates: dict, lin: dict):
    out = []
    for d in sorted(dates):
        L = lin[d]
        if L is None:
            continue
        prev = [e for e in dates if lin[e] == L and dates[e] < dates[d]]
        if prev:
            v = max(prev, key=lambda e: (dates[e], e))
            out.append((L, v, d))
    return out


def ecosystem(dates: dict, lin: dict, old: str, new: str, exclude_lineage: bool = False):
    return [e for e in dates if dates[e] < dates[new] and e != old and not (exclude_lineage and lin[e] == lin[new])]
