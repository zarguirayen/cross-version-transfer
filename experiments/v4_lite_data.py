"""Données de la confirmation v4 : soumissions publiques à SWE-bench Lite (dépôt SWE-bench/experiments,
evaluation/lite). Ce module ne calcule AUCUNE statistique de résultat.

Règles fixées avant toute statistique (mêmes principes qu'en v3) :
- Nœud = une soumission, datée par le préfixe AAAAMMJJ.
- Univers = réunion de tous les identifiants cités dans les listes des results.json (300 tâches attendues).
- Résultat = 1 si l'identifiant figure dans `resolved`, 0 sinon.
- Lignée = même produit (PRODUCTS ci-dessous, à partir des seuls noms de dossiers ; les motifs de la v3 sont
  repris, plus les produits propres à Lite). Transition v -> v' : v = soumission la plus récente de la même lignée
  à une date strictement antérieure ; égalité de date : dernier nom dans l'ordre alphabétique.
- Écosystème de v' : soumissions Lite de date strictement antérieure à v', sauf v.
- Tâches communes avec Verified (93) : incluses dans l'analyse principale ; analyse secondaire sur les 207 autres.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

PRODUCTS = [
    (r"^rag_", "rag"), (r"^sweagent_", "swe-agent"), (r"^amazon-q-developer", "amazon-q"),
    (r"^autocoderover", "autocoderover"), (r"^agentless", "agentless"),
    (r"^(opendevin|OpenHands)", "openhands"), (r"^moatless", "moatless"), (r"^marscode", "marscode"),
    (r"^Isoform", "isoform"), (r"^(opencsg|OpenCSG)", "opencsg"), (r"^codefuse", "codefuse"),
    (r"^KGCompass", "kgcompass"), (r"^ExpeRepair", "experepair"), (r"^entroPO", "entropo"), (r"^gru$", "gru"),
    (r"^IBM", "ibm"), (r"^(appmap-navie|navie-2)", "appmap-navie"), (r"^SWE-Fixer", "swe-fixer"),
]


def lineage(dirname: str):
    rest = dirname.split("_", 1)[1]
    for pat, lab in PRODUCTS:
        if re.search(pat, rest):
            return lab
    return None


def load(root):
    base = Path(root) / "evaluation" / "lite"
    res, universe = {}, set()
    for d in sorted(os.listdir(base)):
        f = base / d / "results" / "results.json"
        if not f.exists():
            continue
        r = json.load(f.open())
        for v in r.values():
            universe |= set(v)
        res[d] = set(r.get("resolved", []))
    ids = sorted(universe)
    Y = {d: np.array([i in r for i in ids]) for d, r in res.items()}
    dates = {d: d[:8] for d in Y}
    lin = {d: lineage(d) for d in Y}
    return ids, Y, dates, lin


def verified_ids(root):
    base = Path(root) / "evaluation" / "verified"
    for d in sorted(os.listdir(base)):
        p = base / d / "per_instance_details.json"
        if p.exists():
            return set(json.load(p.open()))
    return set()


def transitions(dates, lin):
    out = []
    for d in sorted(dates):
        L = lin[d]
        if L is None:
            continue
        prev = [e for e in dates if lin[e] == L and dates[e] < dates[d]]
        if prev:
            out.append((L, max(prev, key=lambda e: (dates[e], e)), d))
    return out


def ecosystem(dates, old, new):
    return [e for e in dates if dates[e] < dates[new] and e != old]
