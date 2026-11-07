"""Données de la confirmation v5 (généralité) : 11 jeux de tâches jamais analysés.
- Multi-SWE-bench (dépôt multi-swe-bench/experiments, evaluation/<langage>/verified) : C, C++, Go, Java,
  JavaScript, Rust, TypeScript. Python est EXCLU (risque de recouvrement avec SWE-bench Verified, déjà vu).
- SWE-bench (dépôt SWE-bench/experiments) : découpages test et multilingual (multimodal exclu : il ne publie
  que des totaux).
Ce module ne calcule AUCUNE statistique de résultat.

Règles fixées avant toute statistique :
- Nœud = soumission d'un jeu de tâches, datée par son préfixe AAAAMMJJ. Chaque jeu de tâches est un graphe séparé.
- Univers d'un jeu = réunion des identifiants cités par ses soumissions (submitted_ids, completed_ids,
  resolved, unresolved_ids… ; per_instance_details.json pour mini-SWE-agent). Résultat = 1 si résolu, 0 sinon.
- Lignée :
  * échafaudages fournis par les auteurs du banc (Multi-SWE-bench : MSWE-agent, MagentLess, MopenHands ;
    mini-SWE-agent) : même échafaudage ET même fournisseur de modèle (table PROVIDERS) ;
  * autres soumissions : même produit (motifs PRODUCTS sur le nom sans la date, insensibles à la casse).
- Découpage test : seules les tâches absentes de Verified et de Lite (déjà analysés) sont gardées.
- Transition v -> v' : v = soumission la plus récente de la même lignée à date strictement antérieure ; égalité :
  dernier nom dans l'ordre alphabétique. Écosystème de v' : soumissions du même jeu, antérieures à v', sauf v.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

MSWE_ROOT = Path("/home/claude/data/mswe/experiments/evaluation")
SWE_ROOT = Path("/home/claude/data/swe/experiments/evaluation")
MSWE_LANGS = ["c", "c++", "go", "java", "javascript", "rust", "typescript"]
SWE_SPLITS = ["test", "multilingual"]   # multimodal exclu : totaux seulement, pas de résultat par tâche

PROVIDERS = [(r"doubao", "bytedance"), (r"claude", "anthropic"), (r"(gpt|openai|o1|o3|o4)", "openai"),
             (r"deepseek", "deepseek"), (r"qwen", "alibaba"), (r"gemini", "google"), (r"llama", "meta"),
             (r"(kimi)", "moonshot"), (r"glm", "zhipu"), (r"minimax", "minimax")]
SCAFFOLDS = [(r"^mswe-agent_", "MSWE-agent"), (r"^magentless_", "MagentLess"), (r"^mopenhands_", "MopenHands"),
             (r"^mini-", "mini")]
PRODUCTS = [(r"^rag_", "rag"), (r"^sweagent[_-]", "swe-agent"), (r"^swe-agent", "swe-agent"),
            (r"^amazon-q-developer", "amazon-q"), (r"^autocoderover", "autocoderover"),
            (r"^agentless", "agentless"), (r"^(opendevin|openhands)", "openhands"), (r"^zencoder", "zencoder"),
            (r"^guirepair", "guirepair"), (r"^reporepair", "reporepair"), (r"^infcode", "infcode"),
            (r"^iswe", "iswe"), (r"^codearts-agent", "codearts"), (r"^codefuse", "codefuse")]


def provider(name):
    n = name.lower()
    for pat, lab in PROVIDERS:
        if re.search(pat, n):
            return lab
    return None


def lineage(dirname):
    rest = dirname.split("_", 1)[1].lower()
    for pat, lab in SCAFFOLDS:
        if re.search(pat, rest):
            p = provider(rest)
            return f"{lab}:{p}" if p else None
    for pat, lab in PRODUCTS:
        if re.search(pat, rest):
            return lab
    return None


def _read(d: Path):
    det = d / "per_instance_details.json"
    if det.exists():
        r = json.load(det.open())
        return set(r), {k for k, v in r.items() if v.get("resolved")}
    f = d / "results" / "results.json"
    if not f.exists():
        return None
    r = json.load(f.open())
    universe = set()
    for v in r.values():
        if isinstance(v, list):
            universe |= {x for x in v if isinstance(x, str)}
    res = r.get("resolved", r.get("resolved_ids", []))
    if not isinstance(res, list):
        return None
    return universe, set(res)


def load_taskset(name: str):
    base = MSWE_ROOT / name / "verified" if name in MSWE_LANGS else SWE_ROOT / name
    res, universe = {}, set()
    for d in sorted(os.listdir(base)):
        out = _read(base / d)
        if out is None:
            continue
        universe |= out[0]
        res[d] = out[1]
    if name == "test":      # tâches déjà vues (Verified, Lite) exclues avant toute statistique
        seen = set()
        for sub in ("verified", "lite"):
            for dd in os.listdir(SWE_ROOT / sub):
                o = _read(SWE_ROOT / sub / dd)
                if o:
                    seen |= o[0]
        universe -= seen
    ids = sorted(universe)
    Y = {d: np.array([i in r for i in ids]) for d, r in res.items()}
    dates = {d: d[:8] for d in Y}
    lin = {d: lineage(d) for d in Y}
    return ids, Y, dates, lin


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


TASKSETS = [f"mswe-{l}" for l in MSWE_LANGS] + [f"swe-{s}" for s in SWE_SPLITS]


def load(ts: str):
    return load_taskset(ts.split("-", 1)[1])
