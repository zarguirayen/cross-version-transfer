"""Données de la confirmation v6 (méthode unifiée) : LiveCodeBench (modèles, 713 problèmes communs) et
TheAgentCompany (agents de bureau, 175 tâches). Ce module ne calcule AUCUNE statistique de résultat.

Règles fixées avant toute statistique :
LiveCodeBench (dépôt LiveCodeBench/submissions, commit 6ca212e)
- Résultat d'un problème = premier échantillon correct (graded_list[0]). Univers = 713 problèmes présents pour
  tous les modèles.
- Nœuds exclus (doublons ou réglages d'un même modèle) : « DeepSeek-V3 copy », « O1-2024-12-17 (Low)_prompt_old »,
  « Gemini-Flash-2.0-Exp (N=10) », « Gemini-Flash-2.0-Thinking » (sans date). Les variantes d'effort (Low, Med,
  High) restent des nœuds de l'écosystème, à la date du modèle.
- Date d'un nœud = date de sortie publique du modèle (table RELEASE, écrite de mémoire par l'assistant, à vérifier ;
  None = date incertaine -> nœud exclu de tous les écosystèmes et de toutes les transitions).
- Lignées = gammes de produit (table LINES), transitions entre versions successives d'une gamme.
- Écosystème de v' = nœuds de date strictement antérieure à v', sauf v.
TheAgentCompany (dépôt TheAgentCompany/experiments, evaluation/1.0.0)
- Résultat d'une tâche = tâche entièrement réussie (final_score.result == final_score.total).
- Noms de tâches normalisés (suffixe « -image » retiré : deux conventions coexistent).
- Soumissions avec moins de 170 tâches exclues ; tâche manquante = échec. Lignée = échafaudage OpenHands + fournisseur du modèle.
"""
import json
import os
from pathlib import Path

import numpy as np

LCB = Path("/home/claude/data/LiveCodeBench_submissions")
TAC = Path("/home/claude/data/TheAgentCompany_experiments/evaluation/1.0.0")

EXCLUDE = {"DeepSeek-V3 copy", "O1-2024-12-17 (Low)_prompt_old", "Gemini-Flash-2.0-Exp (N=10)",
           "Gemini-Flash-2.0-Thinking"}
RELEASE = {
    "GPT-4-0613": "20230613", "GPT-4-Turbo-1106": "20231106", "GPT-4-Turbo-2024-04-09": "20240409",
    "GPT-4O-2024-05-13": "20240513", "GPT-4O-mini-2024-07-18": "20240718", "GPT-4O-2024-08-06": "20240806",
    "O1-Preview-2024-09-12": "20240912", "O1-Mini-2024-09-12": "20240912",
    "O1-2024-12-17 (Low)": "20241217", "O1-2024-12-17 (Med)": "20241217", "O1-2024-12-17 (High)": "20241217",
    "O3-Mini-2025-01-31 (Low)": "20250131", "O3-Mini-2025-01-31 (Med)": "20250131",
    "O3-Mini-2025-01-31 (High)": "20250131", "O3 (High)": "20250416", "O4-Mini (Low)": "20250416",
    "O4-Mini (Medium)": "20250416", "O4-Mini (High)": "20250416",
    "Claude-3-Haiku": "20240313", "Claude-3.5-Sonnet-20240620": "20240620", "Claude-3.5-Sonnet-20241022": "20241022",
    "Claude-3.7-Sonnet": "20250224", "Claude-Sonnet-4": "20250522", "Claude-Sonnet-4 (Thinking)": "20250522",
    "Claude-Opus-4": "20250522", "Claude-Opus-4 (Thinking)": "20250522",
    "Gemini-Pro-1.5-002": "20240924", "Gemini-Flash-1.5-002": "20240924", "Gemini-Exp-1206": "20241206",
    "Gemini-Flash-2.0-Exp": "20241211", "Gemini-Flash-2.0-Thinking-12-19": "20241219",
    "Gemini-Flash-2.0-Thinking-01-21": "20250121", "Gemini-2.5-Pro-03-25": "20250325",
    "Gemini-2.5-Flash-04-17": "20250417", "Gemini-2.5-Pro-05-06": "20250506", "Gemini-2.5-Flash-05-20": "20250520",
    "Gemini-2.5-Pro-06-05": "20250605",
    "DSCoder-1.3b-Ins": "20231102", "DSCoder-6.7b-Ins": "20231102", "DSCoder-33b-Ins": "20231102",
    "DeepSeek-R1-Lite-Preview": "20241120", "DeepSeek-V3": "20241226", "DeepSeek-R1-Preview": "20250120",
    "DeepSeek-R1-0528": "20250528",
    "Qwen2-Ins-72B": "20240607", "Qwen2.5-Ins-7B": "20240919", "Qwen2.5-Ins-32B": "20240919",
    "Qwen2.5-Ins-72B": "20240919", "Qwen2.5-Coder-Ins-7B": "20241112", "Qwen2.5-Coder-Ins-32B": "20241112",
    "QwQ-32B-Preview": "20241128", "QwQ-Max-Preview": "20250225", "QwQ-32B_temp": "20250306",
    "Qwen3-235B-A22B": "20250429",
    "LLama3.3-70b-Ins": "20241206", "Llama-3_1-Nemotron-Nano-8B-v1": "20250318",
    "Llama-3_1-Nemotron-Ultra-253B-v1": "20250407", "OpenReasoning-Nemotron-32B": "20250718",
    "DeepCoder-14B-Preview": "20250408", "Grok-3-Mini (High)": "20250217", "EXAONE-4.0-32B": "20250715",
    "Mistral-Large": None, "Codestral-Latest": None, "AzeroGPT-64b": None, "Kimi-k1.6-IOI": None,
    "Kimi-k1.6-IOI-high": None, "MetaStone-L1-7B": None, "XBai-o4-medium": None,
}
LINES = {
    "openai-gpt4": ["GPT-4-0613", "GPT-4-Turbo-1106", "GPT-4-Turbo-2024-04-09", "GPT-4O-2024-05-13", "GPT-4O-2024-08-06"],
    "openai-o": ["O1-Preview-2024-09-12", "O1-2024-12-17 (High)", "O3 (High)"],
    "openai-o-mini": ["O1-Mini-2024-09-12", "O3-Mini-2025-01-31 (High)", "O4-Mini (High)"],
    "anthropic-sonnet": ["Claude-3.5-Sonnet-20240620", "Claude-3.5-Sonnet-20241022", "Claude-3.7-Sonnet", "Claude-Sonnet-4"],
    "google-pro": ["Gemini-Pro-1.5-002", "Gemini-Exp-1206", "Gemini-2.5-Pro-03-25", "Gemini-2.5-Pro-05-06", "Gemini-2.5-Pro-06-05"],
    "google-flash": ["Gemini-Flash-1.5-002", "Gemini-Flash-2.0-Exp", "Gemini-2.5-Flash-04-17", "Gemini-2.5-Flash-05-20"],
    "google-flash-thinking": ["Gemini-Flash-2.0-Thinking-12-19", "Gemini-Flash-2.0-Thinking-01-21"],
    "deepseek-chat": ["DSCoder-33b-Ins", "DeepSeek-V3"],
    "deepseek-r1": ["DeepSeek-R1-Lite-Preview", "DeepSeek-R1-Preview", "DeepSeek-R1-0528"],
    "qwen-instruct": ["Qwen2-Ins-72B", "Qwen2.5-Ins-72B", "Qwen3-235B-A22B"],
    "qwen-qwq": ["QwQ-32B-Preview", "QwQ-Max-Preview", "QwQ-32B_temp"],
    "nvidia-nemotron": ["Llama-3_1-Nemotron-Ultra-253B-v1", "OpenReasoning-Nemotron-32B"],
}


def load_lcb():
    per = {}
    for d in sorted(os.listdir(LCB)):
        if not (LCB / d).is_dir() or d.startswith(".") or d in EXCLUDE:
            continue
        fs = [f for f in os.listdir(LCB / d) if f.endswith(".json")]
        if not fs:
            continue
        per[d] = {x["question_id"]: bool(x["graded_list"][0]) for x in json.load((LCB / d / fs[0]).open())}
    ids = sorted(set.intersection(*[set(v) for v in per.values()]))
    Y = {d: np.array([v[i] for i in ids]) for d, v in per.items()}
    dates = {d: RELEASE.get(d) for d in Y}
    trans = []
    for L, seq in LINES.items():
        for a, b in zip(seq, seq[1:]):
            if dates.get(a) and dates.get(b) and dates[a] < dates[b]:
                trans.append((f"lcb:{L}", a, b))
    return ids, Y, dates, trans


TAC_PROVIDER = {"gemini": "google", "sonnet": "anthropic", "claude": "anthropic", "gpt": "openai",
                "llama": "meta", "qwen": "alibaba", "nova": "amazon", "deepseek": "deepseek"}


def load_tac():
    Y, dates, lin = {}, {}, {}
    tasks = None
    for d in sorted(os.listdir(TAC)):
        rd = TAC / d / "results"
        if not rd.is_dir():
            continue
        res = {}
        for f in os.listdir(rd):
            if f.startswith("eval_") and f.endswith(".json"):
                s = json.load((rd / f).open())["final_score"]
                name = f[5:-5]
                name = name[:-6] if name.endswith("-image") else name   # deux conventions de nommage
                res[name] = s["result"] == s["total"]
        if len(res) < 170:
            continue
        Y[d] = res; dates[d] = d[:8]
        rest = d.split("_", 1)[1].lower()
        if rest.startswith("openhands-0."):
            prov = next((p for k, p in TAC_PROVIDER.items() if k in rest.split("-", 2)[2]), None)
            lin[d] = f"openhands:{prov}" if prov else None
        else:
            lin[d] = None
    ids = sorted(set.union(*[set(v) for v in Y.values()]))
    Yb = {d: np.array([v.get(i, False) for i in ids]) for d, v in Y.items()}
    trans = []
    for d in sorted(Yb):
        if lin[d] is None:
            continue
        prev = [e for e in Yb if lin[e] == lin[d] and dates[e] < dates[d]]
        if prev:
            trans.append((f"tac:{lin[d]}", max(prev, key=lambda e: (dates[e], e)), d))
    return ids, Yb, dates, trans


def ecosystem(dates, old, new):
    return [e for e in dates if dates[e] is not None and dates[e] < dates[new] and e != old]
