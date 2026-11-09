"""Data loader for study v8: agentic patterns, multi-agent systems and asynchronous evaluation (AppWorld).

Computes no outcome statistic. It reads, for each AppWorld agent and split, the task-level success and the number of
environment interactions (steps) the agent used on each task, which serves as the completion-time proxy for the
asynchronous-evaluation analysis.

Sources
-------
- The 14 agents of the AppWorld paper (4 patterns x models), already analysed in v2 and used for development:
  ``<aw_root>/experiments/outputs/<agent>_<split>/``.
- 13 agents submitted to the AppWorld leaderboard later (2025-04 to 2026-10), never analysed before v8: the official
  ``leaderboard.bundle`` files of ``stonybrooknlp/appworld-leaderboard``, unpacked with the AppWorld package
  (``appworld.common.utils.unpack_bundle`` with the package's public password) into ``<aw_new>/<agent>_<split>/``.
  Decrypted content is not redistributed.

Pattern labels were fixed from each submission's own description (``metadata.json`` tooltip and linked page) before
any outcome of the new agents was computed; see ``NEW_AGENTS``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

AW_ROOT = Path("/home/claude/data/aw_root/experiments/outputs")
AW_NEW = Path("/home/claude/data/aw_new")
LEADERBOARD = Path("/home/claude/data/appworld-leaderboard/experiments/outputs/_leaderboard.json")
SPLITS = ("test_normal", "test_challenge")

#: The 14 agents of the AppWorld paper, all released on 2024-07-26 (leaderboard date): pattern = method.
OLD_AGENTS = {
    f"{m}_{l}": ("2024-07-26", {"react": "ReAct loop", "plan_exec": "plan-and-execute workflow",
                                "full_code_refl": "code generation with reflection",
                                "ipfuncall": "function-calling loop"}[m])
    for m, ls in {"react": ("gpt4turbo", "gpt4o", "llama3", "deepseekcoder"),
                  "plan_exec": ("gpt4turbo", "gpt4o", "llama3", "deepseekcoder"),
                  "full_code_refl": ("gpt4turbo", "gpt4o", "llama3", "deepseekcoder"),
                  "ipfuncall": ("gpt4turbo", "gpt4o")}.items()
    for l in ls
}

#: The 13 later agents: (release date from the leaderboard file, or metadata when absent; pattern class; lineage
#: parent among earlier agents or None). Pattern classes, fixed from the submissions' descriptions:
#:   "multi-agent"  - several LLM agents with distinct roles coordinated by a controller or a plan
#:                    (LARA MAS: an explorer writes a plan, per-app specialist executors run ReAct loops;
#:                     IBM CUGA: hierarchical planner-executor, a plan controller delegates to specialised sub-agents);
#:   "RL-trained"   - a single agent whose model was trained with reinforcement learning for the environment
#:                    (LOOP, SynthAPI, Alibaba AgentRL, RooAGI MEMTune RL);
#:   "single-agent" - a single prompted agent with a tool harness, demonstrations or learned context
#:                    (AtlasHarness, Button Agent, capybara, ReAct + SetBSR demos, IBM HCL-GP).
NEW_AGENTS = {
    "loop": ("2025-04-09", "RL-trained", None),
    "cuga": ("2025-07-12", "multi-agent", None),
    "react-gpt4o-2_setbsr_upfront_demos": ("2025-07-13", "single-agent", "react_gpt4o"),
    "alibaba_cloud_apsaralab_0118_qwen3_14b_agentrl": ("2026-02-15", "RL-trained", None),
    "rooagi_agentbuilder_memtune_rl_gptoss_20b": ("2026-02-22", "RL-trained", None),
    "button_agent": ("2026-03-23", "single-agent", None),
    "ibm-hcl-gp-claude-sonnet-4-6": ("2026-05-18", "single-agent", None),
    "kecaipan_capybara_gpt56luna_20260808": ("2026-08-08", "single-agent", None),
    "lara": ("2026-08-19", "multi-agent", None),
    "lara_opus": ("2026-09-02", "multi-agent", "lara"),
    "ibm_cuga_glm53flash": ("2026-09-07", "multi-agent", "cuga"),
    "atlasharness": ("2026-10-21", "single-agent", None),
    "synthapi_27b": ("2026-10-28", "RL-trained", None),
}

_STEP = re.compile(r"^### Environment Interaction", re.M)
_REPORT = re.compile(r"Num Passed Tests : (\d+)\s+Num Failed Tests : (\d+)\s+Num Total\s+Tests : (\d+)")


def _dir(agent: str, split: str) -> Path:
    """Output directory of ``agent`` on ``split``."""
    return (AW_ROOT if agent in OLD_AGENTS else AW_NEW) / f"{agent}_{split}"


def _success_from_reports(d: Path, ids: list[str]) -> np.ndarray:
    """Success from per-task evaluation reports (used only when the summary file is missing): a task succeeds when
    it has at least one test and no failed test, which is AppWorld's definition of task success."""
    out = []
    for t in ids:
        m = _REPORT.search((d / "tasks" / t / "evaluation" / "report.md").read_text())
        out.append(m is not None and int(m.group(1)) == int(m.group(3)) and int(m.group(3)) > 0 and int(m.group(2)) == 0)
    return np.array(out)


def load(agent: str, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Task ids (sorted), success (bool) and number of environment interactions per task for one agent."""
    d = _dir(agent, split)
    ev = d / "evaluations" / f"{split}.json"
    if ev.exists():
        ind = json.load(ev.open())["individual"]
        ids = sorted(ind)
        y = np.array([bool(ind[t]["success"]) for t in ids])
    else:
        ids = sorted(p.name for p in (d / "tasks").iterdir())
        y = _success_from_reports(d, ids)
    steps = np.array([len(_STEP.findall((d / "tasks" / t / "logs" / "environment_io.md").read_text()))
                      if (d / "tasks" / t / "logs" / "environment_io.md").exists() else 0 for t in ids])
    return ids, y, steps


def all_agents() -> dict[str, tuple[str, str, str | None, bool]]:
    """Agent -> (release date, pattern, parent, is_new)."""
    out = {a: (d, p, None, False) for a, (d, p) in OLD_AGENTS.items()}
    out.update({a: (d, p, par, True) for a, (d, p, par) in NEW_AGENTS.items()})
    return out


def load_split(split: str):
    """Load every agent on ``split``; check that all agents share the same task ids.

    Returns
    -------
    ids : list of str
    Y : dict agent -> bool array
    S : dict agent -> int array of steps
    """
    ids0, Y, S = None, {}, {}
    for a in all_agents():
        ids, y, s = load(a, split)
        if ids0 is None:
            ids0 = ids
        assert ids == ids0, f"task ids differ for {a} on {split}"
        Y[a], S[a] = y, s
    return ids0, Y, S
