# How Much Is an Agent's History Worth?

Code, pre-registrations, results and paper for

> Rayen Zargui. *How Much Is an Agent's History Worth? A Law of Evaluation Transfer Across Agent Versions,
> Tested in Pre-Registered Studies.* Preprint, 2026.

When a new version of an AI agent ships, we organise everything already known as a **graph of agents**: nodes are
agent (or model) versions, lineage edges link a version to its predecessor, and every node carries its outcome on
each task of a shared suite. The history is worth `1/(1 − ρ²)` times the tasks, where `ρ²` is the share of the new
version's task-level variance the graph already explains (`φ²` with the predecessor, `η²` with the task difficulty
seen by earlier agents). The method: the graph **chooses** the tasks, a cross-fitted model-assisted estimator
**estimates** the new version's success rate with calibrated intervals, and the same tasks **decide**
keep-or-replace. See `IDEA.md` for a one-page summary and `paper/main.pdf` for the paper.

Study v8 extends the method to **agentic patterns and multi-agent systems** (13 agents submitted to AppWorld after its
paper, including two multi-agent frameworks treated as single nodes) and to **asynchronous evaluation** (results that
arrive out of order and an evaluation stopped at a deadline). The graph here records evaluation evidence; it is not an
orchestration graph of agents calling one another. A prospective pre-registration (`prereg-v9`, `docs/PREREGISTRATION_V9.md`) fixes how the deadline correction will be
tested on AppWorld agents released after 2026-11-04.

## Repository layout

| Path | Content |
|---|---|
| `src/cvt/estimators.py` | random testing (reset), predecessor-only Bayesian double sampling (`double_sampling_bma`), the efficiency law |
| `src/cvt/graph.py` | graph features, model-based graph estimator (`graph_v3`, `graph_bma_logit`), GREG and cross-fitted GREG (`graph_greg_cf`) |
| `src/cvt/kernel.py` | the agent-graph kernel: event log, slots and versions, `record`, `plan`, `estimate`, `decide`, `lint`, pages |
| `tests/` | 30 tests (`python -m pytest -q tests`) |
| `experiments/` | frozen analysis scripts of each study (`*_preregistered.py`), data loaders, development scripts (`dev_*`) |
| `docs/` | protocols (`PREREGISTRATION*.md`), hashes (`PREREG*_SHA256.txt`), results (`RESULTS_*.md`), deviations log, code guide (`CODE_GUIDE.md`) |
| `tools/` | `verify_preregistration.py`: checks every freeze against its hashes |
| `results/` | derived results (replays, summaries) — every number of the paper is generated from these files |
| `paper/` | LaTeX source; `make_numbers.py` writes all numbers and figures, `check_numbers.py` cross-checks them |
| `demo/` | kernel case study on the OpenHands lineage (SWE-bench Verified) |

## Pre-registration chain

Each study was frozen by a git tag and SHA-256 hashes of its protocol and analysis script before any statistic was
computed on its data. Check every freeze at once with:

    git fetch --tags
    python tools/verify_preregistration.py

It recomputes each hash on the files *as committed at the tag*, and checks that the shared library
(`src/cvt/*.py`) was only extended afterwards: no line that existed at a freeze has been changed or removed. Hashes
must be checked at the tag, not on the working tree, because `src/cvt/graph.py` gained new estimators from one study
to the next. To inspect one freeze by hand:

    git show prereg-v6 --stat
    git diff prereg-v6 HEAD -- experiments/v6_unified_preregistered.py experiments/v6_data.py

| Tag | Study | Protocol | Results |
|---|---|---|---|
| `prereg-v1` | pilot, model versions (Sheng & Lu) | `docs/PREREGISTRATION.md` | `docs/RESULTS_G2.md` |
| `prereg-v2` | AppWorld agents | `docs/PREREGISTRATION_V2.md` | `docs/RESULTS_V2.md` |
| `prereg-v3` | SWE-bench Verified | `docs/PREREGISTRATION_V3.md` | `docs/RESULTS_V3.md` |
| `prereg-v4` | SWE-bench Lite | `docs/PREREGISTRATION_V4.md` | `docs/RESULTS_V4.md` |
| `prereg-v5` | Multi-SWE-bench (7 languages), SWE-bench test and multilingual | `docs/PREREGISTRATION_V5.md` | `docs/RESULTS_V5.md` |
| `prereg-v6` | LiveCodeBench, TheAgentCompany (unified method) | `docs/PREREGISTRATION_V6.md` | `docs/RESULTS_V6.md` |

Git timestamps are self-reported: the order of tags and results is recorded, not certified by a third party.
Deviations are logged in `docs/DEVIATIONS.md`. Protocols and results documents are in French; the paper is in English.

## Reproduce

Python 3.11+ with `numpy`, `scipy`, `matplotlib`, `pyyaml`, `pytest`. `docs/CODE_GUIDE.md` explains the code module by module and maps each estimator to the paper. Data are not redistributed here; `DATA.md`
lists every source, its exact commit and where the scripts expect it.

    python -m pytest -q tests
    python experiments/v6_unified_preregistered.py 400          # one study (about 40 min on 2 cores)
    python experiments/kernel_replay_verified.py 100            # kernel validation
    python experiments/v8_patterns_async_preregistered.py 200   # v8 (needs the unpacked AppWorld bundles, see DATA.md)
    cd paper && python make_numbers.py && python check_numbers.py && latexmk -pdf main.tex

## Licence and citation

Code under the MIT licence (`LICENSE`); paper and documents under CC BY 4.0. Cite with `CITATION.cff`.
The benchmark outputs belong to their authors and keep their own licences.
