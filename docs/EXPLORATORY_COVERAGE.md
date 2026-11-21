# Exploratory analysis: coverage of the outcome matrices (protocol)

Status: **exploratory, retrospective**. This protocol is committed before `experiments/exploratory_coverage.py`
is run. Nothing in the frozen studies is changed; their verdicts stand as reported.

## Question

The frozen data loaders of studies v3 (SWE-bench Verified), v4 (SWE-bench Lite) and v5 (seven Multi-SWE-bench
languages, SWE-bench test and multilingual) code an outcome as 1 if the task is listed as resolved and 0
otherwise, as the official leaderboards do. Some of the 0s are not evaluated attempts: the agent produced no
patch, the evaluation left no log, or the task was never run. How many such cells are there, and does
treating them as missing change the quantities that the predecessor law uses?

## Cell statuses

Read from each submission's own result files, in this order of precedence:

| Status | SWE-bench (`results.json` / `per_instance_details.json`) | Multi-SWE-bench (`results.json`) |
| --- | --- | --- |
| success | listed in `resolved` (or `resolved: true`) | listed in `resolved` |
| not run | absent from `per_instance_details.json`; or, when `generated` exists, absent from `generated` and `no_generation` | absent from `submitted_ids` |
| no patch | listed in `no_generation` | listed in `empty_patch_ids` or `empty_error_patch_ids` |
| no log | listed in `no_logs`; or, when `with_logs` exists, generated but absent from `with_logs` | listed in `error_ids` or `incomplete_ids` |
| attempted failure | every other cell | every other cell |

Task universes, submissions, lineages and transitions are those of the frozen loaders, unchanged.

## Sensitivity analysis

For every transition v -> v' of v3, v4 and v5 (as built by the frozen loaders), compare three codings:

- **official** (the frozen one): every non-success cell is a failure;
- **S1**: cells with status *no log* or *not run* in either version are removed from that transition;
- **S2**: cells with status *no patch*, *no log* or *not run* in either version are removed.

For each coding, compute on the tasks kept: the old and new success rates, phi² between the two versions, and
the predecessor law at m = 20, `gain_law(phi2, 20, N_kept)`. Report the number of transitions with at least one
removed cell, and the median and maximum absolute change from the official coding in p_new and phi², and in
the law's gain (relative change). The replays of the frozen studies are **not** re-run: this analysis bounds
how much the inputs of the law move, not the estimators' errors.

## Outputs

`results/exploratory_coverage/{status_counts.csv, transitions.csv, summary.txt}`.
