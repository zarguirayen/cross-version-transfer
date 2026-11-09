# Code guide

This guide explains the code module by module and maps each function to the paper. It exists because the core of
the code is **frozen**: the library files `src/cvt/estimators.py`, `src/cvt/graph.py`, `src/cvt/simulate.py` and
every `experiments/*_preregistered.py` and `experiments/*_data.py` are covered by the SHA-256 hashes of the
pre-registrations (`docs/PREREG*_SHA256.txt`). Editing them, even to add a comment, would change their hashes, so
their in-code documentation (written in French during the studies) is left exactly as it was frozen, and this guide
documents them instead. Every other file carries NumPy-style docstrings in English.

`python tools/verify_preregistration.py` checks every freeze against its hashes and confirms that the frozen library
was only extended afterwards (no line that existed at a freeze has been changed or removed).

## Notation

| Symbol | Meaning | In the code |
|---|---|---|
| N | number of tasks in the suite | `N`, `n`, `len(y)` |
| m | number of probes (tasks run on the new version) | `m`, `M` |
| v, v′ | current (old) version and new version | `y1`/`y_prev` and `y2`/`y_new` |
| n11, n10, n01, n00 | flips on the probes: (old, new) = (1,1), (1,0), (0,1), (0,0) | `flips(a, b)` |
| φ² | squared correlation between old and new outcomes | `phi2`, `phi_signed` |
| η²_G | share of the new version's variance between difficulty strata | `r2_strata` |
| graph / ecosystem | earlier agents measured on the same tasks | `aux` (agents × tasks) |

Every estimator returns an `Estimate(mean, lo, hi, method, extra)` with a 90% interval by default.

## Frozen library

### `src/cvt/estimators.py` — predecessor-based estimation (paper Sections 3–4, pilot and v2)

| Function | What it does | Paper |
|---|---|---|
| `gain_law(phi2, m, n_v)` | Finite-population double-sampling law, 1 / (1 − φ² + φ² m / n_v) | Prop. 1, eq. (2) |
| `phi_signed`, `phi2_posterior` | Signed φ from the flip table; posterior draws of φ² (Dirichlet on the flips) | Sec. 3 |
| `borrowed_information` | Number of tasks borrowed from the history (lower quantile for decisions, median for the atlas) | Table 2 |
| `reset` | **Random testing**: the probes alone, Jeffreys or hybrid Clopper–Pearson prior. The baseline of every gain | Sec. 5 |
| `full_inheritance`, `power_prior` | Baselines that reuse the old version's history with weight 1 or a fixed weight | — |
| `double_sampling_bayes` | First version of the method (v0.3.1), with a fallback guard; superseded | App. B |
| `double_sampling_bma` | **Predecessor estimator**: Bayesian model averaging between "no link" (p′ estimated from the probes) and "link" (double sampling p′ = π q1 + (1 − π) q0), weighted by marginal likelihood | Sec. 4 |
| `double_sampling_freq` | Frequentist two-phase baseline (Cochran), normal interval | — |

### `src/cvt/graph.py` — the graph of agents (paper Section 4, studies v2–v7)

The file grew study by study; each block below was added before the freeze named in brackets and never changed
afterwards.

| Function | What it does | Added |
|---|---|---|
| `score_strata`, `graph_bma` | Strata on the ecosystem score, model averaging over stratifications | v2 |
| `ecosystem_features(aux, y_prev)` | Task-level covariates, standardised: column 0 = ecosystem difficulty (logit of the smoothed mean success of earlier agents), column 1 = predecessor's outcome | v2 |
| `graph_logit` | Model-based graph estimator: Bayesian logistic regression on the covariates, importance sampling around the mode | v2 |
| `r2_strata`, `gain_law_r2` | η² between strata and the corresponding law 1/(1 − η²) | v2 |
| `_logit_posterior`, `graph_bma_logit` | Finite-population prediction (probes enter with their observed outcome, the other N − m tasks by Bernoulli draws) and averaging over covariate subsets | v3 |
| `_logit_map`, `ecosystem_slope_prior` | Empirical slope prior learned from the graph (variant tried in development, not used in the confirmations) | v3 |
| `graph_v3` | `graph_bma_logit` with an identifiability guard: if all probes agree, fall back to `double_sampling_bma` | v3 |
| `graph_greg` | **GREG** (Särndal et al. 1992): graph predictions corrected by the probes' residuals, design-unbiased whatever the model; stratum weights; paired-strata variance | v4 |
| `_bma_predict`, `graph_greg_cf` | **Cross-fitted GREG**, the estimator of the unified method (see below) | v5 |
| `neighbour_feature`, `graph_greg_cf_knn`, `graph_v3_knn` | Learned edges: the k earlier agents that agree most with the new version on the probes become an extra covariate (chosen within each fold for the cross-fitted version) | v7 |

### `src/cvt/asynchronous.py` — evaluation stopped at a deadline (study v8)

| Function | What it does |
|---|---|
| `fit_slow_model(Y, S)` | On the earlier agents: logit P = ability + task easiness + γ · slow, where slow = more steps than on half of the agent's tasks |
| `deadline_estimate(...)` | Ability of the new version from its returned probes; unreturned probes imputed with the slowness effect; stratified estimate with an interval that adds imputation, ability and γ uncertainty |

This module is frozen by `prereg-v8`. In v8 its correction did not transfer to newer agents (hypotheses A2, A3 not
supported); see `docs/RESULTS_V8.md`.

**`graph_greg_cf` step by step** (paper Section 4, "Estimate"):

1. The probes come from the stratified design, one per stratum, in stratum order. Fold A = even strata, fold B =
   odd strata, so each fold spans the whole difficulty scale.
2. A model fitted on fold A predicts every task (`PA`); a model fitted on fold B predicts every task (`PB`).
3. Each probe's residual is computed with the model of the *other* fold, so residuals are out of sample.
4. Estimate = mean of (PA + PB)/2 over the N tasks + weighted sum of the cross-fitted residuals / N.
5. Variance from adjacent pairs of strata (Cochran §5A.12), floored at a quarter of the Jeffreys binomial variance.
6. Interval: Jeffreys Beta quantiles at the effective sample size p(1 − p)/variance, with a degrees-of-freedom
   correction (Korn & Graubard 1998); lower bound 0 when no probe succeeds and upper bound 1 when all do (Brown,
   Cai & DasGupta 2001).

### `src/cvt/simulate.py` — synthetic worlds (validation G1, before the pilot)

| Function | What it does |
|---|---|
| `Scenario` | A synthetic transition: rate of v, P(v′ succeeds given v succeeded), P(v′ succeeds given v failed) |
| `draw_world` | Draws the outcomes of v and v′ on n_v tasks |
| `pick_probes`, `resolve_allocation` | Probe allocation: proportional, with a per-stratum floor, or adaptive |
| `evaluate` | MSE, coverage and mean width of an estimator over many worlds |

## Frozen study scripts (`experiments/`)

Each study has a data loader (`*_data.py`, which computes no outcome statistic) and an analysis script
(`*_preregistered.py`). The analysis scripts share the same structure:

| Function | Role |
|---|---|
| `flips` | flip table on the probes |
| `stratified` | stratified draw on the ecosystem difficulty; returns probe indices and stratum sizes |
| `law_strat` | exact stratified gain (Prop. 2) for one transition |
| `run_transition` / `analyse` | R replays of one transition: draw probes (uniform and stratified, independently), apply every estimator to the same probes, store squared errors, coverage, widths, decision errors |
| `main` | loops over transitions (in parallel from v3 on), writes `results/<study>/replays.csv` |
| `summarise` | applies the frozen criteria and writes `results/<study>/summary.txt` (the verdicts quoted in the paper) |

| Study | Data loader | Analysis | Results |
|---|---|---|---|
| v1 pilot (Sheng & Lu) | inside the script (`load`) | `g2_preregistered.py` | `results/g2/` |
| v2 AppWorld | inside the script (`agents_available`, `load`, `transitions`) | `v2_appworld_preregistered.py` | `results/v2/` |
| v3 SWE-bench Verified | `v3_swebench_data.py` | `v3_swebench_preregistered.py` | `results/v3/` |
| v4 SWE-bench Lite | `v4_lite_data.py` | `v4_lite_preregistered.py` | `results/v4/` |
| v5 Multi-SWE-bench, SWE-bench test and multilingual | `v5_general_data.py` | `v5_general_preregistered.py` | `results/v5/` |
| v6 LiveCodeBench, TheAgentCompany | `v6_data.py` | `v6_unified_preregistered.py` | `results/v6/` |
| v7 structure (secondary) | data of v4–v6 | `v7_structure_preregistered.py` | `results/v7/` |
| v8 patterns, multi-agent, asynchronous | `v8_data.py` | `v8_patterns_async_preregistered.py` | `results/v8/` |

The data loaders implement the rules of the paper's "Data" paragraph: `lineage` / `provider` (product lines),
`transitions` (each submission paired with the latest strictly earlier one of the same lineage) and `ecosystem`
(submissions released strictly before the new version). From v3 on, the analysis scripts accept `--fake`, which replaces outcomes
by random draws, to dry-run the pipeline before the freeze.

The column names of `replays.csv` follow `<quantity>_<estimator>_<design>`: e.g. `gain_greg_cf_s` is the gain of
cross-fitted GREG under the stratified design, `cov_reset_u` the coverage of random testing under the uniform design,
`decerr_v3_s` the keep-or-replace error of the stratified graph estimator.

## Documented, non-frozen code

| Path | Content |
|---|---|
| `src/cvt/kernel.py` | the kernel of the graph of agents (paper Section 7) |
| `tests/` | unit and statistical tests (`python -m pytest -q tests`) |
| `experiments/dev_*.py` | development scripts, run only on data already analysed; each docstring says which study it prepared |
| `experiments/g1_*.py` | synthetic validation G1 of the first estimator |
| `experiments/*exploratory*.py`, `*_descriptive.py` | analyses added after a freeze, labelled exploratory or descriptive |
| `experiments/kernel_*.py` | kernel validation (replay of 89 transitions) and case study |
| `paper/make_numbers.py`, `paper/check_numbers.py` | every number and figure of the paper, and their cross-check |
| `tools/verify_preregistration.py` | verification of the pre-registration chain |

Data paths (`/home/claude/data/...`) are hard-coded in the frozen scripts as they were run; `DATA.md` lists each
source, its commit and the expected location.
