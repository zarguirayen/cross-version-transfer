# A graph of agents that remembers what it has already measured

*Idea file — Rayen Zargui, November 2026. Code, pre-registrations and results in this repository.*

Every time you ship a new version of an agent (new model, new prompt, new scaffold) you re-run the whole eval
suite, as if nothing had been learned before. But you already know a lot: how every earlier agent did on every
task. That history is a graph: nodes are agents (or models, or tools), edges are versions of the same thing, and
each node carries its results on a shared set of tasks.

**The claim.** The history in that graph is worth a measurable number of tasks, and you can compute how many:

    gain = 1 / (1 − ρ²)

where ρ² is the share of the new version's task-level variance that the graph already explains. Two forms:
ρ² = φ² with the predecessor (double sampling, Neyman 1938), and ρ² = η² with the task difficulty seen by all
earlier agents (stratified sampling). With 20 tasks chosen by the graph you get the precision of about 40–45
random tasks.

**How to use the graph (three moves).**
1. *Choose* the tasks: sort tasks by how hard earlier agents found them, cut into 20 strata, test one per stratum.
2. *Estimate* the new version's success rate: graph prediction + correction by out-of-sample residuals on the
   tested tasks (cross-fitted GREG). Unbiased even when the graph is wrong; calibrated intervals.
3. *Decide* keep-or-switch from the same 20 tasks, and say whether the decision is sure.

**Evidence (all pre-registered, frozen by git tags before looking at the data).** 219 real version transitions,
6 independent sources: model versions (Sheng & Lu; LiveCodeBench), agents on AppWorld, SWE-bench Verified, Lite,
test, multilingual, Multi-SWE-bench in 7 languages, and office agents on TheAgentCompany. The law predicts the
observed gain within 3–6 % (median) in every source. On the last two confirmations, the cross-fitted estimator
never lost to plain random testing (worst ×1.08 and ×1.45) with 94–95 % coverage of 90 % intervals. Version
decisions made with graph-chosen tasks are wrong about half as often.

**What did not work, kept in the record.** The predecessor alone adds almost nothing once the whole graph is
used. A purely model-based estimator can lose badly on versions that solve < 10 % of tasks. An unbiased estimator
is not automatically a good decision rule when two versions both solve almost nothing.

**The kernel.** `src/cvt/kernel.py` — an append-only event log, a registry of slots and versions, and five
operations: `record`, `plan`, `estimate`, `decide`, `lint`, plus one Markdown page per slot. It observes, it does
not orchestrate. Replaying 89 real SWE-bench transitions through the kernel alone reproduces the research numbers
(×2.18 median gain, 94 % coverage).

**Related work.** IRT-based efficient evaluation (tinyBenchmarks; Agent Psychometrics; Efficient Benchmarking in
Production; Task-Efficient Agent Evaluation Using Cached Traces) predicts scores from few items. What is added
here: a closed-form law for how much history is worth, calibrated intervals that survive model failure, and the
version decision.
