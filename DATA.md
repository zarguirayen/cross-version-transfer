# Data sources

No benchmark output is redistributed in this repository; `results/` contains only derived statistics. The analysis
scripts read the sources below at the paths shown (frozen scripts keep their original paths so that their SHA-256
hashes stay valid; create the same paths or symbolic links).

| Study | Source | Version used | Expected path |
|---|---|---|---|
| v1 (pilot) | `github.com/jiashengsally/llm-regression-signals` (Sheng & Lu, arXiv:2608.13607) | commit `ce974d7` | `/home/claude/jiashengsally/llm-regression-signals` |
| v2 | AppWorld experiment outputs (`github.com/stonybrooknlp/appworld-leaderboard`, `leaderboard.bundle` files, unpacked with `appworld` 0.1.3) | 14 agents, splits `test_normal`, `test_challenge` | `/home/claude/data/aw_root` |
| v3, v4, v5 | `github.com/SWE-bench/experiments`, `evaluation/{verified,lite,test,multilingual}` | commit `40f164d` | `/home/claude/data/swe/experiments` |
| v5 | `github.com/multi-swe-bench/experiments`, `evaluation/<language>/verified` | commit `6a7d556` | `/home/claude/data/mswe/experiments` |
| v6 | `github.com/LiveCodeBench/submissions` | commit `6ca212e` | `/home/claude/data/LiveCodeBench_submissions` |
| v6 | `github.com/TheAgentCompany/experiments`, `evaluation/1.0.0` | commit `f9b1e94` | `/home/claude/data/TheAgentCompany_experiments` |

Only per-task outcome files are needed (`results/results.json`, `per_instance_details.json`, `metadata.yaml`,
`eval_*.json`); a sparse checkout is enough. Task-set rules (exclusions, name normalisation, removal of SWE-bench test
tasks that belong to Verified or Lite) are applied by the loaders in `experiments/` and described in each protocol.

## Study v8 (AppWorld leaderboard bundles)

The 13 agents submitted to `stonybrooknlp/appworld-leaderboard` after the AppWorld paper (snapshot of 2026-11-04) are
distributed as encrypted `leaderboard.bundle` files. Unpack each with the official package
(`pip install appworld`; `appworld.common.utils.unpack_bundle(path, out_dir, PASSWORD, SALT)` with the constants of
`appworld.common.constants`) into `/home/claude/data/aw_new/`. Do not redistribute the decrypted files (see each
bundle's `README_BEFORE_SHARING.md`); this repository only contains derived statistics.
