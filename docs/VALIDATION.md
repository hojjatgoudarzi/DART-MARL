# Standalone extraction validation (2026-11-16)

The following steps were performed **on the extracted standalone repository**, not inferred from a prior release:

- Imported the standalone `hr_mappo` module without an installed `dart_marl` package (no `dart_marl` imports in source/tests/scripts).
- Installed the editable `hr-mappo-research` entry point using the environment's installed build backend (`pip install -e . --no-build-isolation --no-deps`). A normal internet-enabled user installation is documented in README.
- Verified `hr-mappo --help` and the training CLI.
- Executed a **real one-update CPU PPO run** with raw team reward. It saved `checkpoint.pt`, `actor_only.pt`, `train_history.jsonl`, `manifest.json`, `research_readiness.json` and a live terminal log; finite recorded return and entropy. This is a software smoke, not learning evidence.
- Loaded the standalone checkpoint and completed a real 120-step actor evaluation.
- Executed two-seed benchmark preflight with Always-IDLE=0, nonzero feasible rewards, and a privileged diagnostic re-planning witness. This witness is NOT a fair decentralized comparison.
- Executed a one-seed minimal-feasible-cut stress using both Actor and Nearest; saved intervention data and summary.
- Run the 74 HR-MAPPO regression tests in six separate groups, all passing; the monolithic test run reached the environment time limit, and **is not claimed to have completed**. Test groups: core+credit 26, asynchronous/communication 8, service/curriculum 17, adaptive scenarios 11, extended recurrent/PPO 12.
- Verified a new standalone ZIP can be created from code/docs/config/tests/scripts only; no trained checkpoint, historical results dataset, guided MAPPO or global allocator is packaged.

**Not tested here:** CUDA execution on RTX 3070 Ti / RTX 5090 hardware, nine-run reproductions, statistical superiority, or new learning of causal fault-triggered task reassignment. Those tasks require independent runs and are intentionally deferred.

Additional robustness: the standalone code refuses to start a fresh training run into an existing history/checkpoint directory, preventing accidental provenance corruption. The smoke script chooses a new timestamped directory each time.
