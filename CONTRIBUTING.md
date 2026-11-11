# Contributing

Thanks for your interest in improving DART-MARL. Keep contributions focused, reproducible, and explicit about information available to the agents.

1. Open an issue or describe the intended change and expected evaluation criterion.
2. Work in a separate branch; include a minimal regression test for behavioral changes.
3. Run `python -m pytest -q` and `bash scripts/validate.sh` before a pull request.
4. For algorithm claims, include the training seeds, same-controller paired baselines, environment config hash, episode-level returns, and exact checkpoint identity.
5. Do not use an already inspected held-out test for selecting or tuning a new method; record negative results faithfully.

Avoid committing local checkpoints, large raw experimental outputs, personal paths, or proprietary third-party data. The three bundled actor-only checkpoints (two historical six-rover, one twelve-rover) are verified, intentional exceptions. The complete optimizer-bearing training checkpoint remains outside Git.
