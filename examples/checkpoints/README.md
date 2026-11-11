# Example trained actors

Two compact **actor-only** exports of independently trained fault-robust MAPPO checkpoints, seed 202 and 404. Each contains:

- Actor parameter tensors, the seed, hidden size and engineered role-guidance gain
- The original training-environment hash and training update
- No optimizer, critic, rollout buffer, or unsafe NumPy RNG objects

These load with `torch.load(..., weights_only=True)` and are intended for **inference/demos**, not continued training. Source full-checkpoint hashes and exported checkpoint hashes are recorded in [manifest.json](manifest.json).

Use the same `configs/env/dev_roles.yaml` training configuration. The exact SHA-256 of that YAML is checked before actor inference.
