# Security and checkpoint loading

Only load models and experiment records from sources you trust. Full PyTorch training checkpoints may include Python/NumPy objects that require unsafe pickle deserialization. The included example actors deliberately use a small weights-only format and can be read using `torch.load(..., weights_only=True)`.

`src/dart_marl/roles/evaluation.py` tries restricted weights-only loading first, but retains legacy compatibility for full locally trusted training checkpoints. **Never pass an untrusted `.pt` file to its legacy loader.** To report a security issue, contact the project maintainers privately rather than publishing exploitation details in a public issue.
